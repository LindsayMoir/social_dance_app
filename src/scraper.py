"""
scraper.py

The EventSpider handles URL extraction,
dynamic content extraction, and Google Calendar event processing.

Dependencies:
    - Scrapy, requests, pandas, yaml, logging, shutil, etc.
    - Local modules: DatabaseHandler (from db.py), LLMHandler (from llm.py),
      and credentials (from credentials.py).
"""

import base64
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup, Comment
import html
import json
import logging
import os
import pandas as pd
import re
import requests
import scrapy
from scrapy.http import TextResponse
from scrapy_playwright.page import PageMethod
from scrapy.crawler import CrawlerProcess
import shutil
import sys
import yaml
import subprocess
import csv
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.parse import parse_qs, unquote

from credentials import get_credentials
from config_runtime import get_config_path, load_config
from db import DatabaseHandler
from llm import LLMHandler
from logging_config import setup_logging
from output_paths import codex_review_path, test_output_path
from page_classifier import (
    apply_historical_routing_memory,
    classify_page,
    classify_page_with_confidence,
    evaluate_step_ownership,
    has_event_signal as classifier_has_event_signal,
    is_facebook_url as classifier_is_facebook_url,
    is_google_calendar_like_url,
    is_instagram_url as classifier_is_instagram_url,
    resolve_prompt_type,
)
from page_classifier_features import extract_html_features

# --------------------------------------------------
# Global objects initialization.
# (These globals will be used by EventSpider.)
# --------------------------------------------------
config = load_config()

# Handlers will be instantiated when needed to avoid blocking module import
_handlers_cache = None


def normalize_url_for_compare(url: str) -> str:
    """
    Normalize URL for deterministic comparisons and de-duplication.
    """
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        if netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]
        path = (parsed.path or "").rstrip("/")
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except Exception:
        return (url or "").strip().lower().rstrip("/")


def normalize_http_links(base_url: str, raw_links: list[str], limit: int | None = None) -> list[str]:
    """
    Normalize links using base_url and keep only unique http/https URLs.

    This accepts relative and protocol-relative links (e.g., '/events', '//calendar.google.com/...').
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_link in raw_links:
        if not raw_link:
            continue
        candidate = urljoin(base_url, str(raw_link).strip())
        scheme = urlparse(candidate).scheme.lower()
        if scheme not in ("http", "https"):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def extract_hycal_proxy_links(base_url: str, html_text: str) -> list[str]:
    """
    Extract Hydrogen Calendar Embed proxy URLs from rendered HTML.

    HyCal renders calendars via WordPress REST URLs like:
    /wp-json/hycal/v1/ics-proxy?url=<encoded_google_ics_url>
    """
    if not html_text:
        return []
    text = html.unescape(str(html_text))
    candidates: list[str] = []
    # Match both absolute and relative proxy URLs across possible HyCal versions/routes.
    patterns = [
        r"(?:https?://[^\s\"'<>]+/wp-json/[^\s\"'<>]*ics-proxy[^\s\"'<>]*)",
        r"(/wp-json/[^\s\"'<>]*ics-proxy[^\s\"'<>]*)",
        r"(?:https?://[^\s\"'<>]+\?[^\s\"'<>]*rest_route=[^\s\"'<>]*ics-proxy[^\s\"'<>]*)",
        r"(/[^\s\"'<>]*\?[^\s\"'<>]*rest_route=[^\s\"'<>]*ics-proxy[^\s\"'<>]*)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            candidates.append(match)
    return normalize_http_links(base_url, candidates)


def is_facebook_url(url: str) -> bool:
    """
    Return True when URL host belongs to Facebook.

    scraper.py must not crawl Facebook directly; fb.py owns Facebook crawling.
    """
    return classifier_is_facebook_url(url)


def is_calendar_candidate(url: str, calendar_roots: set[str]) -> bool:
    """
    Returns True when URL is a known calendar seed/root or a Google Calendar link.
    """
    if is_google_calendar_like_url(url):
        return True
    norm_url = normalize_url_for_compare(url)
    return any(norm_url.startswith(root) for root in calendar_roots)


def is_whitelist_candidate(url: str, whitelist_roots: set[str]) -> bool:
    """
    Returns True when URL matches or is under a whitelisted root URL.
    """
    norm_url = normalize_url_for_compare(url)
    return any(norm_url.startswith(root) for root in whitelist_roots)


def merge_seed_urls(seed_df: pd.DataFrame, whitelist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge regular crawl seeds with whitelist seeds and de-duplicate by normalized URL.
    """
    combined = pd.concat([seed_df, whitelist_df], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    for required_col in ("source", "keywords", "link"):
        if required_col not in combined.columns:
            combined[required_col] = ""
    combined["link"] = combined["link"].fillna("").astype(str).str.strip()
    combined = combined[combined["link"] != ""].copy()
    combined["_norm"] = combined["link"].map(normalize_url_for_compare)
    combined = combined.drop_duplicates(subset=["_norm"], keep="first")
    return combined.drop(columns=["_norm"])


def should_force_follow_link(parent_url: str, parent_is_whitelisted: bool, link: str, calendar_roots: set[str]) -> bool:
    """
    Determine whether crawler should force-follow a child link.

    Force follow when:
    - child link itself is whitelisted
    - child link is calendar-related
    - parent page is whitelisted and child is on same domain
    """
    if not link:
        return False
    try:
        child_is_whitelisted = db_handler.is_whitelisted_url(link)
    except Exception:
        child_is_whitelisted = False
    if child_is_whitelisted:
        return True
    if is_calendar_candidate(link, calendar_roots):
        return True
    if parent_is_whitelisted:
        try:
            return urlparse(parent_url).netloc.lower() == urlparse(link).netloc.lower()
        except Exception:
            return False
    return False


def prioritize_links_for_crawl(links: list[str], max_links: int) -> list[str]:
    """
    Prioritize event/calendar links before applying crawl cap.
    """
    if max_links <= 0:
        return []
    scored: list[tuple[int, int, str]] = []
    for idx, link in enumerate(links):
        low = link.lower()
        score = 0
        if any(token in low for token in ("calendar", "events", "event", "schedule", "social", "month", "list")):
            score += 5
        if "calendar.google.com" in low or "@group.calendar.google.com" in low or "%40group.calendar.google.com" in low:
            score += 8
        if "/events/" in low:
            score += 4
        scored.append((score, idx, link))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [link for _, _, link in scored[:max_links]]


def has_event_signal(text: str) -> bool:
    """
    Fast lexical gate to avoid expensive LLM calls on low-signal pages.
    """
    return classifier_has_event_signal(text)


def extract_visible_text_from_html(html: str, max_chars: int = 20000) -> str:
    """
    Extract visible page text while removing script/template noise.

    This reduces false positives from embedded JSON, related-thumbnail payloads,
    and other non-rendered artifacts often present in raw HTML.
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template", "svg"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        main_node = soup.select_one("main, article, [role='main'], .event, .event-content")
        text_chunks: list[str] = []
        if main_node is not None:
            text_chunks.append(" ".join(main_node.stripped_strings))
        text_chunks.append(" ".join(soup.stripped_strings))
        visible = " ".join(t for t in text_chunks if t).strip()
        if not visible:
            return ""
        return visible[:max_chars]
    except Exception:
        # Fail-soft: if parsing fails, preserve previous behavior.
        return (html or "")[:max_chars]


def _is_event_detail_url(url: str) -> bool:
    """Heuristic for single-event detail pages vs listing pages."""
    try:
        path = (urlparse(url).path or "").lower()
    except Exception:
        return False
    if path.rstrip("/").endswith("/events"):
        return False
    if any(token in path for token in ("/events/month", "/events/list", "/calendar", "/schedule", "/upcoming")):
        return False
    if any(token in path for token in ("/event/", "/events/", "/show/", "/nm_event/")):
        return True
    if re.search(r"/event(?:[/?#]|$)", path):
        return True
    return False


def classify_page_archetype(
    url: str,
    visible_text: str,
    page_links: list[str],
    calendar_sources: list[str],
    calendar_ids_count: int = 0,
) -> str:
    """
    Classify page into extraction archetypes to guide scrape strategy.
    """
    event_like_links = [
        l for l in page_links
        if any(
            token in l.lower()
            for token in ("/event/", "/events/", "/show/", "/nm_event/", "ticket", "rsvp")
        )
        or re.search(r"/event(?:[/?#]|$)", l.lower())
    ]
    classification = classify_page(
        url=url,
        visible_text=visible_text,
        page_links_count=len(event_like_links),
        calendar_sources_count=len(calendar_sources),
        calendar_ids_count=calendar_ids_count,
    )
    return classification.archetype


def should_extract_on_parent_page(archetype: str, url: str, confidence: float = 1.0) -> bool:
    """
    Decide if this page should go through event extraction directly.
    """
    if archetype in {"simple_page", "google_calendar"}:
        if archetype == "simple_page" and confidence < 0.70 and not _is_event_detail_url(url):
            return False
        return True
    if archetype == "incomplete_event":
        # Still extract when URL itself appears to be a concrete event-detail page.
        return _is_event_detail_url(url)
    return False


def max_links_to_follow_for_page(
    archetype: str,
    *,
    confidence: float,
    base_limit: int,
    url: str,
    is_whitelisted_origin: bool,
    is_calendar_root: bool,
) -> int:
    """Apply smaller crawl fanout on low-confidence, low-value parent pages."""
    if base_limit <= 0:
        return 0
    if is_whitelisted_origin or is_calendar_root or _is_event_detail_url(url):
        return base_limit
    if archetype == "incomplete_event":
        return min(base_limit, 2)
    if archetype == "simple_page" and confidence < 0.8:
        return min(base_limit, 3)
    return min(base_limit, 5)


def get_handlers():
    """Initialize and return handlers only when needed."""
    global _handlers_cache
    if _handlers_cache is None:
        db_handler = DatabaseHandler(config)
        llm_handler = LLMHandler(config_path=get_config_path())
        db_handler.set_llm_handler(llm_handler)  # Connect the LLM to the DB handler
        _handlers_cache = {
            'db_handler': db_handler,
            'llm_handler': llm_handler
        }
    return _handlers_cache

# --------------------------------------------------
# EventSpider: Handles the crawling process
# --------------------------------------------------
class EventSpider(scrapy.Spider):
    name = "event_spider"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.visited_link = set()  # Track visited URLs
        # Initialize handlers when spider starts, not at module import
        handlers = get_handlers()
        global db_handler, llm_handler
        db_handler = handlers['db_handler']
        llm_handler = handlers['llm_handler']
        self.keywords_list = llm_handler.get_keywords()

        # Load calendar URLs for special handling - these should always be processed
        calendar_urls_file = self.config.get('input', {}).get('calendar_urls', 'data/other/calendar_urls.csv')
        self.calendar_urls_set = set()
        try:
            if os.path.exists(calendar_urls_file):
                calendar_df = pd.read_csv(calendar_urls_file)
                self.calendar_urls_set = {normalize_url_for_compare(str(u)) for u in calendar_df['link'].dropna().tolist()}
                logging.info(f"__init__(): Loaded {len(self.calendar_urls_set)} calendar URLs for special handling")
        except Exception as e:
            logging.warning(f"__init__(): Could not load calendar URLs: {e}")

        # Load whitelist URLs as mandatory crawl seeds
        self.whitelist_urls_df = pd.DataFrame(columns=["source", "keywords", "link"])
        try:
            urls_dir = self.config.get("input", {}).get("urls", "data/urls")
            whitelist_path = os.path.join(urls_dir, "aaa_urls.csv")
            if os.path.exists(whitelist_path):
                wl_df = pd.read_csv(whitelist_path, dtype=str)
                for required_col in ("source", "keywords", "link"):
                    if required_col not in wl_df.columns:
                        wl_df[required_col] = ""
                wl_df["link"] = wl_df["link"].fillna("").astype(str).str.strip()
                self.whitelist_urls_df = wl_df[wl_df["link"] != ""][["source", "keywords", "link"]]
                logging.info(f"__init__(): Loaded {len(self.whitelist_urls_df)} whitelist URLs from {whitelist_path}")
        except Exception as e:
            logging.warning(f"__init__(): Could not load whitelist URLs: {e}")
        self.whitelist_roots = {
            normalize_url_for_compare(str(u))
            for u in self.whitelist_urls_df["link"].dropna().tolist()
            if str(u).strip()
        }
        self.attempted_whitelist_roots: set[str] = set()
        self.whitelist_transferred_to_fb_roots: set[str] = set()
        self.whitelist_non_text_response_roots: set[str] = set()
        crawling_cfg = self.config.get("crawling", {})
        self.scraper_download_timeout_seconds = int(
            crawling_cfg.get("scraper_download_timeout_seconds", 35) or 35
        )
        self.scraper_retry_times = int(crawling_cfg.get("scraper_retry_times", 1) or 1)
        self.scraper_priority_download_timeout_seconds = int(
            crawling_cfg.get(
                "scraper_priority_download_timeout_seconds",
                max(self.scraper_download_timeout_seconds * 3, 90),
            ) or max(self.scraper_download_timeout_seconds * 3, 90)
        )
        self.scraper_priority_retry_times = int(
            crawling_cfg.get("scraper_priority_retry_times", max(self.scraper_retry_times + 2, 3))
            or max(self.scraper_retry_times + 2, 3)
        )
        self.scraper_domain_failure_threshold = int(
            crawling_cfg.get("scraper_domain_failure_threshold", 3) or 3
        )
        self.scraper_domain_failure_window_seconds = int(
            crawling_cfg.get("scraper_domain_failure_window_seconds", 600) or 600
        )
        self.scraper_domain_cooldown_seconds = int(
            crawling_cfg.get("scraper_domain_cooldown_seconds", 600) or 600
        )
        self.domain_failure_events: dict[str, deque[datetime]] = defaultdict(deque)
        self.domain_cooldown_until: dict[str, datetime] = {}
        self.domain_cooldown_skip_count = 0
        self.domain_cooldown_trigger_count = 0
        self.domain_transient_failure_counts: Counter = Counter()
        self.domain_timeout_failure_counts: Counter = Counter()
        self.domain_exception_failure_counts: Counter = Counter()
        self.invalid_calendar_ids: set[str] = set()
        self.processed_calendar_ids: set[str] = set()
        self.page_archetype_stats: dict[str, dict[str, int]] = {}

        logging.info("\n\nscraper.py starting...")

    def _archetype_bucket(self, archetype: str) -> dict[str, int]:
        """Return mutable telemetry bucket for a page archetype."""
        key = str(archetype or "other")
        if key not in self.page_archetype_stats:
            self.page_archetype_stats[key] = {
                "pages_seen": 0,
                "parent_extraction_attempted": 0,
                "parent_extraction_succeeded": 0,
                "parent_extraction_failed": 0,
                "parent_extraction_skipped": 0,
                "child_links_followed": 0,
            }
        return self.page_archetype_stats[key]

    @staticmethod
    def _domain_for_url(url: str) -> str:
        """Return lowercase netloc for a URL."""
        try:
            return urlparse(url).netloc.lower().strip()
        except Exception:
            return ""

    def _whitelist_root_for_url(self, url: str) -> str | None:
        """Return the matching whitelist root for a URL, if any."""
        norm_url = normalize_url_for_compare(url)
        for root in getattr(self, "whitelist_roots", set()):
            if norm_url.startswith(root):
                return root
        return None

    def _mark_whitelist_status(self, url: str, status: str) -> str | None:
        """
        Track per-root whitelist status for clearer run-limit diagnostics.

        status:
            - attempted: parse() entered for this whitelist root.
            - transferred_fb: scraper intentionally handed URL to fb.py.
            - non_text: parse() got non-text response for whitelist root.
        """
        root = self._whitelist_root_for_url(url)
        if not root:
            return None

        if status == "attempted":
            if not hasattr(self, "attempted_whitelist_roots"):
                self.attempted_whitelist_roots = set()
            self.attempted_whitelist_roots.add(root)
            return root
        if status == "transferred_fb":
            if not hasattr(self, "whitelist_transferred_to_fb_roots"):
                self.whitelist_transferred_to_fb_roots = set()
            self.whitelist_transferred_to_fb_roots.add(root)
            return root
        if status == "non_text":
            if not hasattr(self, "whitelist_non_text_response_roots"):
                self.whitelist_non_text_response_roots = set()
            self.whitelist_non_text_response_roots.add(root)
            return root
        return root

    def _remaining_scraper_owned_whitelist_roots(self) -> set[str]:
        """
        Return whitelist roots still pending scraper parse attempts.

        Roots owned by fb.py are excluded so the pending count reflects
        only scraper-owned backlog.
        """
        whitelist_roots = getattr(self, "whitelist_roots", set())
        attempted = getattr(self, "attempted_whitelist_roots", set())
        transferred_fb = getattr(self, "whitelist_transferred_to_fb_roots", set())
        return whitelist_roots - attempted - transferred_fb

    def _prune_domain_failures(self, domain: str, now_ts: datetime) -> None:
        """Keep only failures that are still inside the rolling failure window."""
        events = self.domain_failure_events.get(domain)
        if not events:
            return
        cutoff = now_ts - timedelta(seconds=self.scraper_domain_failure_window_seconds)
        while events and events[0] < cutoff:
            events.popleft()

    def _domain_in_cooldown(self, url: str) -> bool:
        """True when domain is currently blocked by transient-failure circuit breaker."""
        domain = self._domain_for_url(url)
        if not domain:
            return False
        until = self.domain_cooldown_until.get(domain)
        if not until:
            return False
        now_ts = datetime.now()
        if now_ts >= until:
            self.domain_cooldown_until.pop(domain, None)
            self.domain_failure_events.pop(domain, None)
            return False
        return True

    def _record_domain_success(self, url: str) -> None:
        """Clear failure history on successful response for a domain."""
        domain = self._domain_for_url(url)
        if not domain:
            return
        self.domain_failure_events.pop(domain, None)

    def _record_domain_transient_failure(self, url: str, reason: str) -> None:
        """Track transient download failures and trigger domain cooldown when threshold is crossed."""
        domain = self._domain_for_url(url)
        if not domain:
            return
        now_ts = datetime.now()
        events = self.domain_failure_events[domain]
        events.append(now_ts)
        self._prune_domain_failures(domain, now_ts)
        if len(events) < self.scraper_domain_failure_threshold:
            return
        cooldown_until = now_ts + timedelta(seconds=self.scraper_domain_cooldown_seconds)
        current_until = self.domain_cooldown_until.get(domain)
        if current_until and current_until >= cooldown_until:
            return
        self.domain_cooldown_until[domain] = cooldown_until
        self.domain_cooldown_trigger_count += 1
        logging.warning(
            "Domain circuit breaker triggered for %s until %s after %d transient failures in %ds. Last reason: %s",
            domain,
            cooldown_until.isoformat(),
            len(events),
            self.scraper_domain_failure_window_seconds,
            reason,
        )

    @staticmethod
    def _is_transient_download_failure(failure) -> bool:
        """Return True for transient network/download failures worth circuit-breaking."""
        err_name = ""
        err_text = ""
        try:
            err_name = failure.type.__name__.lower() if getattr(failure, "type", None) else ""
            err_text = str(failure.value).lower() if getattr(failure, "value", None) else str(failure).lower()
        except Exception:
            err_text = str(failure).lower()
        markers = (
            "timeouterror",
            "responseneverreceived",
            "connectionlost",
            "tcptimedout",
            "dnslookuperror",
            "connecterror",
            "connection was refused",
            "connection reset by peer",
            "timed out",
            "timeout",
        )
        return any(marker in err_name or marker in err_text for marker in markers)

    @staticmethod
    def _is_timeout_download_failure(failure) -> bool:
        """Return True when a transient failure is timeout-related."""
        err_name = ""
        err_text = ""
        try:
            err_name = failure.type.__name__.lower() if getattr(failure, "type", None) else ""
            err_text = str(failure.value).lower() if getattr(failure, "value", None) else str(failure).lower()
        except Exception:
            err_text = str(failure).lower()
        timeout_markers = (
            "timeouterror",
            "timed out",
            "timeout",
            "tcptimedout",
        )
        return any(marker in err_name or marker in err_text for marker in timeout_markers)

    def handle_request_error(self, failure) -> None:
        """Scrapy errback to track transient domain failures and cooldown behavior."""
        request = getattr(failure, "request", None)
        req_url = request.url if request else ""
        if self._is_transient_download_failure(failure):
            domain = self._domain_for_url(req_url)
            if domain:
                self.domain_transient_failure_counts[domain] += 1
                if self._is_timeout_download_failure(failure):
                    self.domain_timeout_failure_counts[domain] += 1
                else:
                    self.domain_exception_failure_counts[domain] += 1
            self._record_domain_transient_failure(req_url, str(failure.value) if getattr(failure, "value", None) else str(failure))
        else:
            logging.warning("handle_request_error(): Non-transient request failure for %s: %s", req_url, failure)

    def closed(self, reason: str) -> None:
        """Emit circuit-breaker summary at spider shutdown for reporting."""
        active_blocks = 0
        now_ts = datetime.now()
        for until in self.domain_cooldown_until.values():
            if until > now_ts:
                active_blocks += 1
        logging.info(
            "domain_circuit_breaker_summary: triggers=%d skips=%d active_blocks=%d reason=%s",
            self.domain_cooldown_trigger_count,
            self.domain_cooldown_skip_count,
            active_blocks,
            reason,
        )
        top_transient = self.domain_transient_failure_counts.most_common(10)
        top_timeout = self.domain_timeout_failure_counts.most_common(10)
        top_exception = self.domain_exception_failure_counts.most_common(10)
        logging.info("domain_transient_failures_top: %s", top_transient)
        logging.info("domain_timeout_failures_top: %s", top_timeout)
        logging.info("domain_exception_failures_top: %s", top_exception)
        summary = {
            "reason": str(reason or ""),
            "archetypes": self.page_archetype_stats,
        }
        logging.info("scraper_archetype_summary: %s", json.dumps(summary, ensure_ascii=True, sort_keys=True))

    def _build_playwright_request_meta(self, high_priority: bool = False) -> dict:
        """
        Build per-request Playwright metadata.
        High-priority requests (whitelist/calendar/forced-follow) get more time and retries.
        """
        wait_ms = int(self.config.get("crawling", {}).get("scraper_post_load_wait_ms", 1000) or 1000)
        meta = {
            "playwright": True,
            "playwright_page_methods": [
                PageMethod("wait_for_selector", "body"),
                PageMethod("wait_for_load_state", "networkidle"),
                PageMethod("wait_for_timeout", wait_ms),
            ],
        }
        if high_priority:
            meta["download_timeout"] = self.scraper_priority_download_timeout_seconds
            meta["max_retry_times"] = self.scraper_priority_retry_times
        return meta


    async def start(self):
        """
        Generate start requests from URLs either in the DB or CSV files.

        This method replaces the deprecated start_requests() for Scrapy 2.13+ compatibility.
        Using async start() provides better support for async operations and is the recommended
        approach for modern Scrapy implementations.
        """
        conn = db_handler.get_db_connection()
        if conn is None:
            raise ConnectionError("Failed to connect to the database in start().")
        logging.info(f"def start(): Connected to the database: {conn}")

        if self.config['startup']['use_db']:
            query = "SELECT * FROM urls WHERE relevant = true;"
            urls_df = pd.read_sql_query(query, conn)
        else:
            urls_dir = self.config['input']['urls']
            csv_files = [os.path.join(urls_dir, f) for f in os.listdir(urls_dir) if f.endswith('.csv')]
            dataframes = [pd.read_csv(file) for file in csv_files]
            urls_df = pd.concat(dataframes, ignore_index=True)

        # Always include whitelist seeds, even in DB mode.
        urls_df = merge_seed_urls(urls_df, self.whitelist_urls_df)

        for _, row in urls_df.iterrows():
            source = row.get('source', '')
            keywords = row.get('keywords', '')
            url = row.get('link', '')
            if not url:
                continue

            # Whitelist check
            is_whitelisted_seed = is_whitelist_candidate(url, self.whitelist_roots)
            try:
                is_whitelisted = is_whitelisted_seed or db_handler.is_whitelisted_url(url)
            except Exception:
                is_whitelisted = is_whitelisted_seed

            # Facebook URLs are owned by fb.py and must never be crawled by scraper.py.
            if is_facebook_url(url):
                logging.info("start(): Skipping Facebook URL (owned by fb.py): %s", url)
                self._mark_whitelist_status(url, "transferred_fb")
                child_row = [url, '', source, [], False, 1, datetime.now()]
                db_handler.write_url_to_db(child_row)
                continue

            # Skip Instagram URLs unless whitelisted.
            if 'instagram.com' in url.lower() and not is_whitelisted:
                logging.info(f"start(): Skipping social media URL (ig): {url}")
                child_row = [url, '', source, [], False, 1, datetime.now()]
                db_handler.write_url_to_db(child_row)
                continue

            if db_handler.avoid_domains(url) and not is_whitelisted:
                logging.info(f"start(): Skipping blacklisted URL {url}.")
                continue

            # Special handling for calendar URLs - always process them regardless of historical relevancy
            norm_url = normalize_url_for_compare(url)
            is_calendar_url = is_calendar_candidate(norm_url, self.calendar_urls_set)
            if is_calendar_url:
                logging.info(f"start(): Processing calendar URL {url} (bypassing historical relevancy)")
            elif is_whitelisted_seed:
                logging.info(f"start(): Processing whitelist URL {url} (bypassing historical relevancy)")
            elif not db_handler.should_process_url(url):
                try:
                    if is_whitelisted:
                        # csv imported at module level
                        with open(test_output_path('skipped_whitelist.csv'), 'a', newline='') as f:
                            w = csv.writer(f)
                            w.writerow([datetime.now().isoformat(), url, source, 'history_gate'])
                except Exception:
                    pass
                logging.info(f"start(): Skipping URL {url} based on historical relevancy.")
                continue

            logging.info(f"start(): Starting crawl for URL: {url}")
            if self._domain_in_cooldown(url):
                self.domain_cooldown_skip_count += 1
                logging.info("start(): Skipping URL in active domain cooldown: %s", url)
                continue
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_request_error,
                cb_kwargs={'keywords': keywords, 'source': source, 'url': url},
                priority=1000 if is_whitelisted_seed else 0,
                meta=self._build_playwright_request_meta(
                    high_priority=(is_whitelisted_seed or is_calendar_url)
                ),
            )


    def parse(self, response, keywords, source, url):
        """
        1) Render page via Playwright and get HTML text.
        2) Identify keywords in the page and run LLM to decide relevance.
        3) Record the URL in the database (with metadata).
        4) Extract <a> links and iframe/calendar URLs.
        5) Fetch Google Calendar events where found.
        6) Filter out unwanted links, record them, and follow remaining links.
        """
        # Track whitelist attempt immediately when parse starts.
        matched_whitelist_root = self._mark_whitelist_status(url, "attempted")
        routing_decision = evaluate_step_ownership(url, current_step="scraper.py")

        # Skip non-text responses (e.g., images, PDFs, etc.)
        if not isinstance(response, TextResponse):
            if matched_whitelist_root:
                self._mark_whitelist_status(url, "non_text")
                logging.info(
                    "def parse(): Whitelist URL returned non-text response; root=%s url=%s",
                    matched_whitelist_root,
                    url,
                )
            try:
                db_handler.write_url_scrape_metric(
                    {
                        "run_id": os.getenv("DS_RUN_ID", "na"),
                        "step_name": os.getenv("DS_STEP_NAME", "scraper"),
                        "link": url,
                        "parent_url": "",
                        "source": source,
                        "keywords": [],
                        "archetype": "other",
                        "extraction_attempted": False,
                        "extraction_succeeded": False,
                        "extraction_skipped": True,
                        "decision_reason": "non_text_response",
                        "handled_by": "scraper.py",
                        "routing_reason": "non_text_response",
                        "classification_confidence": None,
                        "classification_stage": "non_text",
                        "classification_features_json": None,
                        "links_discovered": 0,
                        "links_followed": 0,
                        "time_stamp": datetime.now(),
                    }
                )
            except Exception:
                pass
            return
        self._record_domain_success(response.url)
        
        try:
            is_whitelisted_origin = db_handler.is_whitelisted_url(url)
        except Exception:
            is_whitelisted_origin = False
        is_whitelisted_origin = is_whitelisted_origin or is_whitelist_candidate(url, self.whitelist_roots)
        if is_whitelisted_origin and matched_whitelist_root is None:
            # Backstop for URLs considered whitelisted via DB lookup only.
            self._mark_whitelist_status(url, "attempted")
        if not routing_decision.allow:
            if routing_decision.owner_step == "fb.py" and matched_whitelist_root:
                self._mark_whitelist_status(url, "transferred_fb")
            child_row = [
                url,
                "",
                source,
                [],
                False,
                1,
                datetime.now(),
                routing_decision.routing_reason,
            ]
            db_handler.write_url_to_db(child_row)
            logging.info(
                "def parse(): Skipping URL owned by %s (%s): %s",
                routing_decision.owner_step,
                routing_decision.routing_reason,
                url,
            )
            try:
                db_handler.write_url_scrape_metric(
                    {
                        "run_id": os.getenv("DS_RUN_ID", "na"),
                        "step_name": os.getenv("DS_STEP_NAME", "scraper"),
                        "link": url,
                        "parent_url": "",
                        "source": source,
                        "keywords": [],
                        "archetype": routing_decision.classification.archetype,
                        "extraction_attempted": False,
                        "extraction_succeeded": False,
                        "extraction_skipped": True,
                        "decision_reason": routing_decision.routing_reason,
                        "handled_by": routing_decision.owner_step,
                        "routing_reason": routing_decision.routing_reason,
                        "classification_confidence": None,
                        "classification_stage": "routing_skip",
                        "classification_features_json": None,
                        "links_discovered": 0,
                        "links_followed": 0,
                        "time_stamp": datetime.now(),
                    }
                )
            except Exception:
                pass
            return
        
        # 1) Build visible page text (avoid raw-HTML/script noise in LLM extraction).
        extracted_text = extract_visible_text_from_html(response.text)

        # 2) Pre-extract links/calendars so archetype routing can decide page-level extraction.
        raw_links = response.css('a::attr(href)').getall()
        page_links_all = normalize_http_links(
            response.url,
            raw_links,
        )
        page_links = prioritize_links_for_crawl(page_links_all, self.config['crawling']['max_website_urls'])
        logging.info(f"def parse(): Found {len(page_links)} links on {response.url}")

        iframe_links = normalize_http_links(response.url, response.css('iframe::attr(src)').getall())
        calendar_emails = re.findall(
            r'"gcal"\s*:\s*"([A-Za-z0-9_.+-]+@group\.calendar\.google\.com)"',
            response.text
        )
        calendar_anchor_links = [link for link in page_links if is_google_calendar_like_url(link)]
        hycal_proxy_links = extract_hycal_proxy_links(response.url, response.text)
        calendar_sources = iframe_links + calendar_anchor_links + hycal_proxy_links
        calendar_ids: set[str] = {
            c for c in calendar_emails if self.is_valid_calendar_id(c, allow_gmail=False)
        }
        # Only trust high-confidence group calendar IDs from page text.
        calendar_ids.update(self.extract_calendar_ids(extracted_text, allow_gmail=False))
        # Some calendar embeds exist only in script/data attributes, not visible page text.
        calendar_ids.update(self.extract_calendar_ids(response.text, allow_gmail=False))

        event_like_links = [
            l for l in page_links
            if any(
                token in l.lower()
                for token in ("/event/", "/events/", "/show/", "/nm_event/", "ticket", "rsvp")
            )
            or re.search(r"/event(?:[/?#]|$)", l.lower())
        ]
        html_features = extract_html_features(response.text, url=url)
        class_decision = classify_page_with_confidence(
            url=url,
            visible_text=extracted_text,
            html_features=html_features,
            page_links_count=len(event_like_links),
            calendar_sources_count=len(calendar_sources),
            calendar_ids_count=len(calendar_ids),
        )
        memory_hint = None
        if class_decision.stage != "rule":
            memory_hint = db_handler.get_historical_classifier_memory(url)
            class_decision = apply_historical_routing_memory(
                class_decision,
                memory_hint=memory_hint,
            )
        page_archetype = class_decision.classification.archetype
        archetype_bucket = self._archetype_bucket(page_archetype)
        archetype_bucket["pages_seen"] += 1
        extract_parent_page = should_extract_on_parent_page(
            page_archetype,
            url,
            confidence=class_decision.confidence,
        )
        logging.info(
            "def parse(): page_archetype=%s confidence=%.2f stage=%s extract_parent_page=%s url=%s",
            page_archetype,
            class_decision.confidence,
            class_decision.stage,
            extract_parent_page,
            url,
        )
        history_reuse = db_handler.maybe_reuse_static_event_detail_from_history(url=url)
        if history_reuse.get("reused"):
            time_stamp = datetime.now()
            decision_reason = str(history_reuse.get("reason") or "history_reuse_static_event_detail")
            history_event_count = int(history_reuse.get("event_count", 0) or 0)
            url_row = [url, "", source, keywords, True, 1, time_stamp, decision_reason]
            db_handler.write_url_to_db(url_row)
            history_features = dict(class_decision.features)
            history_features["history_reuse"] = history_reuse
            try:
                db_handler.write_url_scrape_metric(
                    {
                        "run_id": os.getenv("DS_RUN_ID", "na"),
                        "step_name": os.getenv("DS_STEP_NAME", "scraper"),
                        "link": url,
                        "parent_url": "",
                        "source": source,
                        "keywords": keywords,
                        "archetype": page_archetype,
                        "extraction_attempted": False,
                        "extraction_succeeded": False,
                        "extraction_skipped": True,
                        "decision_reason": decision_reason,
                        "handled_by": "scraper.py",
                        "routing_reason": decision_reason,
                        "classification_confidence": 1.0,
                        "classification_stage": "history_reuse",
                        "classification_owner_step": class_decision.classification.owner_step,
                        "classification_subtype": class_decision.classification.subtype,
                        "classification_features_json": json.dumps(history_features, default=str),
                        "events_written": history_event_count,
                        "links_discovered": 0,
                        "links_followed": 0,
                        "time_stamp": time_stamp,
                    }
                )
            except Exception as e:
                logging.warning("def parse(): Failed recording history-reuse metric for %s: %s", url, e)
            logging.info(
                "def parse(): Reused static event-detail URL from history and skipped extraction: %s (%s)",
                url,
                history_reuse.get("history_kind"),
            )
            return

        # 3) Keyword & LLM logic
        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
        relevant    = False
        parent_url  = ''
        crawl_try   = 1
        time_stamp  = datetime.now()
        extraction_attempted = False
        extraction_succeeded = False
        extraction_skipped = False
        decision_reason = ""
        events_written = 0
        # build the initial record for this URL
        url_row = [url, parent_url, source, found_keywords, relevant, crawl_try, time_stamp]

        if found_keywords:
            logging.info(f"def parse(): Found keywords for URL {url}: {found_keywords}")
            should_run_llm = (
                extract_parent_page
                and (
                    is_whitelisted_origin
                    or is_calendar_candidate(url, self.calendar_urls_set)
                    or has_event_signal(extracted_text)
                )
            )
            if should_run_llm:
                extraction_attempted = True
                archetype_bucket["parent_extraction_attempted"] += 1
                # Use centralized prompt resolution for consistency across scrapers.
                prompt_type = resolve_prompt_type(url, fallback_prompt_type="default")
                llm_result = llm_handler.process_llm_response(
                    url,
                    parent_url,
                    extracted_text,
                    source,
                    keywords,
                    prompt_type,
                )
                if llm_result:
                    extraction_succeeded = True
                    decision_reason = "llm_positive"
                    events_written += int(getattr(llm_result, "events_written", 1))
                    archetype_bucket["parent_extraction_succeeded"] += 1
                    # mark as relevant
                    url_row[4] = True
                    db_handler.write_url_to_db(url_row)
                    logging.info(f"def parse(): URL {url} marked as relevant (LLM positive).")
                else:
                    decision_reason = "llm_negative"
                    archetype_bucket["parent_extraction_failed"] += 1
                    db_handler.write_url_to_db(url_row)
                    logging.info(f"def parse(): URL {url} marked as irrelevant (LLM negative).")
            else:
                extraction_skipped = True
                decision_reason = f"skip_parent_extraction_{page_archetype}"
                archetype_bucket["parent_extraction_skipped"] += 1
                db_handler.write_url_to_db(url_row)
                logging.info(
                    "def parse(): URL %s skipped LLM (archetype=%s, extract_parent_page=%s, low_signal_or_listing_page).",
                    url,
                    page_archetype,
                    extract_parent_page,
                )
        else:
            extraction_skipped = True
            decision_reason = "no_keywords"
            archetype_bucket["parent_extraction_skipped"] += 1
            db_handler.write_url_to_db(url_row)
            logging.info(f"def parse(): URL {url} marked as irrelevant (no keywords).")

        # 4) Process iframes & extract Google Calendar addresses

        for cal_url in calendar_sources:
            # URLs discovered from iframe/embed/calendar links may legitimately reference gmail calendars.
            extracted_ids = self.extract_calendar_ids(cal_url, allow_gmail=True)
            if extracted_ids:
                calendar_ids.update(extracted_ids)
                continue
            events_written += self.fetch_google_calendar_events(cal_url, url, source, keywords)

        if calendar_ids:
            for calendar_id in sorted(calendar_ids):
                events_written += self.process_calendar_id(calendar_id, response.url, url, source, keywords)

        if calendar_sources or calendar_ids:
            # mark the page itself as relevant if calendar events fetched
            url_row = [url, "", source, found_keywords, True, crawl_try, time_stamp]
            db_handler.write_url_to_db(url_row)

        # 5) Filter unwanted links and record them
        dynamic_link_limit = max_links_to_follow_for_page(
            page_archetype,
            confidence=class_decision.confidence,
            base_limit=int(self.config['crawling']['max_website_urls']),
            url=url,
            is_whitelisted_origin=is_whitelisted_origin,
            is_calendar_root=is_calendar_candidate(url, self.calendar_urls_set),
        )
        page_links = prioritize_links_for_crawl(page_links, dynamic_link_limit)
        all_links      = {url} | set(page_links)
        filtered_links = set()
        for link in all_links:
            low = link.lower()
            try:
                wl = db_handler.is_whitelisted_url(link)
            except Exception:
                wl = False
            if is_facebook_url(link):
                child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
                db_handler.write_url_to_db(child_row)
                logging.info("def parse(): Recorded Facebook URL for fb.py ownership: %s", link)
                continue

            if classifier_is_instagram_url(link):
                if wl:
                    logging.info(f"def parse(): Whitelisted social URL kept for crawl: {link}")
                    filtered_links.add(link)
                    continue
                child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
                db_handler.write_url_to_db(child_row)
                logging.info(f"def parse(): Recorded unwanted social URL: {link}")
                continue
            filtered_links.add(link)

        # 6) Follow each remaining link with Playwright rendering
        links_followed_count = 0
        for link in filtered_links:

            # Check urls to see if they should be scraped

            force_follow = should_force_follow_link(url, is_whitelisted_origin, link, self.calendar_urls_set)

            if db_handler.avoid_domains(link) and not force_follow:
                logging.info(f"parse: Skipping blacklisted URL {link}.")
                continue

            if not force_follow and not db_handler.should_process_url(link):
                logging.info(f"def eventbrite_search(): Skipping URL {link} based on historical relevancy.")
                continue

            # Skip if link has already been visited
            if link in self.visited_link:
                continue

            link_reuse = db_handler.maybe_reuse_static_event_detail_from_history(url=link)
            if link_reuse.get("reused"):
                link_classification = classify_page(url=link)
                child_time_stamp = datetime.now()
                child_reason = str(link_reuse.get("reason") or "history_reuse_static_event_detail")
                child_event_count = int(link_reuse.get("event_count", 0) or 0)
                self.visited_link.add(link)
                child_row = [link, url, source, found_keywords, True, 1, child_time_stamp, child_reason]
                db_handler.write_url_to_db(child_row)
                try:
                    db_handler.write_url_scrape_metric(
                        {
                            "run_id": os.getenv("DS_RUN_ID", "na"),
                            "step_name": os.getenv("DS_STEP_NAME", "scraper"),
                            "link": link,
                            "parent_url": url,
                            "source": source,
                            "keywords": found_keywords,
                            "archetype": link_classification.archetype,
                            "extraction_attempted": False,
                            "extraction_succeeded": False,
                            "extraction_skipped": True,
                            "decision_reason": child_reason,
                            "handled_by": "scraper.py",
                            "routing_reason": child_reason,
                            "classification_confidence": 1.0,
                            "classification_stage": "history_reuse",
                            "classification_owner_step": link_classification.owner_step,
                            "classification_subtype": link_classification.subtype,
                            "classification_features_json": json.dumps({"history_reuse": link_reuse}, default=str),
                            "events_written": child_event_count,
                            "links_discovered": 0,
                            "links_followed": 0,
                            "time_stamp": child_time_stamp,
                        }
                    )
                except Exception as e:
                    logging.warning("def parse(): Failed recording child history-reuse metric for %s: %s", link, e)
                logging.info(
                    "def parse(): Reused child static event-detail URL from history and skipped request: %s (%s)",
                    link,
                    link_reuse.get("history_kind"),
                )
                continue

            self.visited_link.add(link)

            # record the child link before crawling
            child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)

            if len(self.visited_link) >= self.config['crawling']['urls_run_limit']:
                remaining_scraper_owned_roots = self._remaining_scraper_owned_whitelist_roots()
                if remaining_scraper_owned_roots and not force_follow:
                    transferred_fb_count = len(getattr(self, "whitelist_transferred_to_fb_roots", set()))
                    non_text_count = len(getattr(self, "whitelist_non_text_response_roots", set()))
                    logging.info(
                        "parse(): URL run limit reached with %d scraper-owned whitelist roots still unattempted "
                        "(fb_owned=%d, non_text=%d); skipping non-whitelist link: %s",
                        len(remaining_scraper_owned_roots),
                        transferred_fb_count,
                        non_text_count,
                        link,
                    )
                    continue
                logging.info(f"parse(): Reached URL run limit ({self.config['crawling']['urls_run_limit']}); stopping crawler.")
                raise scrapy.exceptions.CloseSpider(reason="URL run limit reached")

            logging.info(f"def parse(): Crawling next URL: {link}")
            if self._domain_in_cooldown(link):
                self.domain_cooldown_skip_count += 1
                logging.info("parse(): Skipping link in active domain cooldown: %s", link)
                continue
            archetype_bucket["child_links_followed"] += 1
            links_followed_count += 1
            yield scrapy.Request(
                url=link,
                callback=self.parse,
                errback=self.handle_request_error,
                cb_kwargs={'keywords': keywords, 'source': source, 'url': link},
                priority=800 if force_follow else 0,
                meta=self._build_playwright_request_meta(high_priority=force_follow),
            )

        try:
            db_handler.write_url_scrape_metric(
                {
                    "run_id": os.getenv("DS_RUN_ID", "na"),
                    "step_name": os.getenv("DS_STEP_NAME", "scraper"),
                    "link": url,
                    "parent_url": parent_url,
                    "source": source,
                    "keywords": found_keywords,
                    "archetype": page_archetype,
                    "extraction_attempted": extraction_attempted,
                    "extraction_succeeded": extraction_succeeded,
                    "extraction_skipped": extraction_skipped,
                    "decision_reason": decision_reason or "unknown",
                    "handled_by": "scraper.py",
                    "routing_reason": decision_reason or "unknown",
                    "classification_confidence": class_decision.confidence,
                    "classification_stage": class_decision.stage,
                    "classification_owner_step": class_decision.classification.owner_step,
                    "classification_subtype": class_decision.classification.subtype,
                    "classification_features_json": json.dumps(class_decision.features),
                    "events_written": events_written,
                    "links_discovered": len(page_links),
                    "links_followed": links_followed_count,
                    "time_stamp": time_stamp,
                }
            )
        except Exception as e:
            logging.warning("def parse(): Failed recording url_scrape_metrics for %s: %s", url, e)


    def fetch_google_calendar_events(self, calendar_url, url, source, keywords):
        """
        Fetch and process events from a Google Calendar.
        """
        logging.info(f"def fetch_google_calendar_events(): Inputs - calendar_url: {calendar_url}, URL: {url}, source: {source}, keywords: {keywords}")
        candidate_urls = self._expand_calendar_url_candidates(calendar_url)
        if not candidate_urls:
            logging.info(
                "def fetch_google_calendar_events(): Skipping non-calendar-like URL: %s",
                calendar_url,
            )
            return 0
        events_written = 0
        for candidate_url in candidate_urls:
            calendar_ids = self.extract_calendar_ids(candidate_url, allow_gmail=True)
            if not calendar_ids:
                if self.is_valid_calendar_id(candidate_url, allow_gmail=True):
                    calendar_ids = [candidate_url]
                else:
                    decoded_calendar_id = self.decode_calendar_id(candidate_url)
                    if decoded_calendar_id:
                        calendar_ids = [decoded_calendar_id]
                    else:
                        logging.warning(
                            "def fetch_google_calendar_events(): Failed to extract valid Calendar ID from %s (source=%s)",
                            candidate_url,
                            calendar_url,
                        )
                        continue
            for calendar_id in calendar_ids:
                events_written += self.process_calendar_id(calendar_id, candidate_url, url, source, keywords)
        return events_written


    @staticmethod
    def _is_google_calendar_like_url(candidate_url: str) -> bool:
        """Return True only for URLs that plausibly contain Google Calendar data/IDs."""
        low = (candidate_url or "").lower()
        try:
            parsed = urlparse(str(candidate_url or "").strip())
            path_low = (parsed.path or "").lower()
            query_low = (parsed.query or "").lower()
            # Route-agnostic HyCal proxy detection (covers /wp-json/.../ics-proxy and rest_route forms).
            if "/wp-json/" in path_low and "ics-proxy" in path_low:
                return True
            if "rest_route=" in query_low and "ics-proxy" in query_low:
                return True
        except Exception:
            pass
        return any(
            token in low
            for token in (
                "calendar.google.com",
                "google.com/calendar",
                "/calendar/ical/",
                "@group.calendar.google.com",
                "%40group.calendar.google.com",
                "@gmail.com",
                "%40gmail.com",
                "src=",
                ".ics",
            )
        )

    @staticmethod
    def _extract_hycal_embedded_urls(candidate_url: str) -> list[str]:
        """
        Extract embedded calendar URLs from HyCal proxy endpoint query params.
        """
        raw = str(candidate_url or "").strip()
        if not raw:
            return []
        try:
            parsed = urlparse(raw)
            low_path = (parsed.path or "").lower()
            query_map = parse_qs(parsed.query)
            rest_route_values = [str(v).lower() for v in query_map.get("rest_route", []) if str(v)]
            has_proxy_path = ("/wp-json/" in low_path and "ics-proxy" in low_path)
            has_proxy_rest_route = any("ics-proxy" in value for value in rest_route_values)
            if not (has_proxy_path or has_proxy_rest_route):
                return []
            embedded_values = query_map.get("url", [])
            for alias_key in ("ics", "feed", "calendar_url", "calendar", "src"):
                embedded_values.extend(query_map.get(alias_key, []))
            decoded: list[str] = []
            for value in embedded_values:
                text = unquote(str(value or "").strip())
                if text:
                    decoded.append(text)
            return decoded
        except Exception:
            return []

    def _expand_calendar_url_candidates(self, calendar_url: str) -> list[str]:
        """
        Expand one calendar URL into candidate URLs for ID extraction.

        Includes HyCal proxy unwrapping where needed.
        """
        base = str(calendar_url or "").strip()
        if not base:
            return []
        candidates: list[str] = []
        if self._is_google_calendar_like_url(base):
            candidates.append(base)
        for embedded in self._extract_hycal_embedded_urls(base):
            if self._is_google_calendar_like_url(embedded):
                candidates.append(embedded)

        unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = candidate.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique


    def extract_calendar_ids(self, calendar_url, allow_gmail: bool = False):
        if not calendar_url:
            return []

        text = str(calendar_url)
        # Handle escaped JS strings like "https:\/\/calendar.google.com\/calendar\/ical\/..."
        text = text.replace("\\/", "/")
        # Unquote URL-encoded values (including nested query-parameter URLs).
        for _ in range(2):
            decoded = unquote(text)
            if decoded == text:
                break
            text = decoded

        patterns = [
            # Standard Google Calendar embed src parameter
            r'src=([^&]+%40group\.calendar\.google\.com)',
            # Calendar IDs directly present in text
            r'([A-Za-z0-9_.+-]+(?:%40|@)group\.calendar\.google\.com)',
            # Public ICS feed URLs
            r'/calendar/ical/([^/]+(?:%40|@)group\.calendar\.google\.com)/public',
        ]
        if allow_gmail:
            patterns.extend([
                r'src=([^&]+%40gmail\.com)',
                r'([A-Za-z0-9_.+-]+(?:%40|@)gmail\.com)',
                r'/calendar/ical/([^/]+(?:%40|@)gmail\.com)/public',
            ])

        ids: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                candidate = str(match).replace('%40', '@')
                if self.is_valid_calendar_id(candidate, allow_gmail=allow_gmail):
                    ids.add(candidate)
        return list(ids)
    

    def decode_calendar_id(self, calendar_url):
        try:
            start_marker_idx = calendar_url.find("src=")
            if start_marker_idx == -1:
                return None
            start_idx = start_marker_idx + 4
            end_idx = calendar_url.find("&", start_idx)
            calendar_id = calendar_url[start_idx:end_idx] if end_idx != -1 else calendar_url[start_idx:]
            if self.is_valid_calendar_id(calendar_id, allow_gmail=True):
                return calendar_id
            if not calendar_id:
                return None
            padded_id = calendar_id + '=' * ((4 - len(calendar_id) % 4) % 4)
            decoded = base64.b64decode(padded_id).decode('utf-8', errors='ignore')
            if self.is_valid_calendar_id(decoded, allow_gmail=True):
                return decoded
            logging.warning(f"def decode_calendar_id(): Decoded ID is not valid: {decoded}")
            return None
        except Exception as e:
            logging.warning(f"def decode_calendar_id(): Exception for {calendar_url} - {e}")
            return None
        

    def is_valid_calendar_id(self, calendar_id, allow_gmail: bool = False):
        suffix_pattern = r'(group\.calendar\.google\.com|gmail\.com)' if allow_gmail else r'(group\.calendar\.google\.com)'
        pattern = re.compile(rf'^[a-zA-Z0-9_.+-]+@{suffix_pattern}$')
        return bool(pattern.fullmatch(calendar_id))
    

    def process_calendar_id(self, calendar_id, calendar_url, url, source, keywords):
        if calendar_id in self.processed_calendar_ids:
            logging.info(
                "def process_calendar_id(): Skipping already processed calendar_id in this run: %s",
                calendar_id,
            )
            return 0
        if calendar_id in self.invalid_calendar_ids:
            logging.info(
                "def process_calendar_id(): Skipping previously invalid calendar_id: %s",
                calendar_id,
            )
            return 0
        logging.info(f"def process_calendar_id(): Processing calendar_id: {calendar_id} from {calendar_url}")
        events_df = self.get_calendar_events(calendar_id)
        self.processed_calendar_ids.add(calendar_id)
        if not events_df.empty:
            logging.info(f"def process_calendar_id(): Found {len(events_df)} events for calendar_id: {calendar_id}")
            logging.info(f"def process_calendar_id(): Event columns: {list(events_df.columns)}")
            logging.info(f"def process_calendar_id(): Sample event data:\n{events_df.head(1).to_dict('records')}")
            db_handler.write_events_to_db(events_df, calendar_id, calendar_url, source, keywords)
            return int(len(events_df))
        else:
            logging.warning(f"def process_calendar_id(): No events found for calendar_id: {calendar_id}")
            return 0


    def get_calendar_events(self, calendar_id):
        _, api_key, _ = get_credentials('Google')
        days_ahead = self.config['date_range']['days_ahead']
        api_url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        params = {
            "key": api_key,
            "singleEvents": "true",
            "timeMin": datetime.now(timezone.utc).isoformat(),
            "timeMax": (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat(),
            "fields": "items, nextPageToken",
            "maxResults": 100
        }
        all_events = []
        while True:
            response = requests.get(api_url, params=params, timeout=30)  # 30 second timeout
            if response.status_code == 200:
                data = response.json()
                all_events.extend(data.get("items", []))
                if not data.get("nextPageToken"):
                    break
                params["pageToken"] = data.get("nextPageToken")
            else:
                if response.status_code == 404:
                    self.invalid_calendar_ids.add(calendar_id)
                    logging.warning(
                        "def get_calendar_events(): Calendar not found (404) for calendar_id: %s",
                        calendar_id,
                    )
                else:
                    logging.error(f"def get_calendar_events(): Error {response.status_code} for calendar_id: {calendar_id}")
                break
        df = pd.json_normalize(all_events)
        if df.empty:
            logging.info(f"def get_calendar_events(): No events found for calendar_id: {calendar_id}")
            return df
        return self.clean_calendar_events(df)
    

    def clean_calendar_events(self, df):
        df = df.copy()
        required_columns = ['htmlLink', 'summary', 'start.date', 'end.date', 'location', 'start.dateTime', 'end.dateTime', 'description']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        df['start.dateTime'] = df['start.dateTime'].fillna(df['start.date'])
        df['end.dateTime'] = df['end.dateTime'].fillna(df['end.date'])
        df.drop(columns=['start.date', 'end.date'], inplace=True)
        df = df[['htmlLink', 'summary', 'location', 'start.dateTime', 'end.dateTime', 'description']]
        df['Price'] = pd.to_numeric(df['description'].str.extract(r'\$(\d{1,5})')[0], errors='coerce')
        df['description'] = df['description'].apply(lambda x: re.sub(r'\s{2,}', ' ', re.sub(r'<[^>]*>', ' ', str(x))).strip()
                                                    ).str.replace('&#39;', "'").str.replace("you're", "you are")
        def split_datetime(dt_str):
            if 'T' in dt_str:
                date_str, time_str = dt_str.split('T')
                return date_str, time_str[:8]
            return dt_str, None
        df['Start_Date'], df['Start_Time'] = zip(*df['start.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df['End_Date'], df['End_Time'] = zip(*df['end.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df.drop(columns=['start.dateTime', 'end.dateTime'], inplace=True)
        df = df.rename(columns={'htmlLink': 'URL', 'summary': 'Name_of_the_Event', 'location': 'Location',
                                  'description': 'Description'})
        event_type_map = {
            'class': 'class',
            'classes': 'class',
            'dance': 'social dance',
            'dancing': 'social dance',
            'weekend': 'workshop',
            'workshop': 'workshop',
            'rehearsal': 'rehearsal'
        }
        def determine_event_type(row):
            name = row.get('Name_of_the_Event') or ''
            description = row.get('Description') or ''
            combined = f"{name} {description}".lower()
            if 'class' in combined and 'dance' in combined:
                return 'class, social dance'
            for word, event_type in event_type_map.items():
                if word in combined:
                    return event_type
            return 'other'
        df['Type_of_Event'] = df.apply(determine_event_type, axis=1)
        df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.date
        df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce').dt.date
        df['Day_of_Week'] = pd.to_datetime(df['Start_Date']).dt.day_name()
        df = df[['URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week', 'Start_Date', 
                 'End_Date', 'Start_Time', 'End_Time', 'Price', 'Location', 'Description']]
        df = df.sort_values(by=['Start_Date', 'Start_Time']).reset_index(drop=True)
        return df
    

    def run_crawler(self):
        """
        Runs the crawler inline via Scrapy's CrawlerProcess (no external subprocess).
        """
        logging.info("def run_crawler(): Starting crawler in-process.")
        # Build log_file name
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        logging_file = f"logs/{script_name}_log.txt"

        process = CrawlerProcess(settings={
            "LOG_FILE": logging_file,
            "LOG_LEVEL": "INFO",
            "LOG_FORMAT": (
                "%(asctime)s [%(name)s] %(levelname)s: "
                f"[run_id={os.getenv('DS_RUN_ID', 'na')}] "
                f"[step={os.getenv('DS_STEP_NAME', 'scraper')}] %(message)s"
            ),
            "DEPTH_LIMIT": self.config['crawling']['depth_limit'],
            "FEEDS": {
                codex_review_path("output.json"): {"format": "json"}
            },
            # Pretend to be a real browser
            "DEFAULT_REQUEST_HEADERS": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
                ),
            },
            # Allow 406 so your parse() or Playwright steps still see it
            "HTTPERROR_ALLOWED_CODES": [406],
            # Timeouts/retry tuning to reduce long tail hangs.
            "DOWNLOAD_TIMEOUT": int(self.config.get("crawling", {}).get("scraper_download_timeout_seconds", 35) or 35),
            "PLAYWRIGHT_TIMEOUT": int(self.config.get("crawling", {}).get("scraper_playwright_timeout_ms", 35000) or 35000),
            "RETRY_ENABLED": True,
            "RETRY_TIMES": int(self.config.get("crawling", {}).get("scraper_retry_times", 1) or 1),
            "CONCURRENT_REQUESTS": int(self.config.get("crawling", {}).get("scraper_concurrent_requests", 16) or 16),
            "CONCURRENT_REQUESTS_PER_DOMAIN": int(self.config.get("crawling", {}).get("scraper_concurrent_requests_per_domain", 8) or 8),
        })

        process.crawl(EventSpider, config=self.config)
        process.start()  # will block until crawl finishes
        logging.info("def run_crawler(): Crawler completed successfully.")


# --------------------------------------------------
# Main Block
# --------------------------------------------------
if __name__ == "__main__":
    os.environ["DS_STEP_NAME"] = "scraper"
    setup_logging('scraper')
    logging.info("\n\nscraper.py starting...")
    start_time = datetime.now()

    config = load_config()

    # Get handlers using lazy initialization to avoid blocking imports
    handlers = get_handlers()
    db_handler = handlers['db_handler']
    llm_handler = handlers['llm_handler']
    scraper = EventSpider(config)
    file_name = os.path.basename(__file__)
    start_df = db_handler.count_events_urls_start(file_name)

    # Start the crawler.
    scraper.run_crawler()
    
    logging.info("__main__: Crawler process completed.")
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
