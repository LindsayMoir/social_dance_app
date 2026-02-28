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
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
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

from credentials import get_credentials
from db import DatabaseHandler
from llm import LLMHandler
from logging_config import setup_logging

# --------------------------------------------------
# Global objects initialization.
# (These globals will be used by EventSpider.)
# --------------------------------------------------
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

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


def is_facebook_url(url: str) -> bool:
    """
    Return True when URL host belongs to Facebook.

    scraper.py must not crawl Facebook directly; fb.py owns Facebook crawling.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return host == "facebook.com" or host.endswith(".facebook.com")


def is_calendar_candidate(url: str, calendar_roots: set[str]) -> bool:
    """
    Returns True when URL is a known calendar seed/root or a Google Calendar link.
    """
    low = (url or "").lower()
    if "calendar.google.com" in low or "@group.calendar.google.com" in low or "%40group.calendar.google.com" in low:
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
    low = (text or "").lower()
    event_tokens = (
        "event", "events", "calendar", "schedule", "social", "dance",
        "workshop", "class", "lesson", "friday", "saturday", "sunday",
        "monday", "tuesday", "wednesday", "thursday",
    )
    return any(token in low for token in event_tokens)


def get_handlers():
    """Initialize and return handlers only when needed."""
    global _handlers_cache
    if _handlers_cache is None:
        db_handler = DatabaseHandler(config)
        llm_handler = LLMHandler(config_path="config/config.yaml")
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
        self.invalid_calendar_ids: set[str] = set()
        self.processed_calendar_ids: set[str] = set()

        logging.info("\n\nscraper.py starting...")

    @staticmethod
    def _domain_for_url(url: str) -> str:
        """Return lowercase netloc for a URL."""
        try:
            return urlparse(url).netloc.lower().strip()
        except Exception:
            return ""

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

    def handle_request_error(self, failure) -> None:
        """Scrapy errback to track transient domain failures and cooldown behavior."""
        request = getattr(failure, "request", None)
        req_url = request.url if request else ""
        if self._is_transient_download_failure(failure):
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
                        os.makedirs('output', exist_ok=True)
                        with open('output/skipped_whitelist.csv', 'a', newline='') as f:
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
        # Skip non-text responses (e.g., images, PDFs, etc.)
        if not isinstance(response, TextResponse):
            return
        self._record_domain_success(response.url)
        
        try:
            is_whitelisted_origin = db_handler.is_whitelisted_url(url)
        except Exception:
            is_whitelisted_origin = False
        is_whitelisted_origin = is_whitelisted_origin or is_whitelist_candidate(url, self.whitelist_roots)
        if is_whitelisted_origin:
            norm_current = normalize_url_for_compare(url)
            for root in self.whitelist_roots:
                if norm_current.startswith(root):
                    self.attempted_whitelist_roots.add(root)
                    break

        if is_facebook_url(url):
            child_row = [url, '', source, [], False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)
            logging.info("def parse(): Skipping Facebook URL (owned by fb.py): %s", url)
            return

        if 'instagram' in url.lower() and not is_whitelisted_origin:
            # record it as unwanted and stop processing immediately
            child_row = [url, '', source, [], False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)
            logging.info(f"def parse(): Skipping and recording unwanted original URL: {url}")
            return
        
        # 1) Get rendered page text
        extracted_text = response.text

        # 2) Keyword & LLM logic
        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
        relevant    = False
        parent_url  = ''
        crawl_try   = 1
        time_stamp  = datetime.now()
        # build the initial record for this URL
        url_row = [url, parent_url, source, found_keywords, relevant, crawl_try, time_stamp]

        if found_keywords:
            logging.info(f"def parse(): Found keywords for URL {url}: {found_keywords}")
            should_run_llm = (
                is_whitelisted_origin
                or is_calendar_candidate(url, self.calendar_urls_set)
                or has_event_signal(extracted_text)
            )
            if should_run_llm:
                # Use URL-specific prompt mapping when available; LLMHandler falls back to default.
                prompt_type = url
                llm_status = llm_handler.process_llm_response(url, parent_url, extracted_text, source, keywords, prompt_type)
                if llm_status:
                    # mark as relevant
                    url_row[4] = True
                    db_handler.write_url_to_db(url_row)
                    logging.info(f"def parse(): URL {url} marked as relevant (LLM positive).")
                else:
                    db_handler.write_url_to_db(url_row)
                    logging.info(f"def parse(): URL {url} marked as irrelevant (LLM negative).")
            else:
                db_handler.write_url_to_db(url_row)
                logging.info("def parse(): URL %s skipped LLM due to low event signal.", url)
        else:
            db_handler.write_url_to_db(url_row)
            logging.info(f"def parse(): URL {url} marked as irrelevant (no keywords).")

        # 3) Extract all <a href> links (limit to configured maximum)
        raw_links = response.css('a::attr(href)').getall()
        page_links_all = normalize_http_links(
            response.url,
            raw_links,
        )
        page_links = prioritize_links_for_crawl(page_links_all, self.config['crawling']['max_website_urls'])
        logging.info(f"def parse(): Found {len(page_links)} links on {response.url}")

        # 4) Process iframes & extract Google Calendar addresses
        iframe_links = normalize_http_links(response.url, response.css('iframe::attr(src)').getall())
        calendar_emails = re.findall(
            r'"gcal"\s*:\s*"([A-Za-z0-9_.+-]+@group\.calendar\.google\.com)"',
            response.text
        )
        calendar_anchor_links = [link for link in page_links if is_calendar_candidate(link, self.calendar_urls_set)]
        calendar_sources = iframe_links + calendar_anchor_links
        calendar_ids: set[str] = {
            c for c in calendar_emails if self.is_valid_calendar_id(c, allow_gmail=False)
        }
        # Only trust high-confidence group calendar IDs from raw page text.
        calendar_ids.update(self.extract_calendar_ids(extracted_text, allow_gmail=False))

        for cal_url in calendar_sources:
            # URLs discovered from iframe/embed/calendar links may legitimately reference gmail calendars.
            extracted_ids = self.extract_calendar_ids(cal_url, allow_gmail=True)
            if extracted_ids:
                calendar_ids.update(extracted_ids)
                continue
            self.fetch_google_calendar_events(cal_url, url, source, keywords)

        if calendar_ids:
            for calendar_id in sorted(calendar_ids):
                self.process_calendar_id(calendar_id, response.url, url, source, keywords)

        if calendar_sources or calendar_ids:
            # mark the page itself as relevant if calendar events fetched
            url_row = [url, "", source, found_keywords, True, crawl_try, time_stamp]
            db_handler.write_url_to_db(url_row)

        # 5) Filter unwanted links and record them
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

            if 'instagram' in low:
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
            self.visited_link.add(link)

            # record the child link before crawling
            child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)

            if len(self.visited_link) >= self.config['crawling']['urls_run_limit']:
                remaining_whitelist = len(self.whitelist_roots - self.attempted_whitelist_roots)
                if remaining_whitelist > 0 and not force_follow:
                    logging.info(
                        "parse(): URL run limit reached but %d whitelist roots are still unattempted; "
                        "skipping non-whitelist link: %s",
                        remaining_whitelist,
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
            yield scrapy.Request(
                url=link,
                callback=self.parse,
                errback=self.handle_request_error,
                cb_kwargs={'keywords': keywords, 'source': source, 'url': link},
                priority=800 if force_follow else 0,
                meta=self._build_playwright_request_meta(high_priority=force_follow),
            )


    def fetch_google_calendar_events(self, calendar_url, url, source, keywords):
        """
        Fetch and process events from a Google Calendar.
        """
        logging.info(f"def fetch_google_calendar_events(): Inputs - calendar_url: {calendar_url}, URL: {url}, source: {source}, keywords: {keywords}")
        if not self._is_google_calendar_like_url(calendar_url):
            logging.info(
                "def fetch_google_calendar_events(): Skipping non-calendar-like URL: %s",
                calendar_url,
            )
            return
        calendar_ids = self.extract_calendar_ids(calendar_url, allow_gmail=True)
        if not calendar_ids:
            if self.is_valid_calendar_id(calendar_url, allow_gmail=True):
                calendar_ids = [calendar_url]
            else:
                decoded_calendar_id = self.decode_calendar_id(calendar_url)
                if decoded_calendar_id:
                    calendar_ids = [decoded_calendar_id]
                else:
                    logging.warning(f"def fetch_google_calendar_events(): Failed to extract valid Calendar ID from {calendar_url}")
                    return
        for calendar_id in calendar_ids:
            self.process_calendar_id(calendar_id, calendar_url, url, source, keywords)


    @staticmethod
    def _is_google_calendar_like_url(candidate_url: str) -> bool:
        """Return True only for URLs that plausibly contain Google Calendar data/IDs."""
        low = (candidate_url or "").lower()
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
            )
        )


    def extract_calendar_ids(self, calendar_url, allow_gmail: bool = False):
        if not calendar_url:
            return []

        text = str(calendar_url)
        # Handle escaped JS strings like "https:\/\/calendar.google.com\/calendar\/ical\/..."
        text = text.replace("\\/", "/")

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
            return
        if calendar_id in self.invalid_calendar_ids:
            logging.info(
                "def process_calendar_id(): Skipping previously invalid calendar_id: %s",
                calendar_id,
            )
            return
        logging.info(f"def process_calendar_id(): Processing calendar_id: {calendar_id} from {calendar_url}")
        events_df = self.get_calendar_events(calendar_id)
        self.processed_calendar_ids.add(calendar_id)
        if not events_df.empty:
            logging.info(f"def process_calendar_id(): Found {len(events_df)} events for calendar_id: {calendar_id}")
            logging.info(f"def process_calendar_id(): Event columns: {list(events_df.columns)}")
            logging.info(f"def process_calendar_id(): Sample event data:\n{events_df.head(1).to_dict('records')}")
            db_handler.write_events_to_db(events_df, calendar_id, calendar_url, source, keywords)
        else:
            logging.warning(f"def process_calendar_id(): No events found for calendar_id: {calendar_id}")


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
                "output/output.json": {"format": "json"}
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

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

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
