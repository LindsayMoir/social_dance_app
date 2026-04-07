# images.py

from pathlib import Path
import asyncio
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import sys
import time
import random
import re
from datetime import datetime
from urllib.parse import quote_plus, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from io import BytesIO
import numpy as np
import pandas as pd
import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from pytesseract import Output
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from scrapy import Selector
from sqlalchemy import text
import yaml
from config_runtime import get_config_path, load_config

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - defensive import fallback
    PaddleOCR = None

from db import DatabaseHandler
from llm import LLMHandler
from page_classifier import is_instagram_post_detail_url, is_instagram_url, resolve_prompt_type
from rd_ext import ReadExtract
from secret_paths import get_auth_file

# Constants
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".tiff", ".tif", ".svg", ".webp", ".ico",
    ".avif", ".jfif"
}
# Default headers
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/100.0.0.0 Safari/537.36"
)

# 0. Configure Tesseract binary path
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 1. Load environment and configuration
load_dotenv()

# Setup centralized logging
from logging_config import setup_logging
setup_logging('images')

config_path = Path(get_config_path())
config = load_config(str(config_path))

logger = logging.getLogger(__name__)
logger.info("\n\nStarting images.py ...")

_INSTAGRAM_DEGRADED_SHELL_TOKENS = (
    "this content is no longer available",
    "the content you requested cannot be displayed right now",
    "see everyday moments from your close friends",
    "continue use another profile create new account",
    "sign up for instagram",
    "log in by continuing",
)
_VISION_MODEL = "gpt-4.1-mini"
_VISION_REQUEST_TIMEOUT_SECONDS = 12
_INSTAGRAM_MANUAL_RECOVERY_TIMEOUT_SECONDS = 180
_IMAGE_REPLAY_FRAGMENT_KEY = "image"
_INSTAGRAM_DYNAMIC_MEDIA_QUERY_PARAMS = frozenset(
    {
        "_nc_gid",
        "_nc_ohc",
        "_nc_oc",
        "oh",
        "oe",
        "_nc_zt",
        "_nc_ad",
        "_nc_cid",
        "ccb",
    }
)


@dataclass(frozen=True)
class ImagePosterDateAnalysis:
    """Structured date analysis for one poster image."""

    poster_type: str
    primary_date: str | None = None
    primary_day_of_week: str | None = None
    candidate_dates: tuple[str, ...] = ()
    reason: str = ""


def _safe_bool(value: object) -> bool:
    """Coerce config-like boolean values safely."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}



def detect_date_from_image(local_path: Path) -> tuple[str | None, str | None]:
    """
    Detect a Month + Day from a poster-style image using OCR word boxes and infer a full date.

    Returns:
        (yyyy_mm_dd, weekday) or (None, None) if no confident detection.
    """
    from PIL import Image as _PILImage
    try:
        base_img = _PILImage.open(local_path).convert('RGB')
    except Exception:
        return (None, None)

    def _find_month_day(img):
        from pytesseract import Output as _Output
        from PIL import ImageEnhance as _IE
        # Scale and increase contrast
        max_dim = max(img.size)
        if max_dim < 1400:
            scale = 1400 / max_dim
            img = img.resize((int(img.width * scale), int(img.height * scale)), _PILImage.LANCZOS)
        gray = img.convert('L')
        gray = _IE.Contrast(gray).enhance(2.2)

        words = []
        for psm in (11, 6, 4):
            try:
                data = pytesseract.image_to_data(gray, lang='eng', config=f'--oem 3 --psm {psm}', output_type=_Output.DICT)
            except Exception:
                continue
            for i in range(len(data.get('text', []))):
                t = (data['text'][i] or '').strip()
                if not t:
                    continue
                try:
                    conf = float(data.get('conf', ['0'])[i]) if isinstance(data.get('conf'), list) else 0.0
                except Exception:
                    conf = 0.0
                if conf < 0:
                    continue
                words.append({
                    'text': t,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                    'block': data.get('block_num', [0])[i],
                })
        if not words:
            return None

        months = {'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12}
        best = None
        for m in (w for w in words if w['text'].lower() in months):
            # Month center
            mxc = m['x'] + m['w']//2
            myc = m['y'] + m['h']//2
            def _nd(u):
                return u['text'].replace('O','0').replace('o','0')
            # If any two-digit near the month exists, prefer it over lone single-digit candidates
            two_digit_near = any(_nd(v).isdigit() and len(_nd(v))>=2 and abs((v['y']+v['h']//2)-myc) < 140 and abs((v['x']+v['w']//2)-mxc) < 240 for v in words)
            addr_kw = {'studio','st','st.','street','ave','avenue','road','rd','yates','blvd','boulevard','way','lane','ln'}
            best_local = None
            for w in words:
                t = _nd(w)
                if not t.isdigit():
                    continue
                n = int(t)
                if not (1 <= n <= 31):
                    continue
                dy = abs((w['y'] + w['h']//2) - myc)
                dx = abs((w['x'] + w['w']//2) - mxc)
                block_pen = 0 if w['block'] == m['block'] else 40
                # Bonuses (negative) and penalties
                size_bonus = -2 * min(w['h'], m['h'])
                xalign_bonus = -0.5 * dx
                # Context penalty using nearest token
                context_pen = 0
                wx = w['x'] + w['w']//2
                wy = w['y'] + w['h']//2
                nearest = None
                min_d = 10**9
                for u in words:
                    if u is w:
                        continue
                    ux = u['x'] + u['w']//2
                    uy = u['y'] + u['h']//2
                    d = abs(ux - wx) + abs(uy - wy)
                    if d < min_d:
                        min_d = d
                        nearest = u
                if nearest and nearest['text'].lower().strip('.,') in addr_kw and min_d < 160:
                    context_pen += 75
                single_digit_pen = 120 if (len(t)==1 and two_digit_near) else 0
                score = dy*8 + dx*0.5 + block_pen + context_pen + single_digit_pen + size_bonus + xalign_bonus
                cand = (months[m['text'].lower()], n, score)
                if (best_local is None) or (cand[2] < best_local[2]):
                    best_local = cand
            if best_local and ((best is None) or (best_local[2] < best[2])):
                best = best_local
            
            return best

    # Try base and rotated variants
    for angle in (0, 90, 270):
        img = base_img.rotate(angle, expand=True) if angle else base_img
        best = _find_month_day(img)
        if best:
            month_num, day_num, _ = best
            from datetime import date as _date, timedelta as _td, datetime as _dt
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo('America/Los_Angeles')
                today = _dt.now(tz).date()
            except Exception:
                today = _dt.now().date()
            year = today.year
            try_date = _date(year, month_num, day_num)
            if try_date < today and (today - try_date) > _td(days=90):
                year += 1
            final_date = _date(year, month_num, day_num)
            return (final_date.isoformat(), final_date.strftime('%A'))
    return (None, None)


def _current_pacific_date() -> datetime.date:
    """Return current Pacific-local date for poster date inference."""
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:
        return datetime.now().date()


def _extract_textual_candidate_dates(text: str) -> list[str]:
    """Extract explicit month/day date candidates from OCR text."""
    raw_text = str(text or "")
    if not raw_text.strip():
        return []

    month_lookup = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    today = _current_pacific_date()
    candidates: list[str] = []
    pattern = re.compile(
        r"\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)?"
        r"[\s,/-]*"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[\s,/-]+(\d{1,2})(?:st|nd|rd|th)?(?:[\s,/-]+(20\d{2}))?\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(raw_text):
        month_token, day_token, year_token = match.groups()
        month_num = month_lookup.get(str(month_token).lower())
        if not month_num:
            continue
        day_num = int(day_token)
        year_num = int(year_token) if year_token else today.year
        try:
            candidate_date = datetime(year_num, month_num, day_num).date()
        except ValueError:
            continue
        if not year_token and candidate_date < today and (today - candidate_date).days > 90:
            try:
                candidate_date = datetime(year_num + 1, month_num, day_num).date()
            except ValueError:
                continue
        candidates.append(candidate_date.isoformat())
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _analyze_image_poster_dates(local_path: Path, ocr_text: str) -> ImagePosterDateAnalysis:
    """Classify image posters as single-event or schedule-style and emit date hints."""
    primary_date, primary_dow = detect_date_from_image(local_path)
    candidate_dates = _extract_textual_candidate_dates(ocr_text)
    normalized_primary = str(primary_date or "").strip()
    if normalized_primary and normalized_primary not in candidate_dates:
        candidate_dates = [normalized_primary, *candidate_dates]

    weekday_hits = re.findall(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b",
        str(ocr_text or ""),
        flags=re.IGNORECASE,
    )
    month_hits = re.findall(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        str(ocr_text or ""),
        flags=re.IGNORECASE,
    )

    if len(candidate_dates) >= 3 or (len(candidate_dates) >= 2 and len(weekday_hits) >= 3 and len(month_hits) >= 1):
        return ImagePosterDateAnalysis(
            poster_type="schedule_multi_event",
            primary_date=None,
            primary_day_of_week=None,
            candidate_dates=tuple(candidate_dates[:12]),
            reason="multiple_textual_date_candidates",
        )

    if normalized_primary:
        return ImagePosterDateAnalysis(
            poster_type="single_event",
            primary_date=normalized_primary,
            primary_day_of_week=str(primary_dow or "").strip() or None,
            candidate_dates=tuple(candidate_dates[:6]),
            reason="single_detected_primary_date",
        )

    if len(candidate_dates) == 1:
        inferred_day = ""
        try:
            inferred_day = datetime.strptime(candidate_dates[0], "%Y-%m-%d").strftime("%A")
        except Exception:
            inferred_day = ""
        return ImagePosterDateAnalysis(
            poster_type="single_event",
            primary_date=candidate_dates[0],
            primary_day_of_week=inferred_day or None,
            candidate_dates=tuple(candidate_dates),
            reason="single_textual_date_candidate",
        )

    return ImagePosterDateAnalysis(
        poster_type="unknown",
        primary_date=None,
        primary_day_of_week=None,
        candidate_dates=tuple(candidate_dates[:12]),
        reason="no_confident_date_resolution",
    )


def _prepend_image_date_hints(local_path: Path, ocr_text: str) -> tuple[str, ImagePosterDateAnalysis]:
    """Prepend structured poster/date hints to OCR text for downstream extraction."""
    analysis = _analyze_image_poster_dates(local_path, ocr_text)
    hint_lines: list[str] = [f"Detected_Poster_Type: {analysis.poster_type}"]
    if analysis.primary_date and analysis.primary_day_of_week:
        hint_lines.append(f"Detected_Date: {analysis.primary_date}")
        hint_lines.append(f"Detected_Day: {analysis.primary_day_of_week}")
    elif analysis.primary_date:
        hint_lines.append(f"Detected_Date: {analysis.primary_date}")
    if analysis.candidate_dates:
        hint_lines.append(f"Detected_Schedule_Dates: {', '.join(analysis.candidate_dates)}")
    if analysis.reason:
        hint_lines.append(f"Detected_Date_Analysis: {analysis.reason}")
    return "\n".join(hint_lines + [str(ocr_text or "")]), analysis


def _extract_instagram_post_links(rendered_html: str, page_url: str) -> list[str]:
    """Extract unique Instagram post/reel/tv links from rendered profile HTML."""
    if not rendered_html or not is_instagram_url(page_url) or is_instagram_post_detail_url(page_url):
        return []
    selector = Selector(text=rendered_html)
    seen: set[str] = set()
    links: list[str] = []
    for href in selector.xpath('//a/@href').getall():
        candidate = str(href or "").strip()
        if not candidate:
            continue
        absolute = urljoin(page_url, candidate)
        if not is_instagram_post_detail_url(absolute):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def _dedupe_preserve_order(urls: list[str]) -> list[str]:
    """Return unique URLs while preserving their first-seen order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        candidate = str(url or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _normalize_instagram_media_identity_url(image_url: str) -> str:
    """Strip CDN cache-buster params so one Instagram poster image has one stable identity."""
    raw = str(image_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    query_pairs: list[str] = []
    for chunk in (parsed.query or "").split("&"):
        if not chunk:
            continue
        key = chunk.split("=", 1)[0]
        if key in _INSTAGRAM_DYNAMIC_MEDIA_QUERY_PARAMS:
            continue
        query_pairs.append(chunk)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, "&".join(query_pairs), ""))


def _normalize_image_identity_url(image_url: str) -> str:
    """Return a stable in-run identity for one underlying image asset."""
    raw = str(image_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if "instagram" in (parsed.netloc or "").lower() or "fbcdn.net" in (parsed.netloc or "").lower():
        return _normalize_instagram_media_identity_url(raw)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _build_repeated_poster_signature(
    *,
    source: str,
    analysis: ImagePosterDateAnalysis,
    ocr_text: str,
) -> str:
    """Build a stable signature for poster-like OCR content to avoid repeated low-yield work."""
    normalized_text = re.sub(r"\s+", " ", str(ocr_text or "").lower()).strip()
    payload = {
        "source": str(source or "").strip().lower(),
        "poster_type": analysis.poster_type,
        "primary_date": analysis.primary_date or "",
        "candidate_dates": list(analysis.candidate_dates),
        "ocr_text_hash": hashlib.sha1(normalized_text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_image_replay_url(parent_url: str, image_url: str) -> str:
    """Build a stable per-image replay URL anchored to the parent post/page URL."""
    base_parent = str(parent_url or "").strip() or str(image_url or "").strip()
    parsed_parent = urlparse(base_parent)
    parent_without_fragment = urlunparse(
        (parsed_parent.scheme, parsed_parent.netloc, parsed_parent.path, parsed_parent.params, parsed_parent.query, "")
    )
    normalized_image = _normalize_instagram_media_identity_url(image_url)
    digest = hashlib.sha1(normalized_image.encode("utf-8")).hexdigest()[:12]
    return f"{parent_without_fragment}#{_IMAGE_REPLAY_FRAGMENT_KEY}={digest}"


def is_image_replay_url(url: str) -> bool:
    """Return True when URL points to a synthetic per-image replay child URL."""
    parsed = urlparse(str(url or "").strip())
    return bool(parsed.fragment.startswith(f"{_IMAGE_REPLAY_FRAGMENT_KEY}="))


def strip_image_replay_fragment(url: str) -> str:
    """Remove synthetic image replay fragment and return the parent page URL."""
    parsed = urlparse(str(url or "").strip())
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _is_instagram_profile_candidate(url: str) -> bool:
    """Return True when the URL looks like an Instagram profile page."""
    if not is_instagram_url(url) or is_instagram_post_detail_url(url):
        return False
    parsed = urlparse(url)
    path = str(parsed.path or "").strip("/")
    if not path:
        return False
    first_segment = path.split("/", 1)[0].lower()
    return first_segment not in {"accounts", "explore", "direct", "about", "legal", "developer"}


def _extract_instagram_search_links(rendered_html: str, page_url: str) -> list[str]:
    """Extract profile and post links from an Instagram search results page."""
    if not rendered_html or not is_instagram_url(page_url):
        return []

    selector = Selector(text=rendered_html)
    discovered_links: list[str] = []
    for href in selector.xpath("//a/@href").getall():
        candidate = str(href or "").strip()
        if not candidate:
            continue
        absolute = urljoin(page_url, candidate)
        if is_instagram_post_detail_url(absolute) or _is_instagram_profile_candidate(absolute):
            discovered_links.append(absolute)
    return _dedupe_preserve_order(discovered_links)


def _is_keyword_discovered_instagram_target(url: str) -> bool:
    """Return True only for high-value Instagram targets discovered from keyword search."""
    candidate = str(url or "").strip()
    if not candidate or not is_instagram_url(candidate):
        return False
    if is_instagram_post_detail_url(candidate):
        return True
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/").lower()
    if not path:
        return False
    if host.startswith(("about.instagram.com", "help.instagram.com", "business.instagram.com")):
        return False
    disallowed_prefixes = {
        "about",
        "accounts",
        "blog",
        "careers",
        "developer",
        "direct",
        "explore",
        "legal",
        "privacy",
        "reels",
        "web",
    }
    first_segment = path.split("/", 1)[0]
    return first_segment not in disallowed_prefixes


def _build_image_context_text(
    ocr_text: str,
    parent_url: str,
    source: str,
    page_context_text: str | None = None,
) -> str:
    """Combine OCR text with parent-page context for image extraction."""
    parts: list[str] = []
    if source:
        parts.append(f"Source: {source}")
    if parent_url:
        parts.append(f"Parent_URL: {parent_url}")
    if page_context_text:
        cleaned_context = " ".join(str(page_context_text).split())
        if cleaned_context:
            parts.append(f"Parent_Page_Text: {cleaned_context}")
    parts.append(str(ocr_text or ""))
    return "\n".join(part for part in parts if part)


def _is_degraded_instagram_profile_text(text: str) -> bool:
    """Return True when Instagram profile text looks like a shell/login/degraded page."""
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return True
    return any(token in normalized for token in _INSTAGRAM_DEGRADED_SHELL_TOKENS)


def _is_instagram_login_redirect_url(page_url: str) -> bool:
    """Return True when the current Instagram URL is a login or account-selection redirect."""
    normalized_url = str(page_url or "").lower()
    return any(
        token in normalized_url
        for token in (
            "/accounts/login",
            "next=%2f",
            "source=omni_redirect",
        )
    )


def _looks_like_authenticated_instagram_profile(page_url: str, text: str) -> bool:
    """Return True only when the rendered Instagram page looks usable for profile scraping."""
    if _is_instagram_login_redirect_url(page_url):
        return False
    return not _is_degraded_instagram_profile_text(text)


def _extract_rankable_image_urls(rendered_html: str, page_url: str) -> list[str]:
    """Extract image-like URLs from rendered HTML, excluding obvious Instagram UI assets."""
    if not rendered_html:
        return []
    imgs = Selector(text=rendered_html).xpath('//img/@src').getall()
    image_urls = [urljoin(page_url, src) for src in imgs if src and src.strip()]
    rankable_urls: list[str] = []
    for image_url in image_urls:
        if not image_url or not image_url.strip():
            continue
        if "instagram.com" in image_url or "fbcdn.net" in image_url or Path(urlparse(image_url).path).suffix.lower() in IMAGE_EXTENSIONS:
            if _is_ignored_instagram_ui_asset(image_url):
                continue
            rankable_urls.append(image_url)
    return _dedupe_preserve_order(rankable_urls)


def _safe_screenshot_stem(page_url: str) -> str:
    """Create a filesystem-safe stem for screenshot artifacts."""
    parsed = urlparse(str(page_url or ""))
    raw = f"{parsed.netloc or 'page'}_{parsed.path or 'root'}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return cleaned or "page"


def _is_ignored_instagram_ui_asset(image_url: str) -> bool:
    """Return True for obvious Instagram UI/static assets that should not be OCR'd."""
    try:
        parsed = urlparse(str(image_url or ""))
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return "static.cdninstagram.com" in host or "/rsrc.php/" in path or path.endswith("/rsrc.php")


def _score_image_candidate(image_url: str, width: int, height: int) -> int:
    """Prefer likely poster/media images over UI assets and awkward aspect ratios."""
    try:
        parsed = urlparse(str(image_url or ""))
    except Exception:
        parsed = urlparse("")
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if _is_ignored_instagram_ui_asset(image_url):
        return -1000

    score = 0
    if "fbcdn.net" in host:
        score += 120
    elif "cdninstagram.com" in host:
        score += 90
    if "/v/t51" in path:
        score += 35

    area = max(0, int(width) * int(height))
    score += min(40, area // 50000)

    if height >= width * 1.1 and height <= width * 2.3:
        score += 25
    elif width >= height * 0.85 and width <= height * 1.2:
        score += 12
    elif width >= height * 2.0 or height >= width * 3.0:
        score -= 30

    return score

class ImageScraper:
    _paddle_ocr_engine = None
    _paddle_ocr_init_attempted = False

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ImageScraper")
        self.urls_visited = set()

        # Handlers
        self.llm_handler = LLMHandler(config_path=str(config_path))
        self.db_handler = self.llm_handler.db_handler  # Use the DatabaseHandler from LLMHandler
        self.keywords_list = self.llm_handler.get_keywords()
        self.images_per_page_limit = int(
            self.config.get("crawling", {}).get("images_per_page_limit", 10) or 10
        )
        self.max_images_per_page = int(
            self.config.get("crawling", {}).get("image_max_processed_per_page", 5) or 5
        )
        self.instagram_vision_image_limit = int(
            self.config.get("crawling", {}).get("instagram_vision_image_limit", 3) or 3
        )
        self.run_id = os.getenv("DS_RUN_ID", "na")
        self.step_name = os.getenv("DS_STEP_NAME", "images")
        self.telemetry_counts: Counter[str] = Counter()
        self._processed_image_identities: set[str] = set()
        self._repeated_poster_outcomes: dict[str, int] = {}

        # Directories
        self.download_dir = Path(config.get("image_download_dir", "images/"))
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Async event loop for Playwright
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize ReadExtract and login
        self.read_extract = ReadExtract(config_path=str(config_path))
        self.read_extract.config = self.config
        self.loop.run_until_complete(self.read_extract.init_browser())
        if not self.loop.run_until_complete(self._login_to_instagram()):
            self.logger.error("Instagram login failed. Exiting.")
            sys.exit(1)

    async def _extract_page_text_playwright(self, page_url: str) -> str | None:
        """
        Extract page text using the existing authenticated Playwright page.

        This path is used for Instagram pages because rd_ext intentionally skips
        social URLs in its generic extractor.
        """
        try:
            await self.read_extract.page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
            await self.read_extract.page.wait_for_timeout(2500)
            content = await self.read_extract.page.content()
        except Exception as e:
            self.logger.warning("_extract_page_text_playwright(): failed for %s: %s", page_url, e)
            return None

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text or None

    def _get_processed_image_identities(self) -> set[str]:
        """Return the in-run set of already-processed image identities."""
        identities = getattr(self, "_processed_image_identities", None)
        if identities is None:
            identities = set()
            self._processed_image_identities = identities
        return identities

    def _get_repeated_poster_outcomes(self) -> dict[str, int]:
        """Return the in-run cache of poster OCR signatures to prior outcomes."""
        outcomes = getattr(self, "_repeated_poster_outcomes", None)
        if outcomes is None:
            outcomes = {}
            self._repeated_poster_outcomes = outcomes
        return outcomes

    def _mark_image_identity_processed(self, image_url: str) -> str:
        """Record one normalized image identity as processed for this run."""
        identity = _normalize_image_identity_url(image_url)
        if identity:
            self._get_processed_image_identities().add(identity)
        return identity

    def _has_processed_image_identity(self, image_url: str) -> bool:
        """Return True when this underlying image asset was already processed in-run."""
        identity = _normalize_image_identity_url(image_url)
        return bool(identity) and identity in self._get_processed_image_identities()

    def _lookup_repeated_poster_outcome(
        self,
        *,
        source: str,
        analysis: ImagePosterDateAnalysis,
        ocr_text: str,
    ) -> tuple[str, int] | None:
        """Return prior repeated-poster outcome when the OCR signature was already seen."""
        signature = _build_repeated_poster_signature(
            source=source,
            analysis=analysis,
            ocr_text=ocr_text,
        )
        outcomes = self._get_repeated_poster_outcomes()
        if signature not in outcomes:
            return None
        return signature, int(outcomes.get(signature, 0) or 0)

    def _remember_repeated_poster_outcome(
        self,
        *,
        source: str,
        analysis: ImagePosterDateAnalysis,
        ocr_text: str,
        events_written: int,
    ) -> None:
        """Cache one poster OCR signature outcome so repeated posters can short-circuit."""
        signature = _build_repeated_poster_signature(
            source=source,
            analysis=analysis,
            ocr_text=ocr_text,
        )
        self._get_repeated_poster_outcomes()[signature] = int(events_written or 0)

    def _fetch_page_response(self, page_url: str, timeout_seconds: int = 10) -> requests.Response | None:
        """Fetch one page with a narrow retry for transient DNS/connection failures."""
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(page_url, timeout=timeout_seconds)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                message = str(exc)
                is_transient_dns = "Temporary failure in name resolution" in message or "NameResolutionError" in message
                if attempt < attempts and is_transient_dns:
                    self.logger.warning(
                        "_fetch_page_response(): transient fetch failure for %s on attempt %d/%d: %s",
                        page_url,
                        attempt,
                        attempts,
                        exc,
                    )
                    time.sleep(1.0)
                    continue
                self.logger.error("process_webpage_url(): Failed to fetch %s: %s", page_url, exc)
                return None

    async def _extract_instagram_post_links_playwright(
        self,
        page_url: str,
        max_links: int,
    ) -> list[str]:
        """Use the authenticated Playwright page to resolve visible Instagram post links."""
        if not is_instagram_url(page_url) or is_instagram_post_detail_url(page_url):
            return []

        page = self.read_extract.page
        discovered_links: list[str] = []
        selector = 'a[href*="/p/"], a[href*="/reel/"], a[href*="/tv/"]'
        max_candidates = max(max_links * 3, max_links)

        try:
            await page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            self.logger.warning(
                "_extract_instagram_post_links_playwright(): failed initial goto for %s: %s",
                page_url,
                exc,
            )
            return []

        for attempt in range(3):
            try:
                await page.wait_for_timeout(2500 if attempt == 0 else 1500)
                rendered_html = await page.content()
                discovered_links.extend(_extract_instagram_post_links(rendered_html, page_url))
                if discovered_links:
                    break

                locator = page.locator(selector)
                try:
                    await locator.first.wait_for(timeout=4000)
                except Exception:
                    pass

                count = await locator.count()
                for index in range(min(count, max_candidates)):
                    href = await locator.nth(index).get_attribute("href")
                    absolute = urljoin(page_url, str(href or "").strip())
                    if is_instagram_post_detail_url(absolute):
                        discovered_links.append(absolute)
                if discovered_links:
                    break

                article_links = page.locator("article a")
                article_count = await article_links.count()
                for index in range(min(article_count, max_candidates)):
                    anchor = article_links.nth(index)
                    href = await anchor.get_attribute("href")
                    absolute = urljoin(page_url, str(href or "").strip())
                    if is_instagram_post_detail_url(absolute):
                        discovered_links.append(absolute)
                        continue
                    try:
                        await anchor.click(timeout=2500)
                        await page.wait_for_timeout(1000)
                    except Exception:
                        continue
                    current_url = str(getattr(page, "url", "") or "")
                    if is_instagram_post_detail_url(current_url):
                        discovered_links.append(current_url)
                    try:
                        if current_url and current_url != page_url:
                            await page.go_back(timeout=3000, wait_until="domcontentloaded")
                        else:
                            await page.keyboard.press("Escape")
                    except Exception:
                        pass
                if discovered_links:
                    break

                await page.mouse.wheel(0, 1400)
            except Exception as exc:
                self.logger.info(
                    "_extract_instagram_post_links_playwright(): attempt %d failed for %s: %s",
                    attempt + 1,
                    page_url,
                    exc,
                )

        return _dedupe_preserve_order(discovered_links)[:max_links]

    async def _capture_page_screenshot(self, page_url: str) -> Path | None:
        """Capture a screenshot of the currently rendered page for OCR fallback."""
        try:
            if getattr(self.read_extract.page, "url", "") != page_url:
                await self.read_extract.page.goto(page_url, wait_until="domcontentloaded", timeout=30000)
                await self.read_extract.page.wait_for_timeout(2500)
            screenshot_path = self.download_dir / f"{_safe_screenshot_stem(page_url)}_rendered.png"
            await self.read_extract.page.screenshot(path=str(screenshot_path), full_page=True)
            self.logger.info("_capture_page_screenshot(): Saved screenshot for %s to %s", page_url, screenshot_path)
            return screenshot_path
        except Exception as exc:
            self.logger.warning("_capture_page_screenshot(): Failed for %s: %s", page_url, exc)
            return None

    async def _search_instagram_keyword_links_playwright(
        self,
        keyword: str,
        max_links: int,
    ) -> list[str]:
        """Search Instagram for a keyword and return profile/post links."""
        cleaned_keyword = str(keyword or "").strip()
        if not cleaned_keyword or max_links <= 0:
            return []

        search_url = f"https://www.instagram.com/explore/search/keyword/?q={quote_plus(cleaned_keyword)}"
        page = self.read_extract.page
        discovered_links: list[str] = []
        max_candidates = max(max_links * 3, max_links)

        try:
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as exc:
            self.logger.warning(
                "_search_instagram_keyword_links_playwright(): failed goto for keyword=%s: %s",
                cleaned_keyword,
                exc,
            )
            return []

        for attempt in range(3):
            try:
                await page.wait_for_timeout(2500 if attempt == 0 else 1500)
                rendered_html = await page.content()
                discovered_links.extend(_extract_instagram_search_links(rendered_html, search_url))
                if len(_dedupe_preserve_order(discovered_links)) >= max_links:
                    break

                locator = page.locator('a[href*="/p/"], a[href*="/reel/"], a[href^="/"], a[href*="instagram.com/"]')
                count = await locator.count()
                for index in range(min(count, max_candidates)):
                    href = await locator.nth(index).get_attribute("href")
                    absolute = urljoin(search_url, str(href or "").strip())
                    if is_instagram_post_detail_url(absolute) or _is_instagram_profile_candidate(absolute):
                        discovered_links.append(absolute)
                if len(_dedupe_preserve_order(discovered_links)) >= max_links:
                    break

                await page.mouse.wheel(0, 1600)
            except Exception as exc:
                self.logger.info(
                    "_search_instagram_keyword_links_playwright(): attempt %d failed for keyword=%s: %s",
                    attempt + 1,
                    cleaned_keyword,
                    exc,
                )

        return _dedupe_preserve_order(discovered_links)[:max_links]

    def _discover_instagram_keyword_links(
        self,
        existing_links: set[str],
        remaining_capacity: int,
    ) -> pd.DataFrame:
        """Discover Instagram profile/post URLs from configured keywords within crawl limits."""
        if remaining_capacity <= 0:
            return pd.DataFrame(columns=["link", "parent_url", "source", "keywords", "relevant", "crawl_try", "time_stamp"])

        crawl_cfg = self.config.get("crawling", {})
        page_link_limit = int(crawl_cfg.get("max_website_urls", 10) or 10)
        image_page_limit = int(crawl_cfg.get("images_per_page_limit", 10) or 10)
        positive_limits = [value for value in (page_link_limit, image_page_limit) if value > 0]
        per_keyword_limit = min(positive_limits) if positive_limits else 0
        if per_keyword_limit <= 0:
            return pd.DataFrame(columns=["link", "parent_url", "source", "keywords", "relevant", "crawl_try", "time_stamp"])

        discovered_rows: list[dict[str, object]] = []
        seen_links = set(existing_links)
        timestamp = datetime.now()

        for keyword in self.keywords_list:
            cleaned_keyword = str(keyword or "").strip()
            if not cleaned_keyword:
                continue

            capacity_left = remaining_capacity - len(discovered_rows)
            if capacity_left <= 0:
                break

            keyword_limit = min(per_keyword_limit, capacity_left)
            discovered_links = self.loop.run_until_complete(
                self._search_instagram_keyword_links_playwright(cleaned_keyword, keyword_limit)
            )
            for link in discovered_links:
                if not _is_keyword_discovered_instagram_target(link):
                    continue
                if link in seen_links:
                    continue
                seen_links.add(link)
                discovered_rows.append(
                    {
                        "link": link,
                        "parent_url": "",
                        "source": "instagram_keyword_search",
                        "keywords": cleaned_keyword,
                        "relevant": False,
                        "crawl_try": 1,
                        "time_stamp": timestamp,
                    }
                )
                if len(discovered_rows) >= remaining_capacity:
                    break

        if not discovered_rows:
            return pd.DataFrame(columns=["link", "parent_url", "source", "keywords", "relevant", "crawl_try", "time_stamp"])
        return pd.DataFrame(discovered_rows)

    def _extract_dynamic_page_text(self, page_url: str) -> str | None:
        """
        Extract dynamic page text with service ownership-aware routing.

        - Instagram pages: use local Playwright extractor in images.py.
        - All other pages: use rd_ext extractor service.
        """
        if "instagram.com" in (page_url or "").lower():
            return self.loop.run_until_complete(self._extract_page_text_playwright(page_url))
        return self.loop.run_until_complete(self.read_extract.extract_event_text(page_url))


    def _load_instagram_cookies(self) -> list | None:
        """
        Loads pre-saved Instagram authentication cookies from database or filesystem.

        Priority order:
        1. Database (auth_storage table) - synced across all environments
        2. Render Secret Files (/etc/secrets/) - for Render environment
        3. Local filesystem - for development

        Validates that critical cookies (sessionid, csrftoken) are still valid and not expired.

        Returns:
            list | None: List of cookie dictionaries if valid, None if invalid, missing, or expired.
        """
        try:
            # Use secret_paths to get auth file from database, Render secrets, or filesystem
            auth_file_path = get_auth_file('instagram')
            self.logger.info(f"_load_instagram_cookies(): Reading from {auth_file_path}")

            if not Path(auth_file_path).exists():
                self.logger.warning(f"_load_instagram_cookies(): Auth file not found at {auth_file_path}")
                return None

            with open(auth_file_path, 'r') as f:
                auth_data = json.load(f)

            cookies = auth_data.get('cookies', [])
            if not cookies:
                self.logger.warning("_load_instagram_cookies(): No cookies found in auth data")
                return None

            # Check if critical cookies are present and valid
            cookie_dict = {c['name']: c for c in cookies}
            critical_cookies = ['sessionid', 'csrftoken']

            for cookie_name in critical_cookies:
                if cookie_name not in cookie_dict:
                    self.logger.warning(f"_load_instagram_cookies(): Missing critical cookie '{cookie_name}'")
                    return None

            # Verify expiration times
            current_time = time.time()
            for cookie in cookies:
                expires = cookie.get('expires', -1)
                if expires > 0 and expires < current_time:
                    self.logger.warning(f"_load_instagram_cookies(): Cookie '{cookie['name']}' has expired (expires: {expires}, current: {current_time})")
                    return None

            self.logger.info("_load_instagram_cookies(): Successfully loaded and validated Instagram cookies from database/filesystem")
            return cookies

        except Exception as e:
            self.logger.warning(f"_load_instagram_cookies(): Failed to load cookies: {e}")
            return None

    async def _verify_instagram_session(self, probe_url: str = "https://www.instagram.com/") -> bool:
        """
        Verify that the current Instagram session can render a usable authenticated page.

        Returns:
            bool: True if session is valid, False otherwise.
        """
        try:
            await self.read_extract.page.goto(probe_url, wait_until="domcontentloaded", timeout=30000)
            await self.read_extract.page.wait_for_timeout(2500)
            current_url = str(getattr(self.read_extract.page, "url", "") or "")
            content = await self.read_extract.page.content()
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            rendered_text = " ".join(soup.get_text(separator=" ").split())

            if _looks_like_authenticated_instagram_profile(current_url, rendered_text):
                self.logger.info(
                    "_verify_instagram_session(): Instagram session verified successfully via %s",
                    current_url,
                )
                return True

            self.logger.warning(
                "_verify_instagram_session(): Instagram session rendered an unusable page. url=%s",
                current_url,
            )
            if _is_instagram_login_redirect_url(current_url):
                self.logger.warning("_verify_instagram_session(): Landed on Instagram login redirect")
            if _is_degraded_instagram_profile_text(rendered_text):
                self.logger.warning("_verify_instagram_session(): Rendered Instagram page looks degraded/shell-like")
                return False
            return False

        except Exception as e:
            self.logger.warning(f"_verify_instagram_session(): Failed to verify session: {e}")
            return False

    async def _attempt_manual_instagram_recovery(self, probe_url: str) -> bool:
        """Pause for manual Instagram login/challenge completion in headful mode and re-verify."""
        if _safe_bool(self.config.get("crawling", {}).get("headless", True)):
            return False

        self.logger.warning(
            "_attempt_manual_instagram_recovery(): Waiting for manual Instagram login/challenge completion"
        )
        recovered = False
        try:
            input(
                "Instagram challenge/login requires manual completion. "
                "Use the visible Chromium window to finish it, then press Enter here to continue..."
            )
            await self.read_extract.page.wait_for_timeout(1500)
            recovered = await self._verify_instagram_session(probe_url)
        except EOFError:
            self.logger.warning(
                "_attempt_manual_instagram_recovery(): stdin is unavailable; keeping browser open for up to %ds "
                "while polling for successful manual login",
                _INSTAGRAM_MANUAL_RECOVERY_TIMEOUT_SECONDS,
            )
            deadline = time.monotonic() + _INSTAGRAM_MANUAL_RECOVERY_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                await self.read_extract.page.wait_for_timeout(2000)
                recovered = await self._verify_instagram_session(probe_url)
                if recovered:
                    break
        if not recovered:
            return False

        try:
            storage_path = get_auth_file("instagram")
            await self.read_extract.context.storage_state(path=storage_path)
            from secret_paths import sync_auth_to_db
            sync_auth_to_db(storage_path, "instagram")
            self.logger.info(
                "_attempt_manual_instagram_recovery(): Saved refreshed Instagram session state to %s",
                storage_path,
            )
        except Exception as exc:
            self.logger.warning(
                "_attempt_manual_instagram_recovery(): Failed to persist recovered Instagram session: %s",
                exc,
            )
        return True

    async def _login_to_instagram(self) -> bool:
        if hasattr(self, 'ig_session'):
            self.logger.info("_login_to_instagram(): Reusing existing Instagram session")
            return True

        # First attempt: Try to load and use pre-saved cookies
        self.logger.info("_login_to_instagram(): Attempting to load pre-saved Instagram cookies")
        saved_cookies = self._load_instagram_cookies()

        if saved_cookies:
            try:
                # Add pre-saved cookies to Playwright context
                self.logger.info("_login_to_instagram(): Adding pre-saved cookies to Playwright context")
                await self.read_extract.context.add_cookies(saved_cookies)

                # Verify the session is valid
                if await self._verify_instagram_session("https://www.instagram.com/bachatavictoria/"):
                    self.logger.info("_login_to_instagram(): Pre-saved cookies verified successfully, using cached session")
                    # Extract cookies for requests.Session
                    cookies = await self.read_extract.context.cookies()
                    session = requests.Session()
                    for ck in cookies:
                        session.cookies.set(ck['name'], ck['value'], domain=ck['domain'])

                    ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
                    ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
                    session.headers.update({
                        "User-Agent": ua,
                        "Referer": "https://www.instagram.com/",
                        "x-csrftoken": session.cookies.get('csrftoken', ''),
                        "x-ig-app-id": ig_app_id,
                        "x-ig-www-claim": "0",
                        "x-requested-with": "XMLHttpRequest",
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "same-origin",
                    })
                    self.ig_session = session
                    return True
                else:
                    self.logger.warning("_login_to_instagram(): Pre-saved cookies failed verification, falling back to fresh login")

            except Exception as e:
                self.logger.warning(f"_login_to_instagram(): Failed to use pre-saved cookies: {e}. Attempting fresh login.")

        # Fallback: Perform fresh login via credentials
        self.logger.info("_login_to_instagram(): Performing fresh Instagram login")
        success = await self.read_extract.login_to_website(
            organization="instagram",
            login_url="https://www.instagram.com/accounts/login/",
            email_selector="input[name='username']",
            pass_selector="input[name='password']",
            submit_selector="button[type='submit']"
        )
        if not success:
            self.logger.error("_login_to_instagram(): Fresh login attempt failed")
            return False

        probe_url = "https://www.instagram.com/bachatavictoria/"
        if not await self._verify_instagram_session(probe_url):
            if await self._attempt_manual_instagram_recovery(probe_url):
                self.logger.info("_login_to_instagram(): Manual Instagram recovery succeeded")
            else:
                self.logger.error(
                    "_login_to_instagram(): Fresh login completed but Instagram still redirected to a login/degraded page"
                )
                return False

        self.logger.info("_login_to_instagram(): Fresh login successful, setting up session")
        cookies = await self.read_extract.context.cookies()
        session = requests.Session()
        for ck in cookies:
            session.cookies.set(ck['name'], ck['value'], domain=ck['domain'])
        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT
        # Use Instagram App ID from environment or fallback to a common one
        ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
        session.headers.update({
            "User-Agent": ua,
            "Referer": "https://www.instagram.com/",
            "x-csrftoken": session.cookies.get('csrftoken', ''),
            "x-ig-app-id": ig_app_id,
            "x-ig-www-claim": "0",
            "x-requested-with": "XMLHttpRequest",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
        })
        self.ig_session = session
        return True
    

    def is_image_url(self, url: str) -> bool:
        """
        Determines whether the given URL points to an image file based on its extension.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL has an image file extension, False otherwise.
        """
        path = urlparse(url).path
        ext = Path(path).suffix.lower()
        return ext in IMAGE_EXTENSIONS

    async def _download_image_playwright(self, image_url: str, max_retries: int = 1) -> Path | None:
        """
        Downloads an image using Playwright to maintain Instagram authentication context.
        This method preserves the logged-in session state that requests.Session loses.

        Args:
            image_url (str): The URL of the image to download.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            Path | None: The local file path to the downloaded image if successful, or None if the download failed.
        """
        filename = Path(urlparse(image_url).path).name
        local_path = self.download_dir / filename
        if local_path.exists():
            return local_path

        # Retry mechanism with exponential backoff
        for attempt in range(max_retries + 1):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying Playwright download after {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    await asyncio.sleep(delay)

                # Use Playwright's authenticated context to fetch the image
                self.logger.debug(f"Using Playwright to download Instagram image: {image_url}")
                response = await self.read_extract.page.request.get(image_url)

                if response.status == 200:
                    image_data = await response.body()
                    with local_path.open('wb') as f:
                        f.write(image_data)
                    self.logger.info(f"Downloaded Instagram image via Playwright to {local_path}")
                    return local_path
                elif response.status == 403:
                    self.logger.warning(f"403 Forbidden via Playwright for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All Playwright retry attempts failed for {image_url} - 403 Forbidden")
                        return None
                elif response.status == 429:
                    self.logger.warning(f"Rate limited via Playwright for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All Playwright retry attempts failed for {image_url} - Rate limited")
                        return None
                else:
                    self.logger.error(f"Playwright HTTP error {response.status} for {image_url}")
                    return None

            except Exception as e:
                self.logger.warning(f"Playwright download attempt {attempt + 1} failed for {image_url}: {e}")
                if attempt == max_retries:
                    self.logger.exception(f"All Playwright retry attempts failed for {image_url}")
                    return None

        return None

    def download_image(self, image_url: str, max_retries: int = 1) -> Path | None:
        """
        Downloads an image from the specified URL and saves it to the local download directory.
        Uses Playwright for Instagram images to maintain authentication context.

        Args:
            image_url (str): The URL of the image to download.
            max_retries (int): Maximum number of retry attempts.

        Returns:
            Path | None: The local file path to the downloaded image if successful, or None if the download failed.

        Notes:
            - If the image already exists in the download directory, the existing file path is returned.
            - Uses Playwright for Instagram/Facebook CDN images to maintain authentication state.
            - Falls back to requests session for non-Instagram images.
            - Implements exponential backoff retry mechanism.
        """
        filename = Path(urlparse(image_url).path).name
        local_path = self.download_dir / filename
        if local_path.exists():
            return local_path

        ua = self.config.get('crawling', {}).get('user_agent') or DEFAULT_USER_AGENT

        # Enhanced headers for Instagram images
        headers = {
            "User-Agent": ua,
            "Accept": "image/webp,image/apng,image/*;q=0.8,*/*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.instagram.com/",
            "Origin": "https://www.instagram.com",
            "sec-ch-ua": '"Google Chrome";v="117", "Not;A=Brand";v="8", "Chromium";v="117"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "image",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-site": "cross-site",
        }

        # Add Instagram-specific headers if we have session cookies
        if hasattr(self, 'ig_session') and self.ig_session.cookies:
            csrf_token = self.ig_session.cookies.get('csrftoken', '')
            if csrf_token:
                headers["x-csrftoken"] = csrf_token

            # Use Instagram App ID from environment or fallback to a common one
            ig_app_id = os.getenv('INSTAGRAM_CSE_ID') or os.getenv('INSTAGRAM_APPID_UID') or '1217981644879628'
            headers["x-ig-app-id"] = ig_app_id

        # Use Playwright for Instagram images to maintain authentication context
        if 'instagram.com' in image_url or 'fbcdn.net' in image_url:
            return self.loop.run_until_complete(self._download_image_playwright(image_url, max_retries))

        # Use requests session for non-Instagram images
        for attempt in range(max_retries + 1):
            try:
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying download after {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)

                resp = self.ig_session.get(image_url, headers=headers, stream=True, timeout=15)
                resp.raise_for_status()

                with local_path.open('wb') as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)

                self.logger.info(f"Downloaded image to {local_path}")
                return local_path

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    self.logger.warning(f"403 Forbidden for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All retry attempts failed for {image_url} - 403 Forbidden")
                        return None
                elif e.response.status_code == 429:
                    self.logger.warning(f"Rate limited for {image_url} (attempt {attempt + 1}/{max_retries + 1})")
                    if attempt == max_retries:
                        self.logger.error(f"All retry attempts failed for {image_url} - Rate limited")
                        return None
                else:
                    self.logger.error(f"HTTP error {e.response.status_code} for {image_url}")
                    return None
            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {image_url}: {e}")
                if attempt == max_retries:
                    self.logger.exception(f"All retry attempts failed for {image_url}")
                    return None

        return None
        

    def ocr_image_to_text(self, local_path: Path) -> str:
        """
        Performs OCR (Optical Character Recognition) on a given image file and returns the extracted text.

        The method attempts to open the image at the specified local path, resizes it if its largest dimension is less than 800 pixels,
        converts it to grayscale, enhances its contrast, and binarizes it before passing it to Tesseract OCR for text extraction.

        Args:
            local_path (Path): The file path to the local image to be processed.

        Returns:
            str: The text extracted from the image. Returns an empty string if the image cannot be opened or OCR fails.

        Logs:
            Exceptions encountered during image opening or OCR processing are logged with the associated file path.
        """
        try:
            img = Image.open(local_path).convert('RGB')
        except Exception:
            self.logger.exception(f"Cannot open image {local_path}")
            return ""
        max_dim = max(img.size)
        if max_dim < 800:
            scale = 800/max_dim
            img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
        paddle_text = self._extract_text_paddleocr(img)
        if paddle_text:
            self.logger.info("ocr_image_to_text(): PaddleOCR succeeded for %s", local_path)
            return paddle_text
        gray = img.convert('L')
        bw = ImageEnhance.Contrast(gray).enhance(1.5).point(lambda x:0 if x<128 else 255,'1')
        try:
            return pytesseract.image_to_string(bw, lang='eng', config='--oem 3 --psm 6')
        except Exception:
            self.logger.exception(f"OCR failed on {local_path}")
            return ""

    def _get_paddle_ocr_engine(self):
        """Lazily initialize and cache the PaddleOCR engine."""
        cls = type(self)
        if cls._paddle_ocr_engine is not None:
            return cls._paddle_ocr_engine
        if cls._paddle_ocr_init_attempted:
            return None
        cls._paddle_ocr_init_attempted = True
        if PaddleOCR is None:
            self.logger.info("_get_paddle_ocr_engine(): PaddleOCR not installed, using Tesseract fallback")
            return None
        try:
            cls._paddle_ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                show_log=False,
            )
            self.logger.info("_get_paddle_ocr_engine(): PaddleOCR initialized successfully")
        except Exception as exc:
            self.logger.warning("_get_paddle_ocr_engine(): PaddleOCR initialization failed: %s", exc)
            cls._paddle_ocr_engine = None
        return cls._paddle_ocr_engine

    def _extract_text_paddleocr(self, image: Image.Image) -> str:
        """Extract text with PaddleOCR and return joined lines, or an empty string on failure."""
        engine = self._get_paddle_ocr_engine()
        if engine is None:
            return ""
        try:
            result = engine.ocr(np.array(image), cls=True)
        except Exception as exc:
            self.logger.warning("_extract_text_paddleocr(): PaddleOCR failed: %s", exc)
            return ""

        lines: list[str] = []
        for block in result or []:
            if not isinstance(block, list):
                continue
            for item in block:
                if not item or len(item) < 2:
                    continue
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and text_info:
                    detected_text = str(text_info[0] or "").strip()
                    if detected_text:
                        lines.append(detected_text)
        return "\n".join(lines).strip()

    @staticmethod
    def _local_image_to_data_url(local_path: Path) -> str | None:
        """Encode a local image as a data URL for multimodal OpenAI requests."""
        suffix = local_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(suffix)
        if not mime_type:
            return None
        try:
            encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
        except Exception:
            logger.exception("_local_image_to_data_url(): Failed to read image %s", local_path)
            return None
        return f"data:{mime_type};base64,{encoded}"

    def _process_local_image_path_with_vision(
        self,
        local_path: Path,
        canonical_url: str,
        parent_url: str,
        source: str,
        keywords: str,
        page_context_text: str | None = None,
    ) -> int:
        """Try multimodal image extraction before falling back to OCR."""
        image_data_url = self._local_image_to_data_url(local_path)
        if not image_data_url:
            self.logger.info(
                "_process_local_image_path_with_vision(): Unsupported screenshot type for vision path: %s",
                local_path,
            )
            return 0

        prompt_basis_url = canonical_url or parent_url
        vision_context = _build_image_context_text(
            ocr_text="Use the screenshot image as the primary source of truth.",
            parent_url=canonical_url,
            source=source,
            page_context_text=page_context_text,
        )
        found = [kw for kw in self.keywords_list if kw.lower() in vision_context.lower()]
        if not found:
            found = [source] if source else ["event"]

        prompt_type = resolve_prompt_type(prompt_basis_url, fallback_prompt_type='default')
        prompt_text, schema_type = self.llm_handler.generate_prompt(prompt_basis_url, vision_context, prompt_type)
        max_prompt_length = int(config['crawling']['prompt_max_length'])
        if len(prompt_text) > max_prompt_length:
            self.logger.warning(
                "_process_local_image_path_with_vision(): Prompt exceeds max length for %s",
                local_path,
            )
            return 0

        openai_model = _VISION_MODEL
        self.logger.info(
            "_process_local_image_path_with_vision(): Querying vision model %s for %s",
            openai_model,
            local_path,
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.llm_handler.query_openai,
                    prompt_text,
                    openai_model,
                    image_data_url,
                    schema_type,
                )
                llm_response = future.result(timeout=_VISION_REQUEST_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            self.logger.warning(
                "_process_local_image_path_with_vision(): Vision query timed out after %ss for %s",
                _VISION_REQUEST_TIMEOUT_SECONDS,
                local_path,
            )
            return 0
        except Exception:
            self.logger.exception(
                "_process_local_image_path_with_vision(): Vision query failed for %s",
                local_path,
            )
            return 0

        if not llm_response:
            self.logger.warning(
                "_process_local_image_path_with_vision(): Vision model returned no response for %s",
                local_path,
            )
            return 0

        parsed_result = self.llm_handler.extract_and_parse_json(llm_response, prompt_basis_url, schema_type)
        if not parsed_result:
            self.logger.warning(
                "_process_local_image_path_with_vision(): Vision model produced no parseable events for %s",
                local_path,
            )
            return 0

        events_df = pd.DataFrame(parsed_result)
        events_df = self.llm_handler._apply_url_context_to_events_df(
            events_df=events_df,
            url=prompt_basis_url,
            parent_url=parent_url,
        )
        if events_df.empty:
            self.logger.warning(
                "_process_local_image_path_with_vision(): Vision model parsed empty events for %s",
                local_path,
            )
            return 0

        self.db_handler.write_events_to_db(
            events_df,
            prompt_basis_url,
            parent_url,
            source,
            keywords,
            write_method="vision_extraction",
            provider="openai",
            model=_VISION_MODEL,
            prompt_type="vision_extraction",
            decision_reason="vision_success",
        )
        self.logger.info(
            "_process_local_image_path_with_vision(): Vision model wrote %d event(s) for %s",
            len(events_df),
            local_path,
        )
        return int(len(events_df))


    def get_image_links(self) -> pd.DataFrame:
        """
        Retrieves and combines image links from a CSV file and Instagram links from the database.

        This method reads image link data from a CSV file specified in the configuration and queries the database
        for Instagram-related links using a parameterized SQL query. The results from both sources are concatenated
        into a single DataFrame, and duplicate entries (based on 'link' and 'parent_url') are removed.

        Instagram CDN URLs (fbcdn.net) contain time-limited access tokens that expire after 24-48 hours.
        Old Instagram URLs are filtered out to avoid unnecessary 403/404 errors.

        Returns:
            pd.DataFrame: A DataFrame containing unique image links and their associated metadata from both the CSV and database.
            If the CSV file does not exist, returns an empty DataFrame.

        Logs:
            - An error if the CSV file is not found.
            - An info message with the count of Instagram URLs retrieved from the database.
        """
        csv_path = Path(self.config['input']['images'])
        if not csv_path.exists():
            self.logger.error(f"CSV not found: {csv_path}")
            return pd.DataFrame()
        df_csv = pd.read_csv(csv_path)

        # Parameterized query using sqlalchemy.text for safe ILIKE
        # Only get Instagram URLs from the last 24 hours to avoid stale CDN tokens
        query = text("""
            SELECT link, parent_url, source, keywords, relevant, crawl_try, time_stamp
            FROM urls
            WHERE link ILIKE :link_pattern
              AND time_stamp >= (CURRENT_TIMESTAMP - INTERVAL '24 hours')
        """ )
        params = {'link_pattern': '%instagram%'}
        df_db = pd.read_sql(query, self.db_handler.conn, params=params)
        self.logger.info(f"get_image_links(): Retrieved {df_db.shape[0]} Instagram URLs from the database (filtered to last 24 hours).")

        # Combine CSV and DB results
        df = pd.concat([df_csv, df_db], ignore_index=True)

        # Remove duplicates
        df = df.drop_duplicates(['link', 'parent_url']).reset_index(drop=True)

        # Filter out old fbcdn URLs from CSV as well (they also expire)
        if 'time_stamp' in df.columns:
            df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')
            before_filter = len(df)
            # For fbcdn URLs, only keep recent ones (24 hours)
            df = df[
                ~(df['link'].str.contains('fbcdn.net', case=False, na=False) &
                  (df['time_stamp'].isna() |
                   (pd.Timestamp.now() - df['time_stamp'] > pd.Timedelta(hours=24))))
            ]
            after_filter = len(df)
            if before_filter > after_filter:
                self.logger.info(f"get_image_links(): Filtered out {before_filter - after_filter} stale fbcdn URLs (older than 24 hours).")

        # Discover additional Instagram URLs by keyword, respecting the configured run cap.
        limit = int(self.config['crawling'].get('urls_run_limit', 0) or 0)
        if limit > 0:
            remaining_capacity = max(limit - len(df), 0)
        else:
            remaining_capacity = 0

        if remaining_capacity > 0:
            existing_links = set(df.get('link', pd.Series(dtype=str)).astype(str))
            discovered_df = self._discover_instagram_keyword_links(existing_links, remaining_capacity)
            if not discovered_df.empty:
                self.logger.info(
                    "get_image_links(): Discovered %d Instagram URL(s) from keyword search within remaining capacity=%d.",
                    len(discovered_df),
                    remaining_capacity,
                )
                df = pd.concat([df, discovered_df], ignore_index=True)
                df = df.drop_duplicates(['link', 'parent_url']).reset_index(drop=True)

        # Restrict to configured maximum
        if limit > 0:
            df = df.iloc[:limit]

        return df


    def process_webpage_url(self, page_url: str, parent_url: str, source: str, keywords: str) -> None:
        """
        Processes a webpage URL by fetching its content, extracting visible text,
        searching for specified keywords, and handling embedded images.

        Args:
            page_url (str): The URL of the webpage to process.
            parent_url (str): The parent page URL for context.
            source (str): The source identifier for categorization.
            keywords(str): The keywords from the original source (.csv or db).
        Returns:
            None
        """
        if page_url in self.urls_visited:
            self.logger.info(f"process_webpage_url(): Already visited {page_url}, skipping.")
            return
        self.urls_visited.add(page_url)

        # Instantiate url_row
        url_row = (page_url, parent_url, source, '', False, 1, datetime.now())
        rendered_html = ""
        post_limit = self.images_per_page_limit
        if post_limit <= 0:
            post_limit = int(self.config.get('crawling', {}).get('max_website_urls', 10) or 10)
        instagram_post_links: list[str] = []
        prefiltered_img_urls: list[str] = []
        skip_profile_llm = False

        resp = self._fetch_page_response(page_url, timeout_seconds=10)
        if resp is None:
            self.db_handler.write_url_to_db(url_row)
            return

        # strip scripts/styles & extract visible text
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        visible_text = soup.get_text(separator=' ')
        text = ' '.join(visible_text.split())
        self.logger.info(f"process_webpage_url(): Extracted text length {len(text)}")
        self.logger.info(f"process_webpage_url(): Extracted text is:\n{text}")

        # Fallback to Playwright for JS‑heavy or too‑short scrapes
        if not text or len(text) < 200 or is_instagram_url(page_url):
            self.logger.info(f"process_webpage_url(): Falling back to Playwright for {page_url}")
            pw_text = self._extract_dynamic_page_text(page_url)
            if not pw_text:
                self.logger.info(f"process_webpage_url(): Playwright extraction failed for {page_url}")
                self.db_handler.write_url_to_db(url_row)
                return
            text = pw_text
            self.logger.info(f"process_webpage_url(): Playwright-extracted text length {len(text)}")
            self.logger.info(f"process_webpage_url(): Playwright text is:\n{text}")

        if not text:
            self.logger.info(f"process_webpage_url(): No visible text in {page_url}")
            self.db_handler.write_url_to_db(url_row)
            return

        if is_instagram_url(page_url):
            rendered_html = self.loop.run_until_complete(self.read_extract.page.content())
            if not is_instagram_post_detail_url(page_url):
                instagram_post_links = _extract_instagram_post_links(rendered_html, page_url)
                if not instagram_post_links:
                    instagram_post_links = self.loop.run_until_complete(
                        self._extract_instagram_post_links_playwright(page_url, post_limit)
                    )
                prefiltered_img_urls = _extract_rankable_image_urls(rendered_html, page_url)
                if _is_degraded_instagram_profile_text(text):
                    if instagram_post_links or prefiltered_img_urls:
                        skip_profile_llm = True
                        self.logger.info(
                            "process_webpage_url(): Instagram profile is degraded shell text, skipping profile-level LLM and using %d post link(s) / %d image candidate(s)",
                            len(instagram_post_links),
                            len(prefiltered_img_urls),
                        )
                    else:
                        self.logger.warning(
                            "process_webpage_url(): Instagram profile appears degraded and exposed no post links or image candidates for %s",
                            page_url,
                        )
                        screenshot_path = self.loop.run_until_complete(self._capture_page_screenshot(page_url))
                        if screenshot_path:
                            self._process_local_image_path(
                                screenshot_path,
                                page_url,
                                parent_url,
                                source,
                                keywords,
                                page_context_text=text,
                            )
                        else:
                            self.db_handler.write_url_to_db(url_row)
                        return

        # keyword filtering
        found = [kw for kw in self.keywords_list if kw.lower() in text.lower()]
        if not found:
            self.logger.info(f"process_webpage_url(): No keywords found in {page_url}")
            self.db_handler.write_url_to_db(url_row)
            return
        self.logger.info(f"process_webpage_url(): Keywords {found} found in {page_url}")

        # LLM processing
        if not skip_profile_llm:
            prompt_type = resolve_prompt_type(page_url, fallback_prompt_type='default')
            prompt_text, schema_type = self.llm_handler.generate_prompt(page_url, text, prompt_type)
            status = self.llm_handler.process_llm_response(
                page_url, parent_url, text, source, found, prompt_type
            )
            if status:
                self.logger.info(f"process_webpage_url(): LLM succeeded for {page_url}")
            else:
                self.logger.warning(f"process_webpage_url(): LLM produced no events for {page_url}")
        else:
            self.logger.info("process_webpage_url(): Skipped profile-level LLM for degraded Instagram profile %s", page_url)

        # extract and process images
        # pull down the browser’s rendered DOM
        if not rendered_html:
            rendered_html = self.loop.run_until_complete(
                self.read_extract.page.content()
            )
        if not instagram_post_links:
            instagram_post_links = _extract_instagram_post_links(rendered_html, page_url)
        if not instagram_post_links and is_instagram_url(page_url) and not is_instagram_post_detail_url(page_url):
            instagram_post_links = self.loop.run_until_complete(
                self._extract_instagram_post_links_playwright(page_url, post_limit)
            )
        if instagram_post_links:
            selected_post_links = instagram_post_links[:post_limit]
            self.logger.info(
                "process_webpage_url(): extracted %d Instagram post link(s), processing %d before page-image OCR",
                len(instagram_post_links),
                len(selected_post_links),
            )
            for post_url in selected_post_links:
                self.process_webpage_url(post_url, page_url, source, keywords)
            return
        img_urls = prefiltered_img_urls or _extract_rankable_image_urls(rendered_html, page_url)
        self.logger.info(
            "process_webpage_url(): considering %d raw image candidate(s) before ranking, cap=%d",
            len(img_urls),
            self.images_per_page_limit,
        )

        # Establish a minimum size for the img_urls.
        MIN_W, MIN_H = 500, 500

        scored_imgs: list[tuple[int, str]] = []
        for url in img_urls:
            try:
                # Use Playwright for Instagram images, requests for others
                if 'instagram.com' in url or 'fbcdn.net' in url:
                    # Use Playwright's authenticated context for Instagram images
                    async def fetch_instagram_image():
                        response = await self.read_extract.page.request.get(url)
                        if response.status == 200:
                            return await response.body()
                        else:
                            self.logger.info(f"Skipping {url}: Playwright returned status {response.status}")
                            return None

                    image_data = self.loop.run_until_complete(fetch_instagram_image())
                    if image_data:
                        img = Image.open(BytesIO(image_data))
                    else:
                        continue
                else:
                    # Use requests session for non-Instagram images
                    resp = self.ig_session.get(url, timeout=10)
                    resp.raise_for_status()
                    img = Image.open(BytesIO(resp.content))
            except Exception as e:
                self.logger.info(f"Skipping {url}: failed to download/open - {e}")
                continue

            w, h = img.size
            if w >= MIN_W and h >= MIN_H:
                score = _score_image_candidate(url, w, h)
                if score < 0:
                    self.logger.info("Skipping %s: low image candidate score=%s (%sx%s)", url, score, w, h)
                    continue
                scored_imgs.append((score, url))
            else:
                self.logger.info(f"Skipping {url}: too small ({w}x{h})")

        scored_imgs.sort(key=lambda item: item[0], reverse=True)
        per_page_process_limit = self.images_per_page_limit
        if per_page_process_limit <= 0:
            per_page_process_limit = int(self.config.get('crawling', {}).get('max_website_urls', 10) or 10)
        per_page_process_limit = min(per_page_process_limit, self.max_images_per_page)
        selected_imgs = scored_imgs[:per_page_process_limit]
        self.logger.info(
            "process_webpage_url(): selected %d ranked image(s) for OCR: %s",
            len(selected_imgs),
            [{"score": score, "url": url} for score, url in selected_imgs],
        )
        for index, (_, src) in enumerate(selected_imgs):
            use_vision_first = True
            if is_instagram_url(page_url):
                use_vision_first = index < self.instagram_vision_image_limit
            self.process_image_url(
                src,
                page_url,
                source,
                keywords,
                page_context_text=text,
                use_vision_first=use_vision_first,
                canonical_event_url=build_image_replay_url(page_url, src),
            )
        if not selected_imgs and is_instagram_url(page_url):
            screenshot_path = self.loop.run_until_complete(self._capture_page_screenshot(page_url))
            if screenshot_path:
                self._process_local_image_path(
                    screenshot_path,
                    page_url,
                    parent_url,
                    source,
                    keywords,
                    page_context_text=text,
                )

    def _process_local_image_path(
        self,
        local_path: Path,
        canonical_url: str,
        parent_url: str,
        source: str,
        keywords: str,
        page_context_text: str | None = None,
    ) -> bool:
        """Run OCR-first extraction against a local screenshot, then fall back to vision."""
        self.logger.info("_process_local_image_path(): Starting OCR for %s", local_path)
        ocr_attempted = True
        text = self.ocr_image_to_text(local_path)
        if not text:
            self.logger.info("_process_local_image_path(): No text extracted from screenshot %s", local_path)
            vision_events_written = self._process_local_image_path_with_vision(
                local_path,
                canonical_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
            )
            vision_success = vision_events_written > 0
            self._record_image_metric(
                link=canonical_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_screenshot",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="screenshot_empty_ocr",
                text_extracted=False,
                keywords_found=False,
                events_written=vision_events_written,
                ocr_attempted=ocr_attempted,
                ocr_succeeded=False,
                vision_attempted=True,
                vision_succeeded=vision_success,
                fallback_used=True,
            )
            return vision_success
        self.logger.info("_process_local_image_path(): Extracted text length %d characters", len(text))
        self.logger.info("_process_local_image_path(): Extracted text:\n%s", text)

        text, analysis = _prepend_image_date_hints(local_path, text)
        if analysis.poster_type == "single_event" and analysis.primary_date:
            self.logger.info(
                "_process_local_image_path(): Detected single-event date hint %s (%s) added to OCR text",
                analysis.primary_date,
                analysis.primary_day_of_week or "",
            )
        elif analysis.poster_type == "schedule_multi_event":
            self.logger.info(
                "_process_local_image_path(): Detected schedule-style poster with %d candidate dates",
                len(analysis.candidate_dates),
            )
        repeated_outcome = self._lookup_repeated_poster_outcome(
            source=source,
            analysis=analysis,
            ocr_text=text,
        )
        if repeated_outcome is not None:
            _, prior_events_written = repeated_outcome
            self.logger.info(
                "_process_local_image_path(): Reused repeated poster OCR signature for %s (prior_events_written=%d)",
                local_path,
                prior_events_written,
            )
            self._record_image_metric(
                link=canonical_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_screenshot",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=prior_events_written > 0,
                extraction_skipped=True,
                decision_reason="screenshot_repeated_poster_reuse",
                text_extracted=True,
                keywords_found=prior_events_written > 0,
                events_written=0,
                ocr_attempted=ocr_attempted,
                ocr_succeeded=True,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return prior_events_written > 0

        llm_text = _build_image_context_text(
            ocr_text=text,
            parent_url=canonical_url,
            source=source,
            page_context_text=page_context_text,
        )
        found = [kw for kw in self.keywords_list if kw.lower() in llm_text.lower()]
        if not found:
            self.logger.info("_process_local_image_path(): No relevant keywords in screenshot OCR for %s", local_path)
            vision_events_written = self._process_local_image_path_with_vision(
                local_path,
                canonical_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
            )
            vision_success = vision_events_written > 0
            self._record_image_metric(
                link=canonical_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_screenshot",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="screenshot_no_keywords",
                text_extracted=True,
                keywords_found=False,
                events_written=vision_events_written,
                ocr_attempted=ocr_attempted,
                ocr_succeeded=True,
                vision_attempted=True,
                vision_succeeded=vision_success,
                fallback_used=True,
            )
            return vision_success

        prompt_basis_url = canonical_url or parent_url
        prompt_type = resolve_prompt_type(prompt_basis_url, fallback_prompt_type='default')
        prompt_text, schema_type = self.llm_handler.generate_prompt(prompt_basis_url, llm_text, prompt_type)
        if len(prompt_text) > config['crawling']['prompt_max_length']:
            self.logger.warning("_process_local_image_path(): Prompt exceeds maximum length for %s", local_path)
            vision_events_written = self._process_local_image_path_with_vision(
                local_path,
                canonical_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
            )
            vision_success = vision_events_written > 0
            self._record_image_metric(
                link=canonical_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_screenshot",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="screenshot_prompt_overflow",
                text_extracted=True,
                keywords_found=True,
                events_written=vision_events_written,
                ocr_attempted=ocr_attempted,
                ocr_succeeded=True,
                vision_attempted=True,
                vision_succeeded=vision_success,
                fallback_used=True,
            )
            return vision_success

        llm_result = self.llm_handler.process_llm_response(
            prompt_basis_url,
            parent_url,
            llm_text,
            source,
            found,
            prompt_type,
        )
        if llm_result:
            llm_events_written = int(getattr(llm_result, "events_written", 1))
            self.logger.info("_process_local_image_path(): LLM processing succeeded for screenshot %s", local_path)
            self._record_image_metric(
                link=canonical_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_screenshot",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=True,
                extraction_skipped=False,
                decision_reason="screenshot_ocr_llm_success",
                text_extracted=True,
                keywords_found=True,
                events_written=llm_events_written,
                ocr_attempted=ocr_attempted,
                ocr_succeeded=True,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            self._remember_repeated_poster_outcome(
                source=source,
                analysis=analysis,
                ocr_text=text,
                events_written=llm_events_written,
            )
            return True
        self.logger.warning("_process_local_image_path(): LLM processing did not produce any events for screenshot %s", local_path)
        self._remember_repeated_poster_outcome(
            source=source,
            analysis=analysis,
            ocr_text=text,
            events_written=0,
        )
        vision_events_written = self._process_local_image_path_with_vision(
            local_path,
            canonical_url,
            parent_url,
            source,
            keywords,
            page_context_text=page_context_text,
        )
        vision_success = vision_events_written > 0
        self._record_image_metric(
            link=canonical_url,
            parent_url=parent_url,
            source=source,
            keywords=keywords,
            archetype="image_screenshot",
            access_succeeded=True,
            extraction_attempted=True,
            extraction_succeeded=vision_success,
            extraction_skipped=False,
            decision_reason="screenshot_ocr_llm_no_events",
            text_extracted=True,
            keywords_found=True,
            events_written=vision_events_written,
            ocr_attempted=ocr_attempted,
            ocr_succeeded=True,
            vision_attempted=True,
            vision_succeeded=vision_success,
            fallback_used=True,
        )
        return vision_success


    def process_image_url(
        self,
        image_url: str,
        parent_url: str,
        source: str,
        keywords: str,
        page_context_text: str | None = None,
        use_vision_first: bool = True,
        canonical_event_url: str | None = None,
    ) -> None:
        """
        Processes an image URL by downloading the image, extracting text using OCR, 
        searching for specified keywords, and handling the result with an LLM handler.
        Args:
            image_url (str): The URL of the image to process.
            parent_url (str): The URL of the parent page where the image was found.
            source (str): The source identifier for the image.
            keywords(str): The keywords that were taken from the original input.
        Returns:
            None
        Workflow:
            - Skips processing if the image URL has already been visited or if related events exist.
            - Downloads the image from the provided URL.
            - Extracts text from the image using OCR.
            - Searches for keywords in the extracted text.
            - If keywords are found, generates a prompt and processes the response using the LLM handler.
        """
        event_url = str(canonical_event_url or image_url).strip() or image_url
        self.logger.info(
            "process_image_url(): Starting processing for %s (event_url=%s)",
            image_url,
            event_url,
        )

        # Instantiate url_row
        url_row = (event_url, parent_url, source, keywords, False, 1, datetime.now())

        # Skip if already visited
        if event_url in self.urls_visited:
            self.logger.info(f"process_image_url(): Already visited {event_url}, skipping.")
            return
        self.urls_visited.add(event_url)
        self.logger.info(f"process_image_url(): Marked {event_url} as visited.")

        if self._has_processed_image_identity(image_url):
            self.logger.info(
                "process_image_url(): Reused previously processed image identity for %s, skipping.",
                image_url,
            )
            self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_attempted=False,
                access_succeeded=True,
                extraction_attempted=False,
                extraction_succeeded=False,
                extraction_skipped=True,
                decision_reason="reused_processed_image_identity",
                text_extracted=False,
                keywords_found=False,
                events_written=0,
                ocr_attempted=False,
                ocr_succeeded=False,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return

        # Skip if events already exist
        if self.db_handler.check_image_events_exist(event_url):
            self.logger.info(f"process_image_url(): Events already exist for {event_url}, skipping OCR.")
            self._mark_image_identity_processed(image_url)
            url_row = (event_url, parent_url, source, keywords, True, 1, datetime.now())
            self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_attempted=False,
                access_succeeded=True,
                extraction_attempted=False,
                extraction_succeeded=False,
                extraction_skipped=True,
                decision_reason="events_already_exist",
                text_extracted=False,
                keywords_found=False,
                events_written=0,
                ocr_attempted=False,
                ocr_succeeded=False,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return
        
        # Check and see if we should process this url
        if not self.db_handler.should_process_url(event_url):
            self.logger.info(f"process_image_url(): should_process_url for {event_url}, returned False.")
            self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_attempted=False,
                access_succeeded=False,
                extraction_attempted=False,
                extraction_succeeded=False,
                extraction_skipped=True,
                decision_reason="should_process_url_false",
                text_extracted=False,
                keywords_found=False,
                events_written=0,
                ocr_attempted=False,
                ocr_succeeded=False,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return

        # Download the image
        self.logger.info(f"process_image_url(): Downloading image from {image_url}")
        path = self.download_image(image_url)
        if not path:
            self.logger.error(f"process_image_url(): download_image() failed for {image_url}")
            self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=image_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=False,
                extraction_attempted=True,
                extraction_succeeded=False,
                extraction_skipped=False,
                decision_reason="download_failed",
                text_extracted=False,
                keywords_found=False,
                events_written=0,
                ocr_attempted=False,
                ocr_succeeded=False,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return
        self.logger.info(f"process_image_url(): Image saved to {path}")
        self._mark_image_identity_processed(image_url)

        # Run OCR
        self.logger.info(f"process_image_url(): Running OCR on {path}")
        text = self.ocr_image_to_text(path)
        if not text:
            self.logger.info(f"process_image_url(): No text extracted from {path}, trying vision fallback if allowed.")
            vision_attempted = bool(use_vision_first)
            vision_events_written = 0
            if use_vision_first:
                vision_events_written = self._process_local_image_path_with_vision(
                Path(path),
                parent_url or image_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
                )
            vision_success = vision_events_written > 0
            if vision_success:
                self.logger.info(
                    "process_image_url(): Vision fallback extraction succeeded for %s after empty OCR",
                    event_url,
                )
            else:
                self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="empty_ocr",
                text_extracted=False,
                keywords_found=False,
                events_written=vision_events_written,
                ocr_attempted=True,
                ocr_succeeded=False,
                vision_attempted=vision_attempted,
                vision_succeeded=vision_success,
                fallback_used=vision_attempted,
            )
            return
        self.logger.info(f"process_image_url(): Extracted text length {len(text)} characters")
        self.logger.info(f"process_image_url(): Extracted text: \n{text}")

        text, analysis = _prepend_image_date_hints(Path(path), text)
        if analysis.poster_type == "single_event" and analysis.primary_date:
            self.logger.info(
                "process_image_url(): Detected single-event date hint %s (%s) added to OCR text",
                analysis.primary_date,
                analysis.primary_day_of_week or "",
            )
        elif analysis.poster_type == "schedule_multi_event":
            self.logger.info(
                "process_image_url(): Detected schedule-style poster with %d candidate dates",
                len(analysis.candidate_dates),
            )
        repeated_outcome = self._lookup_repeated_poster_outcome(
            source=source,
            analysis=analysis,
            ocr_text=text,
        )
        if repeated_outcome is not None:
            _, prior_events_written = repeated_outcome
            self.logger.info(
                "process_image_url(): Reused repeated poster OCR signature for %s (prior_events_written=%d)",
                event_url,
                prior_events_written,
            )
            self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=prior_events_written > 0,
                extraction_skipped=True,
                decision_reason="reused_repeated_poster_pattern",
                text_extracted=True,
                keywords_found=prior_events_written > 0,
                events_written=0,
                ocr_attempted=True,
                ocr_succeeded=True,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            return

        # Keyword filtering
        llm_text = _build_image_context_text(
            ocr_text=text,
            parent_url=parent_url,
            source=source,
            page_context_text=page_context_text,
        )

        found = [kw for kw in self.keywords_list if kw.lower() in llm_text.lower()]
        if not found:
            self.logger.info(f"process_image_url(): No relevant keywords in OCR text for {image_url}")
            vision_attempted = bool(use_vision_first)
            vision_events_written = 0
            if use_vision_first:
                vision_events_written = self._process_local_image_path_with_vision(
                Path(path),
                parent_url or image_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
                )
            vision_success = vision_events_written > 0
            if vision_success:
                self.logger.info(
                    "process_image_url(): Vision fallback extraction succeeded for %s after OCR keyword miss",
                    event_url,
                )
            else:
                self.db_handler.write_url_to_db(url_row)
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="ocr_no_keywords",
                text_extracted=True,
                keywords_found=False,
                events_written=vision_events_written,
                ocr_attempted=True,
                ocr_succeeded=True,
                vision_attempted=vision_attempted,
                vision_succeeded=vision_success,
                fallback_used=vision_attempted,
            )
            return
        self.logger.info(f"process_image_url(): Found keywords {found} in image {image_url}")

        # LLM prompt & response
        prompt_basis_url = parent_url or image_url
        prompt_type = resolve_prompt_type(prompt_basis_url, fallback_prompt_type='default')
        prompt_text, schema_type = self.llm_handler.generate_prompt(prompt_basis_url, llm_text, prompt_type)
        if len(prompt_text) > config['crawling']['prompt_max_length']:
            logging.warning(f"def process_image_url(): Prompt for URL {event_url} exceeds maximum length. Skipping LLM query.")
            vision_attempted = bool(use_vision_first)
            vision_events_written = 0
            if use_vision_first:
                vision_events_written = self._process_local_image_path_with_vision(
                Path(path),
                event_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
                )
            vision_success = vision_events_written > 0
            if vision_success:
                self.logger.info(
                    "process_image_url(): Vision fallback extraction succeeded for %s after OCR prompt overflow",
                    event_url,
                )
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="ocr_prompt_overflow",
                text_extracted=True,
                keywords_found=True,
                events_written=vision_events_written,
                ocr_attempted=True,
                ocr_succeeded=True,
                vision_attempted=vision_attempted,
                vision_succeeded=vision_success,
                fallback_used=vision_attempted,
            )
            return 
        
        self.logger.info(
            "process_image_url(): Generated prompt using basis_url=%s for image_url=%s",
            prompt_basis_url,
            event_url,
        )
        llm_result = self.llm_handler.process_llm_response(
            event_url, parent_url, llm_text, source, found, prompt_type
        )
        if llm_result:
            llm_events_written = int(getattr(llm_result, "events_written", 1))
            self.logger.info(f"process_image_url(): LLM processing succeeded for {event_url}")
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=True,
                extraction_skipped=False,
                decision_reason="ocr_llm_success",
                text_extracted=True,
                keywords_found=True,
                events_written=llm_events_written,
                ocr_attempted=True,
                ocr_succeeded=True,
                vision_attempted=False,
                vision_succeeded=False,
                fallback_used=False,
            )
            self._remember_repeated_poster_outcome(
                source=source,
                analysis=analysis,
                ocr_text=text,
                events_written=llm_events_written,
            )
        else:
            self.logger.warning(f"process_image_url(): LLM processing did not produce any events for {event_url}")
            self._remember_repeated_poster_outcome(
                source=source,
                analysis=analysis,
                ocr_text=text,
                events_written=0,
            )
            vision_attempted = bool(use_vision_first)
            vision_events_written = 0
            if use_vision_first:
                vision_events_written = self._process_local_image_path_with_vision(
                Path(path),
                event_url,
                parent_url,
                source,
                keywords,
                page_context_text=page_context_text,
                )
            vision_success = vision_events_written > 0
            if vision_success:
                self.logger.info(
                    "process_image_url(): Vision fallback extraction succeeded for %s after OCR/LLM miss",
                    event_url,
                )
            self._record_image_metric(
                link=event_url,
                parent_url=parent_url,
                source=source,
                keywords=keywords,
                archetype="image_url",
                access_succeeded=True,
                extraction_attempted=True,
                extraction_succeeded=vision_success,
                extraction_skipped=False,
                decision_reason="ocr_llm_no_events",
                text_extracted=True,
                keywords_found=True,
                events_written=vision_events_written,
                ocr_attempted=True,
                ocr_succeeded=True,
                vision_attempted=vision_attempted,
                vision_succeeded=vision_success,
                fallback_used=vision_attempted,
            )

    def _normalize_replay_events(self, parsed_result: list[dict], canonical_url: str) -> list[dict]:
        """Normalize parsed image events into replay-row shape without DB writes."""
        normalized_events: list[dict] = []
        for event in parsed_result or []:
            if not isinstance(event, dict):
                continue
            event_raw = dict(event)
            mentioned_url = str(event.get("url") or "").strip()
            if mentioned_url:
                event_raw["mentioned_url"] = mentioned_url
            normalized_events.append(
                {
                    "event_name": str(event.get("event_name") or "").strip(),
                    "start_date": str(event.get("start_date") or "").strip()[:10],
                    "start_time": str(event.get("start_time") or "").strip(),
                    "source": str(event.get("source") or "").strip(),
                    "location": str(event.get("location") or "").strip(),
                    "url": canonical_url,
                    "raw": event_raw,
                }
            )
        return normalized_events

    def _parse_replay_events_from_local_image(
        self,
        *,
        local_path: Path,
        canonical_url: str,
        parent_url: str,
        source: str,
        page_context_text: str | None = None,
    ) -> list[dict]:
        """Run OCR-first replay extraction for one local image without writing to the database."""
        text = self.ocr_image_to_text(local_path)
        if text:
            text, _ = _prepend_image_date_hints(local_path, text)
            llm_text = _build_image_context_text(
                ocr_text=text,
                parent_url=parent_url,
                source=source,
                page_context_text=page_context_text,
            )
            prompt_type = resolve_prompt_type(parent_url or canonical_url, fallback_prompt_type="default")
            prompt_text, schema_type = self.llm_handler.generate_prompt(parent_url or canonical_url, llm_text, prompt_type)
            if schema_type:
                llm_response = self.llm_handler.query_llm(canonical_url, prompt_text, schema_type)
                parsed_result = self.llm_handler.extract_and_parse_json(llm_response, canonical_url, schema_type)
                normalized_events = self._normalize_replay_events(parsed_result or [], canonical_url)
                if normalized_events:
                    return normalized_events

        image_data_url = self._local_image_to_data_url(local_path)
        if not image_data_url:
            return []
        vision_context = _build_image_context_text(
            ocr_text="Use the screenshot image as the primary source of truth.",
            parent_url=parent_url,
            source=source,
            page_context_text=page_context_text,
        )
        prompt_type = resolve_prompt_type(parent_url or canonical_url, fallback_prompt_type="default")
        prompt_text, schema_type = self.llm_handler.generate_prompt(parent_url or canonical_url, vision_context, prompt_type)
        if not schema_type:
            return []
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.llm_handler.query_openai,
                    prompt_text,
                    _VISION_MODEL,
                    image_data_url,
                    schema_type,
                )
                llm_response = future.result(timeout=_VISION_REQUEST_TIMEOUT_SECONDS)
        except Exception:
            return []
        parsed_result = self.llm_handler.extract_and_parse_json(llm_response, canonical_url, schema_type)
        return self._normalize_replay_events(parsed_result or [], canonical_url)

    def fetch_replay_events_for_image_replay_url(self, replay_url: str) -> dict:
        """Replay one Instagram image child URL by re-targeting the specific poster image."""
        canonical_url = str(replay_url or "").strip()
        parent_url = strip_image_replay_fragment(canonical_url)
        page_text = self._extract_dynamic_page_text(parent_url) or ""
        rendered_html = self.loop.run_until_complete(self.read_extract.page.content())
        image_urls = _extract_rankable_image_urls(rendered_html, parent_url)
        target_image_url = ""
        for image_url in image_urls:
            if build_image_replay_url(parent_url, image_url) == canonical_url:
                target_image_url = image_url
                break
        if not target_image_url:
            return {
                "ok": False,
                "category": "no_event_extracted_replay",
                "details": "instagram_image_not_found",
                "events": [],
            }
        local_path = self.download_image(target_image_url)
        if not local_path:
            return {
                "ok": False,
                "category": "url_unreachable_replay",
                "details": "instagram_image_download_failed",
                "events": [],
            }
        source = (urlparse(parent_url).netloc or "instagram").strip()
        normalized_events = self._parse_replay_events_from_local_image(
            local_path=Path(local_path),
            canonical_url=canonical_url,
            parent_url=parent_url,
            source=source,
            page_context_text=page_text,
        )
        if not normalized_events:
            return {
                "ok": False,
                "category": "no_event_extracted_replay",
                "details": "instagram_image_replay_no_events",
                "events": [],
            }
        return {"ok": True, "category": "", "details": "instagram_image_replay", "events": normalized_events}


    def process_images(self) -> None:
        """
        Fetches all image and webpage links, then processes each one.

        Workflow:
        1. Retrieve links from CSV and database via `get_image_links()`.
        2. Log the total number of links to process.
        3. Iterate through each entry, logging its index and details.
        4. Dispatch to `process_image_url()` for direct image URLs,
           or to `process_webpage_url()` for webpage URLs.
        5. Log completion when done.
        """
        # Get the file name of the code that is running
        file_name = os.path.basename(__file__)

        # Count events and urls at start
        start_df = self.db_handler.count_events_urls_start(file_name)

        self.logger.info("process_images(): Starting batch processing of image/webpage links.")
        initial_df = self.get_image_links()
        self.logger.info("process_images(): Retrieved %d initial links to process.", len(initial_df))
        self._process_image_rows(initial_df, phase_label="initial")

        remaining_capacity = self._remaining_url_capacity()
        if remaining_capacity > 0:
            refreshed_df = self.get_image_links()
            refreshed_df = self._filter_unvisited_rows(refreshed_df)
            if not refreshed_df.empty:
                self.logger.info(
                    "process_images(): Retrieved %d newly discovered links on final refresh (remaining_capacity=%d).",
                    len(refreshed_df),
                    remaining_capacity,
                )
                self._process_image_rows(refreshed_df, phase_label="final_refresh")
            else:
                self.logger.info("process_images(): Final refresh found no new image/webpage links.")
        else:
            self.logger.info("process_images(): Skipping final refresh because urls_run_limit has been reached.")

        self.logger.info("process_images(): Completed processing all links.")
        self._log_processing_summary()

        self.db_handler.count_events_urls_end(start_df, __file__)
        logging.info(f"Wrote events and urls statistics to: {file_name}")

    def _remaining_url_capacity(self) -> int:
        """Return how many more URLs may be visited within the configured run cap."""
        limit = int(self.config.get("crawling", {}).get("urls_run_limit", 0) or 0)
        if limit <= 0:
            return 0
        return max(limit - len(self.urls_visited), 0)

    def _filter_unvisited_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter queued rows down to links that have not already been visited."""
        if df.empty or "link" not in df.columns:
            return df
        mask = ~df["link"].astype(str).isin(self.urls_visited)
        return df.loc[mask].reset_index(drop=True)

    def _process_image_rows(self, df: pd.DataFrame, phase_label: str) -> None:
        """Process queued image/webpage rows while honoring the configured URL run cap."""
        if df.empty:
            self.logger.info("process_images(): %s queue is empty.", phase_label)
            return

        total = len(df)
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            if self._remaining_url_capacity() <= 0:
                self.logger.info(
                    "process_images(): Reached urls_run_limit during %s queue at item %d/%d.",
                    phase_label,
                    idx,
                    total,
                )
                break

            url, parent, source, keywords, *_ = row
            self.logger.info(
                "process_images(): [%s %d/%d] url=%s, parent=%s, source=%s",
                phase_label,
                idx,
                total,
                url,
                parent,
                source,
            )

            if not self.db_handler.should_process_url(url):
                self.logger.info("process_images(): should_process_url returned False for url: %s", url)
                continue

            if self.is_image_url(url):
                self.logger.info("process_images(): Detected direct image URL (%s), invoking OCR pipeline.", url)
                self.process_image_url(url, parent, source, keywords)
            else:
                self.logger.info("process_images(): Detected webpage URL (%s), extracting embedded images.", url)
                self.process_webpage_url(url, parent, source, keywords)

    def _record_image_metric(
        self,
        *,
        link: str,
        parent_url: str,
        source: str,
        keywords: str | list[str],
        archetype: str,
        access_attempted: bool = True,
        access_succeeded: bool,
        extraction_attempted: bool,
        extraction_succeeded: bool,
        extraction_skipped: bool,
        decision_reason: str,
        text_extracted: bool,
        keywords_found: bool,
        events_written: int,
        ocr_attempted: bool,
        ocr_succeeded: bool,
        vision_attempted: bool,
        vision_succeeded: bool,
        fallback_used: bool,
    ) -> None:
        """Persist image-stage telemetry and update per-run counters."""
        self._ensure_image_telemetry_state()
        self.telemetry_counts["total_urls"] += 1
        self.telemetry_counts["access_attempted"] += int(access_attempted)
        self.telemetry_counts["access_succeeded"] += int(access_succeeded)
        self.telemetry_counts["text_extracted"] += int(text_extracted)
        self.telemetry_counts["keywords_found"] += int(keywords_found)
        self.telemetry_counts["urls_with_events"] += int(events_written > 0)
        self.telemetry_counts["events_written"] += int(events_written)
        self.telemetry_counts["ocr_attempted"] += int(ocr_attempted)
        self.telemetry_counts["ocr_succeeded"] += int(ocr_succeeded)
        self.telemetry_counts["vision_attempted"] += int(vision_attempted)
        self.telemetry_counts["vision_succeeded"] += int(vision_succeeded)
        self.telemetry_counts["fallback_used"] += int(fallback_used)

        if hasattr(self, "db_handler") and hasattr(self.db_handler, "write_url_scrape_metric"):
            self.db_handler.write_url_scrape_metric(
                {
                    "run_id": self.run_id,
                    "step_name": self.step_name,
                    "link": link,
                    "parent_url": parent_url,
                    "source": source,
                    "keywords": keywords,
                    "archetype": archetype,
                    "extraction_attempted": extraction_attempted,
                    "extraction_succeeded": extraction_succeeded,
                    "extraction_skipped": extraction_skipped,
                    "decision_reason": decision_reason,
                    "handled_by": "images.py",
                    "routing_reason": decision_reason,
                    "access_attempted": access_attempted,
                    "access_succeeded": access_succeeded,
                    "text_extracted": text_extracted,
                    "keywords_found": keywords_found,
                    "events_written": events_written,
                    "ocr_attempted": ocr_attempted,
                    "ocr_succeeded": ocr_succeeded,
                    "vision_attempted": vision_attempted,
                    "vision_succeeded": vision_succeeded,
                    "fallback_used": fallback_used,
                    "links_discovered": 0,
                    "links_followed": 0,
                    "time_stamp": datetime.now(),
                }
            )

    def _ensure_image_telemetry_state(self) -> None:
        """Backfill telemetry attributes for lightweight test instances built via __new__."""
        if not hasattr(self, "telemetry_counts") or self.telemetry_counts is None:
            self.telemetry_counts = Counter()
        if not hasattr(self, "run_id") or self.run_id is None:
            self.run_id = os.getenv("DS_RUN_ID", "na")
        if not hasattr(self, "step_name") or self.step_name is None:
            self.step_name = os.getenv("DS_STEP_NAME", "images")

    def _log_processing_summary(self) -> None:
        """Log an easy-to-read summary of image scraper success rates for the current run."""
        self._ensure_image_telemetry_state()
        total_urls = int(self.telemetry_counts.get("total_urls", 0) or 0)
        if total_urls <= 0:
            self.logger.info("process_images(): No structured image telemetry was recorded for this run.")
            return

        def _rate(numerator: str, denominator: int) -> str:
            value = int(self.telemetry_counts.get(numerator, 0) or 0)
            if denominator <= 0:
                return "N/A"
            return f"{value / denominator:.1%} ({value}/{denominator})"

        access_attempted = int(self.telemetry_counts.get("access_attempted", 0) or 0)
        ocr_attempted = int(self.telemetry_counts.get("ocr_attempted", 0) or 0)
        vision_attempted = int(self.telemetry_counts.get("vision_attempted", 0) or 0)
        text_extracted = int(self.telemetry_counts.get("text_extracted", 0) or 0)
        self.logger.info("IMAGES SCRAPER SUMMARY")
        self.logger.info("  URL access success rate: %s", _rate("access_succeeded", access_attempted))
        self.logger.info("  Text extracted rate: %s", _rate("text_extracted", access_attempted))
        self.logger.info("  Keyword hit rate: %s", _rate("keywords_found", text_extracted))
        self.logger.info("  Event extraction success rate: %s", _rate("urls_with_events", access_attempted))
        self.logger.info("  OCR success rate: %s", _rate("ocr_succeeded", ocr_attempted))
        self.logger.info("  Vision success rate: %s", _rate("vision_succeeded", vision_attempted))
        self.logger.info(
            "  Vision fallback usage rate: %s",
            _rate("fallback_used", access_attempted),
        )


if __name__ == '__main__':
    start = datetime.now()

    scraper = ImageScraper(config)
    scraper.process_images()

    scraper.loop.run_until_complete(scraper.read_extract.close())
    scraper.loop.close()

    logger.info(f"Finished in {datetime.now()-start}\n")
