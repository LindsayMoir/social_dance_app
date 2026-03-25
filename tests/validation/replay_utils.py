#!/usr/bin/env python3
"""Pure replay and validation utility helpers."""

from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
import re
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup, Comment

from page_classifier import is_google_calendar_like_url, is_social_url


def normalize_text_value(value: Any) -> str:
    """Normalize loose text fields for replay comparison."""
    text = str(value or "").strip().lower()
    if text in {"", "none", "null", "nan", "<na>"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_optional_int(value: Any) -> int | None:
    """Normalize optional integer fields and reject non-positive values."""
    try:
        if value is None or str(value).strip() == "":
            return None
        coerced = int(value)
        return coerced if coerced > 0 else None
    except (TypeError, ValueError):
        return None


def normalize_dance_style_tokens(value: Any) -> tuple[str, ...]:
    """Normalize dance-style fields into sorted comparable tokens."""
    text = str(value or "").strip().lower()
    if not text:
        return tuple()
    text = text.replace("&", ",").replace("/", ",")
    parts = [re.sub(r"\s+", " ", token).strip() for token in text.split(",")]
    normalized = sorted({token for token in parts if token})
    return tuple(normalized)


def descriptions_equivalent(baseline_description: str, replay_description: str) -> bool:
    """Treat near-identical descriptions as equivalent."""
    if not baseline_description or not replay_description:
        return False
    if baseline_description == replay_description:
        return True
    if baseline_description in replay_description or replay_description in baseline_description:
        return True
    return SequenceMatcher(None, baseline_description, replay_description).ratio() >= 0.85


def name_similarity(left: str, right: str) -> float:
    """Compute fuzzy similarity for event-name comparison."""
    return SequenceMatcher(None, str(left or ""), str(right or "")).ratio()


def name_contains_variant(left: str, right: str) -> bool:
    """Treat substring and pre-@ variants as equivalent event names."""
    normalized_left = re.sub(r"\s+", " ", str(left or "").strip().lower())
    normalized_right = re.sub(r"\s+", " ", str(right or "").strip().lower())
    if not normalized_left or not normalized_right:
        return False
    if normalized_left in normalized_right or normalized_right in normalized_left:
        return True
    left_base = normalized_left.split("@", 1)[0].strip()
    right_base = normalized_right.split("@", 1)[0].strip()
    if left_base and right_base and (left_base in right_base or right_base in left_base):
        return True
    return False


def is_placeholder_source(value: str) -> bool:
    """Return True for non-informative source placeholder values."""
    normalized = str(value or "").strip().lower()
    return normalized in {"", "unknown", "none", "null", "extracted_text", "extracted text", "text extracted"}


def is_calendar_event_url(url: str) -> bool:
    """Return True for Google Calendar-like event URLs."""
    return is_google_calendar_like_url(url)


def normalize_url_value(value: Any) -> str:
    """Normalize URLs for replay comparison."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").rstrip("/")
        fragment = parsed.fragment if str(parsed.fragment or "").startswith("image=") else ""
        return urlunparse((scheme, netloc, path, "", parsed.query, fragment))
    except Exception:
        return text.lower().rstrip("/")


def normalize_date_value(value: Any) -> str:
    """Normalize date values to ISO format when possible."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except Exception:
        return text[:10]


def normalize_time_value(value: Any) -> str:
    """Normalize 12-hour and 24-hour time values to HH:MM:SS."""
    text = str(value or "").strip()
    if not text:
        return ""
    for fmt in ("%I:%M:%S %p", "%I:%M %p"):
        try:
            parsed = datetime.strptime(text[:11] if fmt == "%I:%M:%S %p" else text[:8], fmt)
            return parsed.strftime("%H:%M:%S")
        except Exception:
            continue
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed = datetime.strptime(text[:8], fmt) if fmt == "%H:%M:%S" else datetime.strptime(text[:5], fmt)
            return parsed.strftime("%H:%M:%S")
        except Exception:
            continue
    return text


def times_equivalent_with_12h_guard(left: str, right: str) -> bool:
    """Treat exact and 12-hour drifted times as equivalent."""
    if not left or not right:
        return left == right
    if left == right:
        return True
    try:
        parsed_left = datetime.strptime(left[:8], "%H:%M:%S")
        parsed_right = datetime.strptime(right[:8], "%H:%M:%S")
        return abs((parsed_left - parsed_right).total_seconds()) in {12 * 3600}
    except Exception:
        return False


def is_social_platform_url(url: str) -> bool:
    """Return True for supported social platform URLs."""
    return is_social_url(url)


def extract_visible_text_for_replay(html: str) -> str:
    """Extract visible text from raw HTML for replay parsing."""
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
        visible = " ".join(chunk for chunk in text_chunks if chunk).strip()
        return visible[:20000]
    except Exception:
        return str(html or "")[:20000]


def extract_replay_links(url: str, html: str) -> list[str]:
    """Extract normalized href targets from replay HTML."""
    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    links: list[str] = []
    for node in soup.find_all("a", href=True):
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("/"):
            href = urljoin(url, href)
        elif not href.startswith("http"):
            href = urljoin(url, href)
        links.append(href)
    return links
