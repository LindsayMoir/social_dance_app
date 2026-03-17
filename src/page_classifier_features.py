"""
HTML-derived feature extraction for page classification.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from bs4 import BeautifulSoup


_EVENT_SCHEMA_TYPES = {"event", "musicevent", "danceevent", "educationevent", "festival"}
_TICKET_TOKENS = ("ticket", "tickets", "register", "registration", "buy", "admission", "eventbrite")
_EVENT_LINK_HINTS = ("/event/", "/events/", "/show/", "/calendar/event", "-tickets-", "/nm_event/")
_MONTH_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.' -]+\s(?:st|street|ave|avenue|rd|road|dr|drive|blvd|boulevard|ln|lane|way|hwy|highway)\b",
    re.IGNORECASE,
)


def extract_html_features(html: str, *, url: str = "") -> dict[str, Any]:
    """
    Extract deterministic classifier features from HTML content.
    """
    soup = BeautifulSoup(str(html or ""), "html.parser")
    schema_event_count = _count_event_schema(soup)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    visible_text = soup.get_text(" ", strip=True)
    visible_text_lower = visible_text.lower()

    links = [anchor.get("href", "").strip() for anchor in soup.find_all("a", href=True)]
    link_texts = [" ".join(anchor.stripped_strings).lower() for anchor in soup.find_all("a", href=True)]
    iframe_sources = [frame.get("src", "").strip().lower() for frame in soup.find_all("iframe", src=True)]
    embed_links = [href.lower() for href in links]

    address_count = len(_ADDRESS_RE.findall(visible_text))
    time_count = len(_TIME_RE.findall(visible_text))
    date_count = len(_MONTH_RE.findall(visible_text)) + len(_WEEKDAY_RE.findall(visible_text))
    ticket_link_count = sum(1 for href in links if any(token in href.lower() for token in _TICKET_TOKENS))
    ticket_link_count += sum(1 for text in link_texts if any(token in text for token in _TICKET_TOKENS))
    event_detail_link_count = sum(1 for href in links if any(token in href.lower() for token in _EVENT_LINK_HINTS))
    map_embed_present = any("google.com/maps" in src or "maps.google" in src for src in iframe_sources)
    map_embed_present = map_embed_present or any("google.com/maps" in href or "maps.google" in href for href in embed_links)
    mailto_present = any(href.lower().startswith("mailto:") for href in links) or bool(_EMAIL_RE.search(visible_text))
    listing_card_count = _estimate_listing_card_count(soup)
    single_event_signal = (
        date_count >= 1
        and time_count >= 1
        and address_count >= 1
        and listing_card_count <= 1
        and event_detail_link_count <= 2
    )

    return {
        "feature_schema_event_present": schema_event_count > 0,
        "feature_schema_event_count": schema_event_count,
        "feature_address_count": address_count,
        "feature_time_count": time_count,
        "feature_date_count": date_count,
        "feature_ticket_link_count": ticket_link_count,
        "feature_event_detail_link_count": event_detail_link_count,
        "feature_map_embed_present": map_embed_present,
        "feature_mailto_present": mailto_present,
        "feature_single_event_signal": single_event_signal,
        "feature_listing_card_count": listing_card_count,
        "feature_visible_text_event_score": _compute_visible_text_event_score(visible_text_lower, url=url),
    }


def _count_event_schema(soup: BeautifulSoup) -> int:
    count = 0
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_payload = script.string or script.get_text(strip=True)
        if not raw_payload:
            continue
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            continue
        count += _count_event_schema_nodes(payload)
    return count


def _count_event_schema_nodes(node: Any) -> int:
    if isinstance(node, list):
        return sum(_count_event_schema_nodes(item) for item in node)
    if isinstance(node, dict):
        type_value = node.get("@type")
        if isinstance(type_value, list):
            type_names = {str(item).lower() for item in type_value}
        else:
            type_names = {str(type_value).lower()} if type_value else set()
        current = int(bool(type_names & _EVENT_SCHEMA_TYPES))
        return current + sum(_count_event_schema_nodes(value) for value in node.values())
    return 0


def _estimate_listing_card_count(soup: BeautifulSoup) -> int:
    class_tokens: list[str] = []
    for tag in soup.find_all(True):
        class_tokens.extend(str(token).lower() for token in (tag.get("class") or []))

    counts = Counter(class_tokens)
    return max(
        (
            count
            for name, count in counts.items()
            if any(token in name for token in ("event", "card", "listing", "show", "calendar"))
        ),
        default=0,
    )


def _compute_visible_text_event_score(text: str, *, url: str = "") -> int:
    score = 0
    for token in ("ticket", "tickets", "register", "venue", "location"):
        score += 2 * text.count(token)
    for token in ("event", "events", "dance", "class", "workshop", "lesson", "social"):
        score += text.count(token)
    score += len(_MONTH_RE.findall(text))
    score += len(_WEEKDAY_RE.findall(text))
    if any(token in str(url or "").lower() for token in _EVENT_LINK_HINTS):
        score += 2
    return score
