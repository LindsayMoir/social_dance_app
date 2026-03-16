"""
Centralized URL/page classification for scraper routing and prompt strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse


_FB_EVENT_RE = re.compile(r"/events/(\d+)")
_EB_TICKET_RE = re.compile(r"-tickets-(\d+)")


@dataclass(frozen=True)
class PageClassification:
    url: str
    archetype: str
    owner_step: str
    prompt_type: str
    is_social: bool
    is_calendar: bool
    is_event_detail: bool
    subtype: str


def _safe_url(url: str) -> str:
    return str(url or "").strip()


def _host(url: str) -> str:
    try:
        return (urlparse(_safe_url(url)).netloc or "").lower()
    except Exception:
        return ""


def is_facebook_url(url: str) -> bool:
    host = _host(url)
    return host == "facebook.com" or host.endswith(".facebook.com")


def is_instagram_url(url: str) -> bool:
    host = _host(url)
    return host == "instagram.com" or host.endswith(".instagram.com")


def is_social_url(url: str) -> bool:
    return is_facebook_url(url) or is_instagram_url(url)


def is_facebook_event_detail_url(url: str) -> bool:
    if not is_facebook_url(url):
        return False
    try:
        path = (urlparse(_safe_url(url)).path or "")
    except Exception:
        return False
    return _FB_EVENT_RE.search(path) is not None


def is_instagram_post_detail_url(url: str) -> bool:
    if not is_instagram_url(url):
        return False
    try:
        path = (urlparse(_safe_url(url)).path or "").strip("/")
    except Exception:
        return False
    parts = [p for p in path.split("/") if p]
    return len(parts) >= 2 and parts[0].lower() in {"p", "reel", "tv"}


def is_eventbrite_url(url: str) -> bool:
    return "eventbrite." in _host(url)


def is_eventbrite_event_detail_url(url: str) -> bool:
    if not is_eventbrite_url(url):
        return False
    try:
        path = (urlparse(_safe_url(url)).path or "")
    except Exception:
        return False
    return _EB_TICKET_RE.search(path) is not None


def is_google_calendar_like_url(url: str) -> bool:
    low = _safe_url(url).lower()
    return (
        "calendar.google.com" in low
        or "google.com/calendar" in low
        or "@group.calendar.google.com" in low
        or "%40group.calendar.google.com" in low
        or "/calendar/ical/" in low
    )


def is_event_detail_url(url: str) -> bool:
    low = _safe_url(url).lower()
    try:
        path = (urlparse(_safe_url(url)).path or "").lower()
    except Exception:
        path = low
    path_trimmed = path.rstrip("/")
    if path_trimmed.endswith("/events"):
        return False
    if any(token in path for token in ("/events/month", "/events/list", "/calendar", "/schedule", "/upcoming")):
        return False
    if is_facebook_event_detail_url(url) or is_instagram_post_detail_url(url) or is_eventbrite_event_detail_url(url):
        return True
    if any(token in low for token in ("/event/", "/events/", "/show/", "/tickets/", "/nm_event/")):
        return True
    # Accept query-style event pages, e.g. /event?external=true
    return bool(re.search(r"/event(?:[/?#]|$)", low))


def has_event_signal(text: str) -> bool:
    low = str(text or "").lower()
    tokens = (
        "event", "events", "calendar", "schedule", "social", "dance",
        "workshop", "class", "lesson", "friday", "saturday", "sunday",
        "monday", "tuesday", "wednesday", "thursday",
    )
    return any(token in low for token in tokens)


def classify_page(
    *,
    url: str,
    visible_text: str = "",
    page_links_count: int = 0,
    calendar_sources_count: int = 0,
    calendar_ids_count: int = 0,
) -> PageClassification:
    """
    Return centralized page classification used by all scrapers.
    """
    url = _safe_url(url)
    low_url = url.lower()
    low_text = str(visible_text or "").lower()

    if is_google_calendar_like_url(url) or calendar_sources_count > 0 or calendar_ids_count > 0:
        return PageClassification(
            url=url,
            archetype="google_calendar",
            owner_step="scraper.py",
            prompt_type=url,
            is_social=False,
            is_calendar=True,
            is_event_detail=False,
            subtype="google_calendar",
        )

    if is_facebook_event_detail_url(url):
        return PageClassification(
            url=url,
            archetype="complicated_page",
            owner_step="fb.py",
            prompt_type="fb",
            is_social=True,
            is_calendar=False,
            is_event_detail=True,
            subtype="facebook_event_detail",
        )

    if is_instagram_post_detail_url(url):
        return PageClassification(
            url=url,
            archetype="complicated_page",
            owner_step="fb.py",
            prompt_type="fb",
            is_social=True,
            is_calendar=False,
            is_event_detail=True,
            subtype="instagram_post_detail",
        )

    if is_facebook_url(url) or is_instagram_url(url):
        return PageClassification(
            url=url,
            archetype="complicated_page",
            owner_step="fb.py",
            prompt_type="fb",
            is_social=True,
            is_calendar=False,
            is_event_detail=False,
            subtype="social_listing_or_profile",
        )

    if is_eventbrite_event_detail_url(url):
        return PageClassification(
            url=url,
            archetype="simple_page",
            owner_step="ebs.py",
            prompt_type=url,
            is_social=False,
            is_calendar=False,
            is_event_detail=True,
            subtype="eventbrite_event_detail",
        )

    event_like_links = int(page_links_count or 0)
    has_listing_signal = any(
        sig in low_text
        for sig in (
            "view all",
            "load more",
            "more events",
            "upcoming events",
            "read more",
            "learn more",
            "tickets",
        )
    )
    if not is_event_detail_url(url) and (event_like_links >= 3 or (has_listing_signal and page_links_count >= 6)):
        archetype = "incomplete_event"
    elif has_event_signal(low_text):
        archetype = "simple_page"
    else:
        archetype = "other"

    return PageClassification(
        url=url,
        archetype=archetype,
        owner_step="scraper.py",
        prompt_type=url,
        is_social=False,
        is_calendar=False,
        is_event_detail=is_event_detail_url(url),
        subtype=archetype,
    )


def resolve_prompt_type(url: str, fallback_prompt_type: str = "default") -> str:
    """
    Resolve prompt_type from centralized classification.
    """
    c = classify_page(url=url)
    if c.prompt_type:
        return c.prompt_type
    return str(fallback_prompt_type or "default")
