"""
Centralized URL/page classification for scraper routing and prompt strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict
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


@dataclass(frozen=True)
class RoutingDecision:
    url: str
    current_step: str
    owner_step: str
    allow: bool
    routing_reason: str
    classification: PageClassification


@dataclass(frozen=True)
class ClassificationDecision:
    classification: PageClassification
    confidence: float
    stage: str
    features: Dict[str, Any]


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


def is_listing_page_url(url: str) -> bool:
    """
    Return True when URL strongly indicates a multi-event listing page.
    """
    low = _safe_url(url).lower()
    try:
        parsed = urlparse(_safe_url(url))
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
    except Exception:
        path = low
        query = ""

    if is_event_detail_url(url):
        return False

    path_tokens = (
        "/calendar",
        "/events",
        "/upcoming",
        "/schedule",
        "/list",
        "/month",
    )
    query_tokens = (
        "month_limit=",
        "year_limit=",
        "view=list",
        "view=month",
        "calendar",
    )
    return any(token in path for token in path_tokens) or any(token in query for token in query_tokens)


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

    if is_event_detail_url(url):
        return PageClassification(
            url=url,
            archetype="simple_page",
            owner_step="scraper.py",
            prompt_type=url,
            is_social=False,
            is_calendar=False,
            is_event_detail=True,
            subtype="event_detail",
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
    if is_listing_page_url(url):
        archetype = "incomplete_event"
    elif not is_event_detail_url(url) and (event_like_links >= 3 or (has_listing_signal and page_links_count >= 6)):
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


def classify_page_with_confidence(
    *,
    url: str,
    visible_text: str = "",
    page_links_count: int = 0,
    calendar_sources_count: int = 0,
    calendar_ids_count: int = 0,
) -> ClassificationDecision:
    """
    Two-stage page classification:
    - Stage A: deterministic routing rules
    - Stage B: structural scoring with confidence
    """
    base = classify_page(
        url=url,
        visible_text=visible_text,
        page_links_count=page_links_count,
        calendar_sources_count=calendar_sources_count,
        calendar_ids_count=calendar_ids_count,
    )

    # Stage A: deterministic classes are already reliable.
    deterministic_subtypes = {
        "google_calendar",
        "facebook_event_detail",
        "instagram_post_detail",
        "social_listing_or_profile",
        "eventbrite_event_detail",
        "event_detail",
    }
    if base.subtype in deterministic_subtypes:
        return ClassificationDecision(
            classification=base,
            confidence=0.99,
            stage="rule",
            features={
                "event_like_links": int(page_links_count or 0),
                "calendar_sources_count": int(calendar_sources_count or 0),
                "calendar_ids_count": int(calendar_ids_count or 0),
                "deterministic_subtype": base.subtype,
            },
        )

    # Stage B: structural scoring for generic websites.
    low_text = str(visible_text or "").lower()
    event_like_links = int(page_links_count or 0)
    listing_signal = any(
        token in low_text
        for token in (
            "view all",
            "load more",
            "more events",
            "upcoming events",
            "calendar",
            "list view",
            "month view",
        )
    )
    repeated_date_tokens = sum(low_text.count(token) for token in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))
    listing_score = 0
    if is_listing_page_url(url):
        listing_score += 3
    if event_like_links >= 3:
        listing_score += 2
    if listing_signal:
        listing_score += 1
    if repeated_date_tokens >= 4:
        listing_score += 1

    event_signal = has_event_signal(low_text)
    feature_payload: Dict[str, Any] = {
        "event_like_links": event_like_links,
        "listing_signal": bool(listing_signal),
        "repeated_date_tokens": int(repeated_date_tokens),
        "listing_score": int(listing_score),
        "event_signal": bool(event_signal),
    }

    if listing_score >= 3:
        classification = PageClassification(
            url=base.url,
            archetype="incomplete_event",
            owner_step=base.owner_step,
            prompt_type=base.prompt_type,
            is_social=base.is_social,
            is_calendar=base.is_calendar,
            is_event_detail=False,
            subtype="incomplete_event",
        )
        confidence = min(0.70 + 0.05 * min(listing_score, 4), 0.90)
        return ClassificationDecision(
            classification=classification,
            confidence=confidence,
            stage="structural",
            features=feature_payload,
        )

    if event_signal:
        confidence = 0.68 if event_like_links <= 2 else 0.60
        return ClassificationDecision(
            classification=base,
            confidence=confidence,
            stage="structural",
            features=feature_payload,
        )

    # Unknown/low signal: default to safer listing strategy, avoid parent extraction.
    fallback = PageClassification(
        url=base.url,
        archetype="incomplete_event",
        owner_step=base.owner_step,
        prompt_type=base.prompt_type,
        is_social=base.is_social,
        is_calendar=base.is_calendar,
        is_event_detail=False,
        subtype="low_confidence_fallback",
    )
    return ClassificationDecision(
        classification=fallback,
        confidence=0.52,
        stage="structural",
        features=feature_payload,
    )


def resolve_prompt_type(url: str, fallback_prompt_type: str = "default") -> str:
    """
    Resolve prompt_type from centralized classification.
    """
    c = classify_page(url=url)
    if c.prompt_type:
        return c.prompt_type
    return str(fallback_prompt_type or "default")


def evaluate_step_ownership(url: str, current_step: str, explicit_edge_case: bool = False) -> RoutingDecision:
    """
    Central routing decision for scraper ownership boundaries.

    Args:
        url: URL being processed.
        current_step: Expected step name (e.g., "scraper.py", "fb.py").
        explicit_edge_case: True only for rd_ext.py edge-case execution.
    """
    classification = classify_page(url=url)
    owner_step = classification.owner_step or "scraper.py"
    step_name = str(current_step or "").strip() or "unknown"

    if step_name == owner_step:
        return RoutingDecision(
            url=url,
            current_step=step_name,
            owner_step=owner_step,
            allow=True,
            routing_reason="owner_match",
            classification=classification,
        )

    if step_name == "rd_ext.py":
        # rd_ext can provide explicit edge-case extraction only for scraper-owned URLs.
        allowed = bool(explicit_edge_case and owner_step == "scraper.py")
        reason = "edge_case_delegate_allowed" if allowed else f"owned_by_{owner_step}"
        return RoutingDecision(
            url=url,
            current_step=step_name,
            owner_step=owner_step,
            allow=allowed,
            routing_reason=reason,
            classification=classification,
        )

    return RoutingDecision(
        url=url,
        current_step=step_name,
        owner_step=owner_step,
        allow=False,
        routing_reason=f"owned_by_{owner_step}",
        classification=classification,
    )
