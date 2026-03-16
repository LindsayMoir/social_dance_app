import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from page_classifier import classify_page, evaluate_step_ownership, resolve_prompt_type, is_event_detail_url


def test_classify_facebook_event_detail_owned_by_fb() -> None:
    c = classify_page(url="https://www.facebook.com/events/1482463219409922/")
    assert c.owner_step == "fb.py"
    assert c.subtype == "facebook_event_detail"
    assert c.archetype == "complicated_page"
    assert c.prompt_type == "fb"


def test_classify_instagram_post_owned_by_fb() -> None:
    c = classify_page(url="https://www.instagram.com/p/DT_jAxwD8Wy/")
    assert c.owner_step == "fb.py"
    assert c.subtype == "instagram_post_detail"
    assert c.prompt_type == "fb"


def test_classify_eventbrite_detail_owned_by_ebs() -> None:
    c = classify_page(url="https://www.eventbrite.ca/e/example-tickets-1984648436900")
    assert c.owner_step == "ebs.py"
    assert c.subtype == "eventbrite_event_detail"
    assert c.is_event_detail is True


def test_resolve_prompt_type_defaults_to_url_or_fb() -> None:
    assert resolve_prompt_type("https://www.facebook.com/events/123") == "fb"
    assert resolve_prompt_type("https://example.com/events/foo", fallback_prompt_type="default").startswith("https://")


def test_is_event_detail_url_supports_query_style_event_paths() -> None:
    assert is_event_detail_url("https://www.visitpenticton.com/event?external=true") is True
    assert is_event_detail_url(
        "https://www.visitpenticton.com/nm_event/pentictons-sunday-open-mic-jams-at-the-hub-on-martin?external=true"
    ) is True


def test_livevictoria_calendar_is_listing_not_simple_page() -> None:
    c = classify_page(url="https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026")
    assert c.archetype == "incomplete_event"
    assert c.is_event_detail is False


def test_livevictoria_show_is_detail_simple_page() -> None:
    c = classify_page(url="https://livevictoria.com/show/732324/view")
    assert c.archetype == "simple_page"
    assert c.is_event_detail is True


def test_routing_contract_scraper_rejects_facebook_event_detail() -> None:
    d = evaluate_step_ownership("https://www.facebook.com/events/1482463219409922/", current_step="scraper.py")
    assert d.allow is False
    assert d.owner_step == "fb.py"
    assert d.routing_reason == "owned_by_fb.py"


def test_routing_contract_ebs_accepts_eventbrite_detail() -> None:
    d = evaluate_step_ownership(
        "https://www.eventbrite.ca/e/example-tickets-1984648436900",
        current_step="ebs.py",
    )
    assert d.allow is True
    assert d.owner_step == "ebs.py"
    assert d.routing_reason == "owner_match"


def test_routing_contract_rd_ext_only_explicit_scraper_owned() -> None:
    allowed = evaluate_step_ownership(
        "https://livevictoria.com/show/732324/view",
        current_step="rd_ext.py",
        explicit_edge_case=True,
    )
    denied = evaluate_step_ownership(
        "https://www.facebook.com/events/1482463219409922/",
        current_step="rd_ext.py",
        explicit_edge_case=True,
    )
    assert allowed.allow is True
    assert allowed.routing_reason == "edge_case_delegate_allowed"
    assert denied.allow is False
    assert denied.owner_step == "fb.py"
