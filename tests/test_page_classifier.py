import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from page_classifier import classify_page, resolve_prompt_type


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

