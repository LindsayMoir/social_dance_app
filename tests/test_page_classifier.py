import os
import sys

os.environ.setdefault("PAGE_CLASSIFIER_ML_ENABLED", "0")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from page_classifier import (
    ClassificationDecision,
    PageClassification,
    apply_historical_routing_memory,
    classify_page,
    classify_page_with_confidence,
    evaluate_step_ownership,
    is_email_like_input,
    is_event_detail_url,
    resolve_prompt_type,
)
from page_classifier_features import extract_html_features


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


def test_classify_eventbrite_organizer_owned_by_ebs() -> None:
    c = classify_page(url="https://www.eventbrite.ca/o/silent-dj-victoria-31599471691")
    assert c.owner_step == "ebs.py"
    assert c.subtype == "eventbrite_organizer"
    assert c.is_event_detail is False


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


def test_classify_with_confidence_listing_structural() -> None:
    d = classify_page_with_confidence(
        url="https://example.org/calendar/music?month_limit=4&year_limit=2026",
        visible_text="Upcoming events view all more events",
        page_links_count=8,
    )
    assert d.classification.archetype == "incomplete_event"
    assert d.stage in {"rule", "structural"}
    assert d.confidence >= 0.70


def test_classify_with_confidence_low_signal_fallback_to_incomplete() -> None:
    d = classify_page_with_confidence(
        url="https://example.org/about",
        visible_text="welcome to our organization",
        page_links_count=1,
    )
    assert d.classification.archetype == "incomplete_event"
    assert d.stage == "structural"
    assert d.confidence < 0.70


def test_email_like_input_routes_deterministically_to_emails() -> None:
    assert is_email_like_input("dancevictoria1@shaw.ca") is True
    c = classify_page(url="dancevictoria1@shaw.ca")
    assert c.archetype == "other"
    assert c.owner_step == "emails.py"
    assert c.subtype == "email_input"


def test_apply_historical_routing_memory_overrides_ambiguous_decision() -> None:
    decision = ClassificationDecision(
        classification=PageClassification(
            url="https://example.org/upcoming",
            archetype="simple_page",
            owner_step="scraper.py",
            prompt_type="https://example.org/upcoming",
            is_social=False,
            is_calendar=False,
            is_event_detail=False,
            subtype="simple_page",
        ),
        confidence=0.68,
        stage="structural",
        features={"listing_score": 2},
    )
    updated = apply_historical_routing_memory(
        decision,
        memory_hint={
            "archetype": "incomplete_event",
            "owner_step": "rd_ext.py",
            "subtype": "incomplete_event",
            "sample_count": 4,
            "dominance": 1.0,
            "avg_confidence": 0.87,
            "stage_mode": "ml",
        },
    )
    assert updated.stage == "memory"
    assert updated.classification.archetype == "incomplete_event"
    assert updated.classification.owner_step == "rd_ext.py"
    assert updated.features["memory_sample_count"] == 4


def test_apply_historical_routing_memory_does_not_override_deterministic_rule() -> None:
    decision = ClassificationDecision(
        classification=PageClassification(
            url="https://www.google.com/calendar/event?eid=test",
            archetype="google_calendar",
            owner_step="scraper.py",
            prompt_type="https://www.google.com/calendar/event?eid=test",
            is_social=False,
            is_calendar=True,
            is_event_detail=False,
            subtype="google_calendar",
        ),
        confidence=0.99,
        stage="rule",
        features={},
    )
    updated = apply_historical_routing_memory(
        decision,
        memory_hint={
            "archetype": "incomplete_event",
            "owner_step": "rd_ext.py",
            "subtype": "incomplete_event",
            "sample_count": 5,
            "dominance": 1.0,
            "avg_confidence": 0.9,
            "stage_mode": "structural",
        },
    )
    assert updated is decision


def test_extract_html_features_for_single_event_detail_page() -> None:
    html = """
    <html>
      <head>
        <script type="application/ld+json">
          {"@context":"https://schema.org","@type":"Event","name":"Friday Salsa Social"}
        </script>
      </head>
      <body>
        <article class="event-card">
          <h1>Friday Salsa Social</h1>
          <p>Friday March 20 at 7:30 PM</p>
          <p>919 Douglas St</p>
          <a href="https://tickets.example.com/register">Register now</a>
          <iframe src="https://www.google.com/maps/embed?pb=abc"></iframe>
        </article>
      </body>
    </html>
    """
    features = extract_html_features(
        html,
        url="https://example.org/event/friday-salsa-social",
    )
    assert features["feature_schema_event_present"] is True
    assert features["feature_schema_event_count"] == 1
    assert features["feature_address_count"] >= 1
    assert features["feature_time_count"] >= 1
    assert features["feature_date_count"] >= 1
    assert features["feature_ticket_link_count"] >= 1
    assert features["feature_map_embed_present"] is True
    assert features["feature_single_event_signal"] is True
    assert features["feature_visible_text_event_score"] > 0


def test_extract_html_features_for_listing_page() -> None:
    html = """
    <html>
      <body>
        <div class="event-card"><a href="/event/one">Event One</a><p>March 21</p></div>
        <div class="event-card"><a href="/event/two">Event Two</a><p>March 22</p></div>
        <div class="event-card"><a href="/event/three">Event Three</a><p>March 23</p></div>
      </body>
    </html>
    """
    features = extract_html_features(
        html,
        url="https://example.org/events",
    )
    assert features["feature_event_detail_link_count"] == 3
    assert features["feature_listing_card_count"] >= 3
    assert features["feature_single_event_signal"] is False


def test_classify_with_confidence_merges_html_features() -> None:
    html = """
    <html>
      <body>
        <article>
          <h1>Example Event</h1>
          <p>Saturday March 21 at 8:00 PM</p>
          <p>123 Main St</p>
          <a href="/tickets">Tickets</a>
        </article>
      </body>
    </html>
    """
    html_features = extract_html_features(
        html,
        url="https://example.org/event/example-event",
    )
    d = classify_page_with_confidence(
        url="https://example.org/event/example-event",
        html_features=html_features,
    )
    assert d.features["feature_address_count"] >= 1
    assert d.features["feature_ticket_link_count"] >= 1
    assert d.features["feature_visible_text_event_score"] > 0
