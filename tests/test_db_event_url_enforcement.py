import sys

import pandas as pd


sys.path.insert(0, "src")

from db import DatabaseHandler


def test_enforce_event_url_values_fills_from_default_url_for_non_email_rows() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "Friday Social", "source": "Some Venue", "url": ""},
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="https://venue.example/events/friday-social",
        parent_url="",
        source="Some Venue",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "https://venue.example/events/friday-social"


def test_enforce_event_url_values_allows_email_identifier_for_email_context() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "Mailing List Event", "source": "Email Newsletter", "url": ""},
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="updates@example.org",
        parent_url="email inbox",
        source="Email Newsletter",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "updates@example.org"


def test_enforce_event_url_values_drops_non_email_rows_without_any_url_fallback() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "No URL Event", "source": "Venue A", "url": ""},
            {"event_name": "Valid URL Event", "source": "Venue B", "url": "https://venue.example/event"},
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="",
        parent_url="",
        source="Venue A",
    )

    assert len(out) == 1
    assert out.loc[0, "event_name"] == "Valid URL Event"
    assert out.loc[0, "url"] == "https://venue.example/event"


def test_enforce_event_url_values_drops_example_dot_com_event_without_fallback() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Jazz Night",
                "source": "The Loft Pub Victoria",
                "url": "https://example.com/theloft/event/jazz-night-apr-27-2026",
            },
            {
                "event_name": "Valid URL Event",
                "source": "Venue B",
                "url": "https://venue.example/event",
            },
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="",
        parent_url="",
        source="The Loft Pub Victoria",
    )

    assert len(out) == 1
    assert out.loc[0, "event_name"] == "Valid URL Event"
    assert out.loc[0, "url"] == "https://venue.example/event"


def test_enforce_event_url_values_replaces_example_dot_com_event_with_valid_fallback() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Jazz Night",
                "source": "The Loft Pub Victoria",
                "url": "https://example.com/theloft/event/jazz-night-apr-27-2026",
            },
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="https://loftpubvictoria.com/events/month/",
        parent_url="",
        source="The Loft Pub Victoria",
    )

    assert len(out) == 1
    assert out.loc[0, "event_name"] == "Jazz Night"
    assert out.loc[0, "url"] == "https://loftpubvictoria.com/events/month/"


def test_enforce_event_url_values_uses_parent_when_default_is_example_dot_com() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "Parent URL Event", "source": "Venue C", "url": ""},
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="https://example.com/events/placeholder",
        parent_url="https://realvenue.example/calendar",
        source="Venue C",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "https://realvenue.example/calendar"


def test_enforce_event_url_values_replaces_unreachable_formulated_child_url(monkeypatch) -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    checked_urls: list[str] = []

    def fake_reachable(url: str) -> bool:
        checked_urls.append(url)
        return False

    monkeypatch.setattr(handler, "_event_url_is_reachable", fake_reachable)
    df = pd.DataFrame(
        [
            {
                "event_name": "Rhythm Train",
                "source": "Live Entertainment Victoria",
                "url": "https://liveentertainmentvictoria.com/event/rhythm-train/",
            },
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="https://liveentertainmentvictoria.com/events/",
        parent_url="",
        source="Live Entertainment Victoria",
    )

    assert checked_urls == ["https://liveentertainmentvictoria.com/event/rhythm-train/"]
    assert len(out) == 1
    assert out.loc[0, "url"] == "https://liveentertainmentvictoria.com/events/"


def test_enforce_event_url_values_keeps_reachable_formulated_child_url(monkeypatch) -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)

    monkeypatch.setattr(handler, "_event_url_is_reachable", lambda _url: True)
    df = pd.DataFrame(
        [
            {
                "event_name": "Weekly Salsa Social",
                "source": "The Loft Pub",
                "url": "https://theloftpub.com/event/weekly-salsa-social/",
            },
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="https://theloftpub.com/events/",
        parent_url="",
        source="The Loft Pub",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "https://theloftpub.com/event/weekly-salsa-social/"


def test_enforce_event_url_values_does_not_probe_row_url_without_fetched_source_fallback(monkeypatch) -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)

    def fail_if_called(_url: str) -> bool:
        raise AssertionError("reachability should not be checked without a fetched source fallback")

    monkeypatch.setattr(handler, "_event_url_is_reachable", fail_if_called)
    df = pd.DataFrame(
        [
            {
                "event_name": "Imported URL Event",
                "source": "Venue D",
                "url": "https://venue.example/event/imported-url-event/",
            },
        ]
    )

    out = handler._enforce_event_url_values(
        df,
        default_url="",
        parent_url="https://example.com/blocked-parent",
        source="Venue D",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "https://venue.example/event/imported-url-event/"
