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
        default_url="https://example.com/events/friday-social",
        parent_url="",
        source="Some Venue",
    )

    assert len(out) == 1
    assert out.loc[0, "url"] == "https://example.com/events/friday-social"


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
