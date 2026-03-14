import sys

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler


def test_resolve_event_source_label_keeps_explicit_non_placeholder() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    out = handler._resolve_event_source_label(
        source="Victoria Latin Dance Association",
        url="https://example.com/events",
        parent_url="",
    )
    assert out == "Victoria Latin Dance Association"


def test_resolve_event_source_label_falls_back_from_placeholder_to_url_host() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    out = handler._resolve_event_source_label(
        source="extracted text",
        url="https://thedukesaloon.com/events/",
        parent_url="",
    )
    assert out == "thedukesaloon"


def test_enforce_event_source_values_replaces_extracted_text_rows() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "A", "source": "extracted text"},
            {"event_name": "B", "source": "Eventbrite"},
            {"event_name": "C", "source": ""},
        ]
    )

    out = handler._enforce_event_source_values(df, fallback_source="The Coda")
    assert out.loc[0, "source"] == "The Coda"
    assert out.loc[1, "source"] == "Eventbrite"
    assert out.loc[2, "source"] == "The Coda"
