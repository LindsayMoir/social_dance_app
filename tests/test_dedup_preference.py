from __future__ import annotations

from datetime import datetime
import sys

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler
from dedup_llm import DeduplicationHandler


def test_source_matches_event_url_handles_latindancecanada() -> None:
    assert DatabaseHandler._source_matches_event_url(
        "Latin Dance Canada",
        "https://latindancecanada.com/",
    ) is True
    assert DatabaseHandler._source_matches_event_url(
        "Latin Dance Canada",
        "https://vlda.ca/resources/",
    ) is False


def test_decide_preferred_row_prefers_source_matching_own_url() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    older = datetime(2026, 3, 20, 10, 0, 0)
    newer = datetime(2026, 3, 21, 10, 0, 0)
    row_matching = pd.Series(
        {
            "event_id": 10,
            "source": "Latin Dance Canada",
            "url": "https://latindancecanada.com/",
            "time_stamp": older,
            "event_name": "Friday Social",
            "description": "",
            "location": "Some Venue",
        }
    )
    row_non_matching = pd.Series(
        {
            "event_id": 11,
            "source": "Latin Dance Canada",
            "url": "https://example.com/latin-night",
            "time_stamp": newer,
            "event_name": "Friday Social",
            "description": "More complete row",
            "location": "Some Venue",
        }
    )

    preferred, other = handler.decide_preferred_row(row_matching, row_non_matching)

    assert int(preferred["event_id"]) == 10
    assert int(other["event_id"]) == 11


def test_score_event_row_prefers_source_matching_own_url() -> None:
    handler = DeduplicationHandler.__new__(DeduplicationHandler)
    handler.source_score_penalties = []
    handler._is_low_quality_location_text = lambda _value: False

    address_df = pd.DataFrame([{"address_id": 1, "postal_code": "V8W 2C2"}])
    matching_row = pd.Series(
        {
            "location": "Some Venue",
            "description": "A reasonably descriptive event.",
            "event_name": "Sample Event",
            "address_id": 1,
            "source": "Latin Dance Canada",
            "dance_style": "salsa",
            "event_type": "social dance",
            "start_time": "20:00:00",
            "url": "https://latindancecanada.com/",
        }
    )
    non_matching_row = matching_row.copy()
    non_matching_row["url"] = "https://example.com/event"

    matching_score = handler.score_event_row(matching_row, address_df)
    non_matching_score = handler.score_event_row(non_matching_row, address_df)

    assert matching_score > non_matching_score
