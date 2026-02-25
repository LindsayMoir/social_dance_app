import sys

import pandas as pd
import pytest

pytest.importorskip("sentence_transformers")

sys.path.insert(0, "src")
from dedup_llm import DeduplicationHandler


def test_source_matches_rule_equals() -> None:
    assert DeduplicationHandler._source_matches_rule(
        "Victoria Latin Dance Association",
        {"source_equals": "Victoria Latin Dance Association"},
    )
    assert not DeduplicationHandler._source_matches_rule(
        "Some Other Source",
        {"source_equals": "Victoria Latin Dance Association"},
    )


def test_score_event_row_applies_config_penalty() -> None:
    handler = DeduplicationHandler.__new__(DeduplicationHandler)
    handler.source_score_penalties = [
        {
            "match": {"source_equals": "Victoria Latin Dance Association"},
            "penalty": 100,
        }
    ]

    address_df = pd.DataFrame(
        [{"address_id": 1, "postal_code": "V8W 2C2"}]
    )
    row = pd.Series(
        {
            "location": "Some Venue",
            "description": "A reasonably descriptive event.",
            "event_name": "Sample Event",
            "address_id": 1,
            "source": "Victoria Latin Dance Association",
            "dance_style": "salsa",
            "event_type": "social dance",
            "start_time": "20:00:00",
            "url": "https://example.com",
        }
    )

    score = handler.score_event_row(row, address_df)
    assert score < 0
