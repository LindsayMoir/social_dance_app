import sys

import pandas as pd

sys.path.insert(0, 'src')
from db import DatabaseHandler


def _build_handler_for_unit_test() -> DatabaseHandler:
    """Construct a minimal handler instance without running DB initialization."""
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.config = {}
    handler.event_overrides = []
    return handler


def test_url_matches_rule_contains() -> None:
    assert DatabaseHandler._url_matches_rule(
        "https://www.debrhymerband.com/shows",
        {"url_contains": "debrhymerband.com/shows"},
    )
    assert not DatabaseHandler._url_matches_rule(
        "https://example.com/other",
        {"url_contains": "debrhymerband.com/shows"},
    )


def test_apply_event_overrides_sets_event_type() -> None:
    handler = _build_handler_for_unit_test()
    handler.event_overrides = [
        {
            "name": "debrhymer_shows_event_type",
            "match": {"url_contains": "debrhymerband.com/shows"},
            "set": {"event_type": "social dance, live music"},
        }
    ]

    df = pd.DataFrame([{"event_name": "Sunday Blues Services", "event_type": "live music"}])
    out = handler._apply_event_overrides(
        df=df,
        url="https://www.debrhymerband.com/shows",
        parent_url="",
    )

    assert len(out) == 1
    assert out.iloc[0]["event_type"] == "social dance, live music"
