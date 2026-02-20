import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler


def _make_handler(old_days: int = 30) -> DatabaseHandler:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.config = {"clean_up": {"old_events": old_days}}
    return handler


def test_drop_old_events_by_date_removes_old_rows():
    handler = _make_handler(old_days=30)
    now = datetime.now().date()
    df = pd.DataFrame(
        {
            "end_date": [now - timedelta(days=120), now - timedelta(days=2)],
            "start_date": [now - timedelta(days=120), now - timedelta(days=2)],
            "start_time": ["19:00", "20:00"],
            "end_time": ["21:00", "22:00"],
            "location": ["A", "B"],
            "description": ["old", "new"],
        }
    )

    out = handler._drop_old_events_by_date(df, context="test")
    assert len(out) == 1
    assert out.iloc[0]["description"] == "new"


def test_filter_events_can_skip_date_filter_when_requested():
    handler = _make_handler(old_days=30)
    now = datetime.now().date()
    df = pd.DataFrame(
        {
            "end_date": [now - timedelta(days=120), now - timedelta(days=2)],
            "start_date": [now - timedelta(days=120), now - timedelta(days=2)],
            "start_time": ["19:00", "20:00"],
            "end_time": ["21:00", "22:00"],
            "location": ["A", "B"],
            "description": ["old", "new"],
        }
    )

    out = handler._filter_events(df, apply_date_filter=False)
    assert len(out) == 2
