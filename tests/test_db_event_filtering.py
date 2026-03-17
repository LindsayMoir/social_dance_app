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


def test_write_events_to_db_logs_old_facebook_event_detail_rejection_reason(monkeypatch):
    handler = _make_handler(old_days=30)
    handler._rename_google_calendar_columns = lambda df: df
    handler._keywords_to_specific_dance_styles = lambda _keywords: ""
    handler._resolve_event_source_label = lambda source, url, parent_url: source or "fb"
    handler._enforce_event_source_values = lambda df, _source: df
    handler._enforce_event_url_values = lambda df, default_url, parent_url, source: df
    handler._apply_event_overrides = lambda df, url, parent_url: df
    handler._convert_datetime_fields = DatabaseHandler._convert_datetime_fields.__get__(handler, DatabaseHandler)
    handler._clean_day_of_week_field = lambda df: df
    handler._enforce_live_music_dance_style_policy = lambda df: df
    handler.clean_up_address_basic = lambda df: df
    logged_rows = []
    handler.write_url_to_db = lambda row: logged_rows.append(row)

    df = pd.DataFrame(
        {
            "event_name": ["Past Facebook Event"],
            "event_type": ["social dance"],
            "dance_style": ["salsa"],
            "start_date": [datetime.now().date() - timedelta(days=120)],
            "end_date": [datetime.now().date() - timedelta(days=120)],
            "start_time": ["19:00"],
            "end_time": ["22:00"],
            "location": ["Venue"],
            "description": ["Old event"],
            "url": ["https://www.facebook.com/events/1234567890123456/"],
            "source": ["fb"],
        }
    )

    handler.write_events_to_db(
        df,
        "https://www.facebook.com/events/1234567890123456/",
        "",
        "fb",
        ["salsa"],
    )

    assert logged_rows
    assert len(logged_rows[0]) >= 8
    assert logged_rows[0][7] == "rejected_old_facebook_event_detail"
