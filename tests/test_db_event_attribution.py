import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler


def _make_handler(old_days: int = 30) -> DatabaseHandler:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.config = {"clean_up": {"old_events": old_days}}
    return handler


def test_ensure_event_attribution_tables_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_event_attribution_tables()

    joined = "\n".join(queries)
    assert "CREATE TABLE IF NOT EXISTS event_write_attribution" in joined
    assert "CREATE TABLE IF NOT EXISTS event_delete_attribution" in joined
    assert "idx_event_write_attribution_run_id" in joined
    assert "idx_event_delete_attribution_reason_code" in joined


def test_normalize_delete_reason_code_handles_dynamic_old_event_reason() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)

    normalized = handler._normalize_delete_reason_code("end_date_older_than_30_days")

    assert normalized["delete_reason_code"] == "end_date_older_than_days"
    assert normalized["raw_delete_reason"] == "end_date_older_than_30_days"
    assert normalized["reason_registered"] is True


def test_write_events_to_db_records_canonical_write_attribution(monkeypatch) -> None:
    handler = _make_handler(old_days=30)
    handler._rename_google_calendar_columns = lambda df: df
    handler._keywords_to_specific_dance_styles = lambda _keywords: ""
    handler._resolve_event_source_label = lambda source, url, parent_url: source or "example"
    handler._enforce_event_source_values = lambda df, _source: df
    handler._enforce_event_url_values = lambda df, default_url, parent_url, source: df
    handler._apply_event_overrides = lambda df, url, parent_url: df
    handler._convert_datetime_fields = DatabaseHandler._convert_datetime_fields.__get__(handler, DatabaseHandler)
    handler._clean_day_of_week_field = lambda df: df
    handler._enforce_live_music_dance_style_policy = lambda df: df
    handler.clean_up_address_basic = lambda df: df
    handler.process_event_address = lambda event_dict: event_dict
    handler._filter_events = lambda df, apply_date_filter=False: df
    handler._sanitize_events_dataframe_for_insert = lambda df: df
    handler._insert_events_and_return_ids = lambda df: [101, 102]
    logged_rows: list[list[object]] = []
    handler.write_url_to_db = lambda row: logged_rows.append(row)
    attribution_calls: list[dict] = []
    handler._write_event_write_attribution_rows = lambda **kwargs: attribution_calls.append(kwargs)

    monkeypatch.setenv("DS_RUN_ID", "run-1")
    monkeypatch.setenv("DS_STEP_NAME", "scraper")

    today = datetime.now().date()
    df = pd.DataFrame(
        {
            "event_name": ["One", "Two"],
            "dance_style": ["salsa", "bachata"],
            "description": ["desc 1", "desc 2"],
            "day_of_week": ["Friday", "Saturday"],
            "start_date": [today + timedelta(days=5), today + timedelta(days=6)],
            "end_date": [today + timedelta(days=5), today + timedelta(days=6)],
            "start_time": ["19:00", "20:00"],
            "end_time": ["22:00", "23:00"],
            "source": ["Example Source", "Example Source"],
            "location": ["Venue A", "Venue B"],
            "price": ["", ""],
            "url": ["https://example.com/a", "https://example.com/b"],
            "event_type": ["social dance", "social dance"],
            "address_id": [None, None],
        }
    )

    written_count = handler.write_events_to_db(
        df,
        "https://example.com/events",
        "https://example.com",
        "Example Source",
        ["dance"],
        provider="openrouter",
        model="deepseek",
        prompt_type="event_extraction",
        decision_reason="llm_success",
    )

    assert written_count == 2
    assert logged_rows
    assert attribution_calls
    attribution = attribution_calls[0]
    assert attribution["event_ids"] == [101, 102]
    assert attribution["provider"] == "openrouter"
    assert attribution["model"] == "deepseek"
    assert attribution["decision_reason"] == "llm_success"
    assert attribution["url"] == "https://example.com/events"

    monkeypatch.delenv("DS_RUN_ID", raising=False)
    monkeypatch.delenv("DS_STEP_NAME", raising=False)
