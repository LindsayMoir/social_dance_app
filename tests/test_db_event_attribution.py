import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler, ensure_chatbot_metrics_schema


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


def test_ensure_source_distribution_history_tables_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_source_distribution_history_tables()

    joined = "\n".join(queries)
    assert "CREATE TABLE IF NOT EXISTS source_event_counts_history" in joined
    assert "CREATE TABLE IF NOT EXISTS source_distribution_alerts_history" in joined
    assert "idx_source_event_counts_history_run_id" in joined
    assert "idx_source_distribution_alerts_history_run_id" in joined


def test_ensure_chatbot_metrics_schema_executes_expected_queries() -> None:
    executed: list[str] = []

    class _FakeConn:
        def execute(self, stmt):
            executed.append(str(stmt))

    class _FakeBegin:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeEngine:
        def begin(self):
            return _FakeBegin()

    ensure_chatbot_metrics_schema(_FakeEngine())

    joined = "\n".join(executed)
    assert "CREATE TABLE IF NOT EXISTS chatbot_request_metrics" in joined
    assert "CREATE TABLE IF NOT EXISTS chatbot_stage_metrics" in joined
    assert "uq_chatbot_stage_metrics_nk" in joined


def test_ensure_core_event_tables_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_core_event_tables()

    joined = "\n".join(queries)
    assert "CREATE TABLE IF NOT EXISTS events_history" in joined
    assert "ALTER TABLE events_history ADD COLUMN IF NOT EXISTS original_event_id INTEGER" in joined
    assert "CREATE TABLE IF NOT EXISTS events (" in joined
    assert "CREATE TABLE IF NOT EXISTS address (" in joined


def test_ensure_core_application_tables_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_core_application_tables()

    joined = "\n".join(queries)
    assert "CREATE TABLE IF NOT EXISTS urls (" in joined
    assert "CREATE TABLE IF NOT EXISTS runs (" in joined
    assert "CREATE TABLE IF NOT EXISTS events_history" in joined
    assert "CREATE TABLE IF NOT EXISTS events (" in joined
    assert "CREATE TABLE IF NOT EXISTS address (" in joined


def test_ensure_address_sequence_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_address_sequence(662)

    joined = "\n".join(queries)
    assert "CREATE SEQUENCE IF NOT EXISTS address_address_id_seq" in joined
    assert "START WITH 662" in joined
    assert "ALTER COLUMN address_id SET DEFAULT nextval('address_address_id_seq')" in joined
    assert "ALTER SEQUENCE address_address_id_seq OWNED BY address.address_id" in joined


def test_execute_query_applies_statement_timeout_before_query() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    executed: list[tuple[str, object]] = []

    class _FakeResult:
        returns_rows = True

        def fetchall(self):
            return [("Salsa Caliente", 20)]

    class _FakeConnection:
        def execute(self, stmt, params=None):
            executed.append((str(stmt), params))
            return _FakeResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeEngine:
        def connect(self):
            return _FakeConnection()

    handler.conn = _FakeEngine()

    result = handler.execute_query(
        "SELECT source, COUNT(*) AS counted FROM events GROUP BY source",
        statement_timeout_ms=15000,
    )

    assert result == [("Salsa Caliente", 20)]
    assert len(executed) == 2
    assert "set_config('statement_timeout'" in executed[0][0]
    assert executed[0][1] == {"timeout_value": "15000ms"}
    assert "SELECT source, COUNT(*) AS counted FROM events GROUP BY source" in executed[1][0]


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


def test_write_events_to_db_aligns_recurring_weekday_dates_before_insert(monkeypatch) -> None:
    handler = _make_handler(old_days=30)
    handler._rename_google_calendar_columns = lambda df: df
    handler._keywords_to_specific_dance_styles = lambda _keywords: ""
    handler._resolve_event_source_label = lambda source, url, parent_url: source or "Deb Rhymer"
    handler._enforce_event_source_values = lambda df, _source: df
    handler._enforce_event_url_values = lambda df, default_url, parent_url, source: df
    handler._apply_event_overrides = lambda df, url, parent_url: df
    handler._convert_datetime_fields = DatabaseHandler._convert_datetime_fields.__get__(handler, DatabaseHandler)
    handler._clean_day_of_week_field = DatabaseHandler._clean_day_of_week_field.__get__(handler, DatabaseHandler)
    handler._enforce_live_music_dance_style_policy = lambda df: df
    handler.clean_up_address_basic = lambda df: df
    handler.process_event_address = lambda event_dict: event_dict
    handler._filter_events = lambda df, apply_date_filter=False: df
    handler._sanitize_events_dataframe_for_insert = lambda df: df
    handler.write_url_to_db = lambda _row: None
    handler._write_event_write_attribution_rows = lambda **_kwargs: None

    inserted_batches: list[pd.DataFrame] = []

    def _capture_insert(df: pd.DataFrame) -> list[int]:
        inserted_batches.append(df.copy())
        return [501, 502]

    handler._insert_events_and_return_ids = _capture_insert

    monkeypatch.setenv("DS_RUN_ID", "run-1")
    monkeypatch.setenv("DS_STEP_NAME", "rd_ext")

    df = pd.DataFrame(
        {
            "event_name": ["Sunday Blues Services", "Sunday Blues Services"],
            "dance_style": ["swing", "swing"],
            "description": [
                "Every Sunday blues jam with live music.",
                "Every Sunday blues jam with live music.",
            ],
            "day_of_week": ["Sunday", "Sunday"],
            "start_date": ["2026-03-30", "2026-04-05"],
            "end_date": ["2026-03-30", "2026-04-05"],
            "start_time": ["15:00", "15:00"],
            "end_time": ["19:30", "19:30"],
            "source": ["Deb Rhymer", "Deb Rhymer"],
            "location": ["Studio 919 (Strathcona Hotel)", "Studio 919 (Strathcona Hotel)"],
            "price": ["$5", "$5"],
            "url": ["https://www.debrhymerband.com/shows", "https://www.debrhymerband.com/shows"],
            "event_type": ["social dance, live music", "social dance, live music"],
            "address_id": [None, None],
        }
    )

    written_count = handler.write_events_to_db(
        df,
        "https://www.debrhymerband.com/shows",
        "",
        "Deb Rhymer",
        ["swing"],
        provider="openrouter",
        model="deepseek",
        prompt_type="event_extraction",
        decision_reason="llm_success",
    )

    assert written_count == 2
    assert inserted_batches
    inserted_df = inserted_batches[0]
    assert inserted_df.iloc[0]["start_date"].isoformat() == "2026-03-29"
    assert pd.to_datetime(inserted_df.iloc[0]["end_date"]).date().isoformat() == "2026-03-29"
    assert inserted_df.iloc[1]["start_date"].isoformat() == "2026-04-05"

    monkeypatch.delenv("DS_RUN_ID", raising=False)
    monkeypatch.delenv("DS_STEP_NAME", raising=False)


def test_build_phase1_telemetry_integrity_report_treats_step_mismatch_as_advisory() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    query_results = {
        "SUM(COALESCE(events_written, 0)) AS metrics_events_written_total": [
            ("scraper", 5, 2),
            ("fb", 3, 1),
        ],
        "COUNT(*) AS write_attribution_count": [
            ("scraper", 5, 5),
            ("fb", 2, 2),
        ],
        "COUNT(*) AS delete_attribution_count": [
            ("scraper", 1, 1),
        ],
        "unknown_delete_reason_total": [
            (7, 7, 1, 1, 1),
        ],
    }

    def _exec(query: str, params=None):
        normalized = " ".join(str(query).split())
        for key, value in query_results.items():
            if key in normalized:
                return value
        raise AssertionError(normalized)

    handler.execute_query = _exec  # type: ignore[attr-defined]

    report = handler.build_phase1_telemetry_integrity_report("run-123")

    assert report["available"] is True
    assert report["status"] == "FAIL"
    assert "step_mismatch:fb:metrics_events=3:write_attribution=2" in report["advisories"]
    assert "unknown_delete_reason_total:1" in report["violations"]
    assert report["steps"]["scraper"]["status"] == "FAIL"
    assert report["steps"]["fb"]["status"] == "WARN"
    assert report["summary"]["write_attribution_rows"] == 7
    assert report["summary"]["step_mismatch_advisory_count"] == 1


def test_build_phase1_telemetry_integrity_report_prefers_handled_by_over_pipeline_step() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    query_results = {
        "SUM(COALESCE(events_written, 0)) AS metrics_events_written_total": [
            ("images", 10, 3),
            ("rd_ext", 4, 2),
        ],
        "COUNT(*) AS write_attribution_count": [
            ("images", 10, 10),
            ("rd_ext", 4, 4),
        ],
        "COUNT(*) AS delete_attribution_count": [],
        "unknown_delete_reason_total": [
            (14, 14, 0, 0, 0),
        ],
    }

    def _exec(query: str, params=None):
        normalized = " ".join(str(query).split())
        if "REPLACE(COALESCE(handled_by, ''), '.py', '')" in normalized:
            return query_results["SUM(COALESCE(events_written, 0)) AS metrics_events_written_total"]
        for key, value in query_results.items():
            if key in normalized:
                return value
        raise AssertionError(normalized)

    handler.execute_query = _exec  # type: ignore[attr-defined]

    report = handler.build_phase1_telemetry_integrity_report("run-456")

    assert report["status"] == "PASS"
    assert report["violations"] == []
    assert report["steps"]["images"]["metrics_events_written_total"] == 10
    assert report["steps"]["images"]["write_attribution_count"] == 10
    assert report["steps"]["rd_ext"]["metrics_events_written_total"] == 4
    assert report["steps"]["rd_ext"]["write_attribution_count"] == 4
