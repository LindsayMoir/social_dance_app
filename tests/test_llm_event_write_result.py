from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm import EventWriteResult, LLMHandler


def test_event_write_result_bool_tracks_success() -> None:
    assert bool(EventWriteResult(success=True, events_written=3, decision_reason="ok")) is True
    assert bool(EventWriteResult(success=False, events_written=0, decision_reason="miss")) is False


def test_process_llm_response_returns_written_event_count(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"ignored","description":"x"*120}]}'
            .replace('"x"*120', "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {"event_name": "One", "start_date": "2026-03-23"},
            {"event_name": "Two", "start_date": "2026-03-24"},
        ],
    )

    result = handler.process_llm_response(
        "https://example.com/event",
        "",
        "dance event text",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=2, decision_reason="llm_success")
    assert len(writes) == 1
    assert len(writes[0]) == 2


def test_process_llm_response_too_short_returns_zero_events(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    handler.db_handler = SimpleNamespace(write_events_to_db=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            "short",
            {"provider": "test", "model": "fake"},
        ),
    )

    result = handler.process_llm_response(
        "https://example.com/event",
        "",
        "dance event text",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=False, events_written=0, decision_reason="too_short")


def test_process_llm_response_normalizes_relative_day_of_week_before_write(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"One","start_date":"2026-03-24","day_of_week":"Tomorrow","description":"x"*120}]}'
            .replace('"x"*120', "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {"event_name": "One", "start_date": "2026-03-24", "day_of_week": "Tomorrow"},
        ],
    )

    result = handler.process_llm_response(
        "https://example.com/event",
        "",
        "dance event text",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=1, decision_reason="llm_success")
    assert len(writes) == 1
    assert list(writes[0]["day_of_week"]) == ["Tuesday"]


def test_process_llm_response_drops_rows_still_missing_start_date(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"One","day_of_week":"Tomorrow","description":"x"*120}]}'
            .replace('"x"*120', "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"),
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {"event_name": "One", "day_of_week": "Tomorrow"},
        ],
    )

    result = handler.process_llm_response(
        "https://example.com/event",
        "",
        "dance event text",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=False, events_written=0, decision_reason="invalid_or_missing_start_date")
    assert writes == []


def test_process_llm_response_accepts_compact_json_event_payload(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"Loft Show","start_date":"2026-03-11"}]}',
            {"provider": "test", "model": "fake"},
        ),
    )

    result = handler.process_llm_response(
        "https://loftpubvictoria.com/events/2026-03-11/",
        "",
        "Loft event text",
        "Loft",
        ["music"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=1, decision_reason="llm_success")
    assert len(writes) == 1
    assert writes[0].iloc[0]["event_name"] == "Loft Show"


def test_process_llm_response_accepts_compact_empty_events_payload(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    handler.db_handler = SimpleNamespace(write_events_to_db=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[]}',
            {"provider": "test", "model": "fake"},
        ),
    )

    result = handler.process_llm_response(
        "https://loftpubvictoria.com/events/2026-03-12/",
        "",
        "Loft event text",
        "Loft",
        ["music"],
        "default",
    )

    assert result == EventWriteResult(success=False, events_written=0, decision_reason="no_events")


def test_process_llm_response_fills_missing_image_date_from_detected_hint(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"Poster Event","description":"social dance night with enough detail to exceed the minimum response threshold for parsing validation."}]}',
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {"event_name": "Poster Event", "description": "social dance night"},
        ],
    )

    result = handler.process_llm_response(
        "https://www.instagram.com/p/test/#image=abc123",
        "https://www.instagram.com/p/test/",
        "Detected_Date: 2026-03-24\nDetected_Day: Tuesday\nposter text",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=1, decision_reason="llm_success")
    assert len(writes) == 1
    assert writes[0].iloc[0]["start_date"] == "2026-03-24"
    assert writes[0].iloc[0]["day_of_week"] == "Tuesday"


def test_process_llm_response_drops_unresolved_image_date_conflict(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"Tuesday Night WCS Dance","start_date":"2026-03-31","day_of_week":"Tuesday","description":"dance social with enough response length to pass the short-response guard."}]}',
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {"event_name": "Tuesday Night WCS Dance", "start_date": "2026-03-31", "day_of_week": "Tuesday", "description": "dance social with enough response length to pass the short-response guard."},
        ],
    )

    result = handler.process_llm_response(
        "https://www.instagram.com/p/test/#image=def456",
        "https://www.instagram.com/p/test/",
        "Detected_Date: 2026-03-24\nDetected_Day: Tuesday\nTuesday 24 March 2026 from 19:30-21:45",
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=False, events_written=0, decision_reason="image_date_conflict")
    assert writes == []


def test_process_llm_response_expands_schedule_poster_dates(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"March Social Series","description":"monthly social series with enough descriptive detail to exceed the minimum response threshold for parsing validation."}]}',
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {
                "event_name": "March Social Series",
                "description": "monthly social series with enough descriptive detail to exceed the minimum response threshold for parsing validation.",
            },
        ],
    )

    result = handler.process_llm_response(
        "https://www.instagram.com/p/test/#image=schedule123",
        "https://www.instagram.com/p/test/",
        (
            "Detected_Poster_Type: schedule_multi_event\n"
            "Detected_Schedule_Dates: 2026-03-01, 2026-03-08, 2026-03-15\n"
            "Detected_Date_Analysis: multiple_textual_date_candidates\n"
            "March 1 Salsa Night\nMarch 8 Bachata Social\nMarch 15 Kizomba Night"
        ),
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=3, decision_reason="llm_success")
    assert len(writes) == 1
    assert list(writes[0]["start_date"]) == ["2026-03-01", "2026-03-08", "2026-03-15"]


def test_process_llm_response_keeps_schedule_poster_rows_without_single_date_conflict(monkeypatch) -> None:
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {"crawling": {"prompt_max_length": 5000}}
    writes: list[pd.DataFrame] = []
    handler.db_handler = SimpleNamespace(
        write_events_to_db=lambda df, *_args, **_kwargs: writes.append(df.copy())
    )

    monkeypatch.setattr(
        handler,
        "generate_prompt",
        lambda url, extracted_text, prompt_type: ("prompt text", "event_extraction"),
    )
    monkeypatch.setattr(
        handler,
        "query_llm",
        lambda url, prompt_attempt, schema_type, return_metadata=True: (
            '{"events":[{"event_name":"March Social Series","start_date":"2026-03-08","day_of_week":"Sunday","description":"monthly social series with enough descriptive detail to exceed the minimum response threshold for parsing validation."}]}',
            {"provider": "test", "model": "fake"},
        ),
    )
    monkeypatch.setattr(
        handler,
        "extract_and_parse_json",
        lambda llm_response, url, schema_type: [
            {
                "event_name": "March Social Series",
                "start_date": "2026-03-08",
                "day_of_week": "Sunday",
                "description": "monthly social series with enough descriptive detail to exceed the minimum response threshold for parsing validation.",
            },
        ],
    )

    result = handler.process_llm_response(
        "https://www.instagram.com/p/test/#image=schedule123",
        "https://www.instagram.com/p/test/",
        (
            "Detected_Poster_Type: schedule_multi_event\n"
            "Detected_Date: 2026-03-01\n"
            "Detected_Day: Sunday\n"
            "Detected_Schedule_Dates: 2026-03-01, 2026-03-08, 2026-03-15\n"
            "Detected_Date_Analysis: multiple_textual_date_candidates\n"
            "March 1 Salsa Night\nMarch 8 Bachata Social\nMarch 15 Kizomba Night"
        ),
        "Example Source",
        ["dance"],
        "default",
    )

    assert result == EventWriteResult(success=True, events_written=3, decision_reason="llm_success")
    assert len(writes) == 1
    assert list(writes[0]["start_date"]) == ["2026-03-01", "2026-03-08", "2026-03-15"]
