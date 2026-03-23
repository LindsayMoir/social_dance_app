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
        write_events_to_db=lambda df, *_args: writes.append(df.copy())
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
            "{}",
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
