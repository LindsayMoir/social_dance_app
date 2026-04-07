from __future__ import annotations

import os
import pandas as pd
import sys
from pathlib import Path

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from irrelevant_rows import IrrelevantRowsHandler


class _FakeLlmHandler:
    def __init__(self, parsed_result):
        self.parsed_result = parsed_result
        self.calls: list[tuple[str, str, str | None]] = []

    def query_llm(self, url, prompt, schema_type=None):  # type: ignore[no-untyped-def]
        self.calls.append((url, prompt, schema_type))
        return '[{"event_id": 1, "Label": 0, "event_type_new": "social dance"}]'

    def extract_and_parse_json(self, response, url, schema_type=None):  # type: ignore[no-untyped-def]
        self.calls.append((response, url, schema_type))
        return self.parsed_result


def _build_handler(parsed_result) -> IrrelevantRowsHandler:
    handler = IrrelevantRowsHandler.__new__(IrrelevantRowsHandler)
    handler.llm_handler = _FakeLlmHandler(parsed_result)
    handler._RELEVANCE_SCHEMA_TYPE = "relevance_classification"
    handler._RELEVANCE_REQUIRED_COLUMNS = {"event_id", "Label", "event_type_new"}
    handler.load_prompt = lambda chunk: f"prompt:{chunk}"  # type: ignore[method-assign]
    return handler


def test_parse_llm_response_uses_shared_relevance_parser() -> None:
    handler = _build_handler(
        [
            {"event_id": 10, "Label": 0, "event_type_new": "social dance"},
            {"event_id": 11, "Label": 1, "event_type_new": "other"},
        ]
    )

    df = handler.parse_llm_response("ignored")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) >= {"event_id", "Label", "event_type_new"}


def test_process_chunk_with_llm_requests_relevance_schema() -> None:
    handler = _build_handler([{"event_id": 10, "Label": 0, "event_type_new": "social dance"}])

    result = handler.process_chunk_with_llm(
        pd.DataFrame([{"event_id": 10, "event_name": "Test Event"}]),
        chunk_index=0,
    )

    assert isinstance(result, pd.DataFrame)
    query_call = handler.llm_handler.calls[0]
    assert query_call[2] == "relevance_classification"


def test_parse_llm_response_falls_back_to_deterministic_row_parsing() -> None:
    handler = _build_handler([])

    response = """
    Here are the rows:
    {"event_id": 21, "label": 0, "event_type": "social dance"}
    {"event_id": 22, "Label": 1, "event_type_new": "other"}
    """

    df = handler.parse_llm_response(response)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.to_dict(orient="records") == [
        {"event_id": 21, "Label": 0, "event_type_new": "social dance"},
        {"event_id": 22, "Label": 1, "event_type_new": "other"},
    ]


def test_parse_llm_response_captures_raw_response_when_parsing_fails(
    monkeypatch, tmp_path: Path
) -> None:
    handler = _build_handler([])
    monkeypatch.chdir(tmp_path)

    df = handler.parse_llm_response("not valid json and no recoverable rows")

    assert isinstance(df, pd.DataFrame)
    assert df.empty
    artifacts = list((tmp_path / "output" / "codex_review").glob("irrelevant_rows_bad_response_*.txt"))
    assert len(artifacts) == 1
    contents = artifacts[0].read_text(encoding="utf-8")
    assert "reason=no_valid_rows" in contents
    assert "not valid json and no recoverable rows" in contents
