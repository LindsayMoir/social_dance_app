from __future__ import annotations

import os
import pandas as pd
import sys

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
