"""Regression tests for binary labels returned by the deduplication LLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dedup_llm import DeduplicationHandler, InvalidDeduplicationResponseError


def _handler() -> DeduplicationHandler:
    return DeduplicationHandler.__new__(DeduplicationHandler)


def test_normalize_dedup_labels_maps_negative_one_to_keep() -> None:
    response_df = pd.DataFrame(
        [
            {"group_id": 1, "event_id": 10, "Label": 0},
            {"group_id": 1, "event_id": 11, "Label": -1},
            {"group_id": 1, "event_id": 12, "Label": 1},
        ]
    )

    normalized = _handler()._normalize_dedup_labels(response_df, chunk_index=0)

    assert normalized["Label"].tolist() == [0, 0, 1]


def test_normalize_dedup_labels_rejects_other_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _handler()
    monkeypatch.setattr(handler, "_write_dedup_parse_artifact", lambda **_kwargs: None)
    response_df = pd.DataFrame([{"group_id": 1, "event_id": 10, "Label": 2}])

    with pytest.raises(InvalidDeduplicationResponseError, match="allowed values 0 and 1"):
        handler._normalize_dedup_labels(response_df, chunk_index=0)


def test_parse_llm_response_rejects_invalid_label_before_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _handler()

    class _StubLlmHandler:
        @staticmethod
        def extract_and_parse_json(*_args: object, **_kwargs: object) -> list[dict[str, int]]:
            return [{"group_id": 1, "event_id": 10, "Label": 2}]

    handler.llm_handler = _StubLlmHandler()
    monkeypatch.setattr(handler, "_write_dedup_parse_artifact", lambda **_kwargs: None)

    with pytest.raises(InvalidDeduplicationResponseError):
        handler.parse_llm_response("ignored", chunk_index=0)
