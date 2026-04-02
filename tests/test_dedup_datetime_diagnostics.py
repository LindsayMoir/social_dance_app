from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import dedup_llm
from dedup_llm import DeduplicationHandler


def _build_handler() -> DeduplicationHandler:
    handler = DeduplicationHandler.__new__(DeduplicationHandler)
    handler.engine = object()
    handler.db_handler = object()
    handler.get_git_version = lambda: "abc123"  # type: ignore[method-assign]
    return handler


def test_write_unparseable_datetime_artifact_writes_expected_columns(tmp_path: Path) -> None:
    handler = _build_handler()
    original_duplicates_path = dedup_llm.duplicates_path
    dedup_llm.duplicates_path = lambda filename: str(tmp_path / filename)
    try:
        handler._write_unparseable_datetime_artifact(
            pd.DataFrame(
                [
                    {
                        "event_id": 1,
                        "event_name": "Bad Time Event",
                        "start_date": "2026-04-01",
                        "start_time": "not-a-time",
                        "source": "Instagram",
                        "location": "Somewhere",
                        "url": "https://example.com/1",
                    }
                ]
            )
        )
    finally:
        dedup_llm.duplicates_path = original_duplicates_path

    artifact = tmp_path / "unparseable_datetimes.csv"
    assert artifact.exists()
    artifact_df = pd.read_csv(artifact)
    assert list(artifact_df.columns) == [
        "event_id",
        "event_name",
        "start_date",
        "start_time",
        "source",
        "location",
        "url",
    ]
    assert artifact_df.iloc[0]["start_time"] == "not-a-time"


def test_deduplicate_with_embeddings_persists_invalid_datetime_rows(tmp_path: Path, monkeypatch) -> None:
    handler = _build_handler()
    original_duplicates_path = dedup_llm.duplicates_path
    dedup_llm.duplicates_path = lambda filename: str(tmp_path / filename)

    events_df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_name": "Bad Time Event",
                "dance_style": "salsa",
                "description": "desc",
                "day_of_week": "Friday",
                "start_date": "2026-04-01",
                "end_date": "2026-04-01",
                "start_time": "not-a-time",
                "end_time": "22:00:00",
                "source": "Instagram",
                "location": "Somewhere",
                "price": "",
                "url": "https://example.com/1",
                "event_type": "social dance",
                "address_id": None,
            }
        ]
    )
    address_df = pd.DataFrame(columns=["address_id", "postal_code"])

    def _fake_read_sql(query, engine):  # type: ignore[no-untyped-def]
        query_text = str(query)
        if "FROM events" in query_text:
            return events_df.copy()
        if "FROM address" in query_text:
            return address_df.copy()
        raise AssertionError(query_text)

    class _FakeModel:
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(dedup_llm.pd, "read_sql", _fake_read_sql)
    monkeypatch.setattr(dedup_llm, "SentenceTransformer", _FakeModel)
    try:
        handler.deduplicate_with_embeddings()
    finally:
        dedup_llm.duplicates_path = original_duplicates_path

    artifact = tmp_path / "unparseable_datetimes.csv"
    assert artifact.exists()
    artifact_df = pd.read_csv(artifact)
    assert len(artifact_df) == 1
    assert artifact_df.iloc[0]["event_name"] == "Bad Time Event"

