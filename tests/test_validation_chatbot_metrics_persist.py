import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "validation"))

from test_runner import ValidationTestRunner


def test_persist_validation_chatbot_metrics_writes_query_and_confirm_rows() -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    calls: list[tuple[str, dict | None]] = []

    class _DB:
        @staticmethod
        def execute_query(query, params=None):  # type: ignore[no-untyped-def]
            calls.append((str(query), params))
            return []

    runner.db_handler = _DB()  # type: ignore[attr-defined]

    runner._persist_validation_chatbot_metrics(
        [
            {
                "question": "What is playing at coda tonight?",
                "category": "static",
                "sql_query": "SELECT * FROM events LIMIT 30",
                "execution_success": True,
                "result_count": 3,
                "timestamp": "2026-03-13T21:15:00",
                "execution_duration_ms": 1200.0,
                "clarification_turns_executed": 2,
                "clarification_depth_target": 2,
                "clarification_depth_achieved": 2,
            }
        ]
    )

    request_rows = [c for c in calls if "chatbot_request_metrics" in c[0]]
    stage_rows = [c for c in calls if "chatbot_stage_metrics" in c[0]]
    assert len(request_rows) >= 3  # /query + 2 x /confirm
    assert len(stage_rows) >= 3

    endpoints = [str((params or {}).get("endpoint", "")) for _, params in request_rows]
    assert "/query" in endpoints
    assert "/confirm" in endpoints
