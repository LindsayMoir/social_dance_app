from datetime import date
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from db import DatabaseHandler


def _build_handler_with_rows(rows):
    handler = DatabaseHandler.__new__(DatabaseHandler)
    writes = []

    def _execute_query(query, params=None):
        if query.strip().lower().startswith("select"):
            return rows
        writes.append((query, params))
        return 1

    handler.execute_query = _execute_query
    return handler, writes


def test_check_dow_date_consistent_updates_day_not_date():
    handler, writes = _build_handler_with_rows(
        [
            (101, date(2026, 4, 2), "Wednesday"),  # mismatch: actual is Thursday
        ]
    )

    handler.check_dow_date_consistent()

    assert len(writes) == 1
    query, params = writes[0]
    assert "SET day_of_week = :corrected_day" in query
    assert "start_date" not in query
    assert params["event_id"] == 101
    assert params["corrected_day"] == "Thursday"


def test_check_dow_date_consistent_no_write_when_already_consistent():
    handler, writes = _build_handler_with_rows(
        [
            (102, date(2026, 4, 1), "Wednesday"),
        ]
    )

    handler.check_dow_date_consistent()

    assert writes == []
