import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from db import DatabaseHandler


def test_get_historical_classifier_memory_requires_replay_backed_success() -> None:
    db_handler = DatabaseHandler.__new__(DatabaseHandler)
    db_handler.normalize_url = lambda url: url
    db_handler._normalize_for_compare = lambda url: url.strip().lower()
    db_handler.execute_query = lambda query, params=None: [
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.88, True),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.86, True),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.85, True),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.87, False),
    ]

    hint = db_handler.get_historical_classifier_memory("https://example.org/upcoming")

    assert hint is not None
    assert hint["archetype"] == "incomplete_event"
    assert hint["owner_step"] == "rd_ext.py"
    assert hint["sample_count"] == 4
    assert hint["success_count"] == 3
    assert hint["success_rate"] == 0.75


def test_get_historical_classifier_memory_rejects_routes_with_poor_success_rate() -> None:
    db_handler = DatabaseHandler.__new__(DatabaseHandler)
    db_handler.normalize_url = lambda url: url
    db_handler._normalize_for_compare = lambda url: url.strip().lower()
    db_handler.execute_query = lambda query, params=None: [
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.88, False),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.86, False),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.85, True),
        ("incomplete_event", "rd_ext.py", "incomplete_event", "ml", 0.87, False),
    ]

    hint = db_handler.get_historical_classifier_memory("https://example.org/upcoming")

    assert hint is None
