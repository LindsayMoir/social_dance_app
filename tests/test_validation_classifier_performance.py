import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "validation")))

from test_runner import ValidationTestRunner


class _StubDBHandler:
    def __init__(self) -> None:
        self.metric_calls = []

    def execute_query(self, query, params=None):
        return [
            {
                "link": "https://example.org/a",
                "archetype": "incomplete_event",
                "classification_stage": "ml",
                "classification_confidence": 0.91,
                "classification_owner_step": "rd_ext.py",
                "classification_subtype": "ml_incomplete_event",
                "replay_row_count": 2,
                "replay_true_count": 2,
                "replay_url_success": True,
            },
            {
                "link": "https://example.org/b",
                "archetype": "simple_page",
                "classification_stage": "rule",
                "classification_confidence": 0.99,
                "classification_owner_step": "scraper.py",
                "classification_subtype": "event_detail",
                "replay_row_count": 1,
                "replay_true_count": 0,
                "replay_url_success": False,
            },
        ]

    def record_metric_observation(self, **kwargs):
        self.metric_calls.append(kwargs)


def test_summarize_classifier_performance_reports_stage_specific_accuracy() -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.db_handler = _StubDBHandler()

    summary = runner._summarize_classifier_performance("run-123")

    assert summary["status"] == "OK"
    assert summary["ml_usage_count"] == 1
    assert summary["ml_usage_pct"] == 50.0
    assert summary["replay_url_total"] == 2
    assert summary["replay_url_accuracy_pct"] == 50.0

    stage_details = {item["stage"]: item for item in summary["stage_details"]}
    assert stage_details["ml"]["replay_url_accuracy_pct"] == 100.0
    assert stage_details["rule"]["replay_url_accuracy_pct"] == 0.0
    assert len(runner.db_handler.metric_calls) >= 2
