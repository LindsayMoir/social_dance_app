from __future__ import annotations

from datetime import datetime
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(TESTS_DIR, "validation")
if VALIDATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_DIR)

from test_runner import ValidationTestRunner


class _FakeDbHandler:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.artifact_calls: list[dict] = []

    def record_metric_observation(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def record_validation_run_artifact(self, **kwargs) -> None:
        self.artifact_calls.append(kwargs)


def _build_runner() -> ValidationTestRunner:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {}
    runner.db_handler = None
    return runner


def test_phase1_summary_builders_normalize_existing_validation_data() -> None:
    runner = _build_runner()

    runtime_summary = runner._build_phase1_runtime_summary(
        {
            "available": True,
            "run_id": "run-123",
            "start_ts": "2026-03-18 10:00:00",
            "end_ts": "2026-03-18 12:00:00",
            "runtime_seconds": 7200.0,
            "step_log_spans": [
                {"log_file": "scraper_log.txt", "duration_minutes": 70.0},
                {"log_file": "fb_log.txt", "duration_minutes": 45.5},
            ],
        }
    )
    assert runtime_summary["pipeline_duration_minutes"] == 120.0
    assert runtime_summary["critical_path_minutes"] == 70.0

    llm_cost_summary = runner._build_phase1_llm_cost_summary(
        openrouter_cost={"available": True, "cost_usd": 1.25, "requests": 10, "tokens": 1000, "start_ts": "a", "end_ts": "b"},
        openai_cost={"available": True, "cost_usd": 2.75, "requests": 5, "tokens": 500, "start_ts": "a", "end_ts": "b"},
        accuracy_replay={"true_count": 4},
    )
    assert llm_cost_summary["summary"]["total_usd"] == 4.0
    assert llm_cost_summary["summary"]["total_requests"] == 15
    assert llm_cost_summary["summary"]["cost_per_successful_replay_url_usd"] == 1.0

    chatbot_quality_summary = runner._build_phase1_chatbot_quality_summary(
        chatbot_performance={
            "source": "db",
            "start_ts": "2026-03-18 10:00:00",
            "end_ts": "2026-03-18 12:00:00",
            "query_request_count": 8,
            "confirm_request_count": 2,
            "query_latency_ms": {"p50": 5200.0, "p95": 14200.0},
            "slow_requests": [
                {"duration_ms": 17000.0},
                {"duration_ms": 9000.0},
            ],
            "unfinished_request_count": 1,
            "status": "WATCH",
            "status_reasons": ["query p95 latency high"],
        },
        chatbot_testing={
            "summary": {
                "total_tests": 20,
                "average_score": 88.0,
                "execution_success_rate": 0.95,
            },
            "problem_categories": [{"name": "Weekend Calculation", "count": 3}],
        },
    )
    assert chatbot_quality_summary["summary"]["chatbot_response_within_15s_pct"] == 90.0
    assert chatbot_quality_summary["summary"]["chatbot_answer_correctness_pct"] == 88.0
    assert chatbot_quality_summary["summary"]["chatbot_sql_validity_pct"] == 95.0
    assert chatbot_quality_summary["summary"]["chatbot_user_visible_error_rate_pct"] == 10.0

    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-123",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={"coverage_accuracy_pct": 82.5, "replay_accuracy_pct": 77.0, "total_rows": 10, "true_count": 8, "false_count": 2},
        scraping_results={"summary": {"important_urls_checked": 20, "failed_count": 3, "whitelist_failures": 1, "edge_case_failures": 0}, "source_distribution": {"status": "PASS"}},
        runtime_summary=runtime_summary,
        llm_cost_summary=llm_cost_summary,
        chatbot_quality_summary=chatbot_quality_summary,
        classifier_performance_summary={
            "ml_usage_pct": 25.0,
            "stage_counts": {"rule": 6, "ml": 2},
            "stage_details": [{"stage": "rule", "replay_url_accuracy_pct": 80.0}],
        },
    )
    assert run_scorecard["kpis"]["database_accuracy"]["score"] == 82.5
    assert run_scorecard["kpis"]["events_coverage"]["score"] == 85.0
    assert run_scorecard["kpis"]["chatbot_quality"]["score"] == 88.0
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["classifier_effect"]["ml_usage_pct"] == 25.0
    assert run_scorecard["overall_score"]["status"] == "PRELIMINARY"


def test_phase1_scorecard_metrics_persist_key_trends() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db

    runner._persist_phase1_scorecard_metrics(
        run_id="run-456",
        runtime_summary={
            "start_ts": "2026-03-18 10:00:00",
            "end_ts": "2026-03-18 12:00:00",
            "pipeline_duration_minutes": 120.0,
        },
        llm_cost_summary={"summary": {"total_usd": 3.5}},
        chatbot_quality_summary={
            "summary": {
                "chatbot_response_within_15s_pct": 92.0,
                "chatbot_answer_correctness_pct": 87.5,
            }
        },
    )

    metric_keys = [call["metric_key"] for call in fake_db.calls]
    assert metric_keys == [
        "phase1_pipeline_duration_minutes",
        "phase1_total_llm_cost_usd",
        "phase1_chatbot_response_within_15s_pct",
        "phase1_chatbot_answer_correctness_pct",
    ]
    assert all(call["run_id"] == "run-456" for call in fake_db.calls)
    assert all(isinstance(call["window_end"], datetime) for call in fake_db.calls)


def test_phase1_artifacts_persist_by_run_id_and_type() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db

    runner._persist_validation_artifacts(
        run_id="run-789",
        artifacts={
            "run_scorecard": {"run_id": "run-789", "overall_score": {"value": 80.0}},
            "runtime_summary": {"pipeline_duration_minutes": 100.0},
            "ignore_me": [],
        },
    )

    assert [call["artifact_type"] for call in fake_db.artifact_calls] == [
        "run_scorecard",
        "runtime_summary",
    ]
    assert all(call["run_id"] == "run-789" for call in fake_db.artifact_calls)
