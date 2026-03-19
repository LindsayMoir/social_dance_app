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
    runner.config = {}
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
        duplicate_summary={
            "duplicate_rate_per_100_events": 4.5,
            "severe_duplicate_rate_per_100_events": 1.5,
        },
        coverage_summary={
            "source_hit_rate_pct": 80.0,
            "event_capture_rate_pct": 60.0,
            "watchlist_source": "coverage_watchlist_csv",
        },
        holdout_summary={
            "available": True,
            "holdout_version": "v1",
            "replay_url_accuracy_pct": 78.0,
        },
        domain_capped_summary={
            "available": True,
            "per_domain_cap": 3,
            "replay_url_accuracy_pct": 75.0,
        },
    )
    assert run_scorecard["kpis"]["database_accuracy"]["score"] == 82.5
    assert run_scorecard["kpis"]["events_coverage"]["score"] == 85.0
    assert run_scorecard["kpis"]["chatbot_quality"]["score"] == 88.0
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["duplicate_rate_per_100_events"] == 4.5
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["classifier_effect"]["ml_usage_pct"] == 25.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["watchlist_source_hit_rate_pct"] == 80.0
    assert run_scorecard["evaluation_scope"]["uses_holdout"] is True
    assert run_scorecard["evaluation_scope"]["holdout_summary"]["replay_url_accuracy_pct"] == 78.0
    assert run_scorecard["evaluation_scope"]["uses_dev_split"] is True
    assert run_scorecard["evaluation_scope"]["dev_version"] == "v1"
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
        duplicate_summary={
            "duplicate_rate_per_100_events": 3.2,
            "severe_duplicate_rate_per_100_events": 1.1,
        },
        coverage_summary={
            "source_hit_rate_pct": 84.0,
            "event_capture_rate_pct": 72.0,
        },
        holdout_summary={
            "replay_url_accuracy_pct": 77.0,
        },
        domain_capped_summary={
            "replay_url_accuracy_pct": 74.0,
        },
    )

    metric_keys = [call["metric_key"] for call in fake_db.calls]
    assert metric_keys == [
        "phase1_pipeline_duration_minutes",
        "phase1_total_llm_cost_usd",
        "phase1_chatbot_response_within_15s_pct",
        "phase1_chatbot_answer_correctness_pct",
        "duplicate_rate_per_100_events",
        "severe_duplicate_rate_per_100_events",
        "coverage_watchlist_source_hit_rate_pct",
        "coverage_watchlist_event_capture_rate_pct",
        "holdout_replay_url_accuracy_pct",
        "domain_capped_replay_url_accuracy_pct",
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


def test_phase2_duplicate_and_coverage_summaries() -> None:
    runner = _build_runner()

    class DuplicateDb:
        def execute_query(self, query, params=None):
            q = " ".join(str(query).split())
            if "SELECT COUNT(*) FROM events" in q:
                return [(20,)]
            if "GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1 ORDER BY row_count DESC" in q:
                return [
                    ("https://example.com/a", "Friday Salsa", "2026-03-20", "19:00:00", 3),
                    ("https://example.com/b", "Sunday Swing", "2026-03-22", "18:00:00", 2),
                ]
            if "COUNT(DISTINCT COALESCE(NULLIF(TRIM(url), ''), '(no url)')) > 1" in q:
                return [
                    ("Friday Salsa", "2026-03-20", "19:00:00", 10, 2, 2),
                ]
            raise AssertionError(q)

    runner.db_handler = DuplicateDb()
    duplicate_summary = runner._summarize_duplicate_audit()
    assert duplicate_summary["duplicate_clusters_count"] == 2
    assert duplicate_summary["duplicate_rows"] == 3
    assert duplicate_summary["duplicate_rate_per_100_events"] == 15.0
    assert duplicate_summary["severe_duplicate_rate_per_100_events"] == 5.0

    class FakeValidator:
        days_back = 7

        def _normalize_link_variants(self, url):
            return [url]

        def _query_recent_url_rows(self, variants, window_days):
            if variants[0].endswith("/hit"):
                return [("https://example.com/hit", "source", True, 1, "2026-03-18 10:00:00", "salsa")]
            if variants[0].endswith("/seen"):
                return [("https://example.com/seen", "source", False, 1, "2026-03-18 10:00:00", "")]
            return []

        def _query_recent_child_success_count(self, variants, window_days):
            return 2 if variants[0].endswith("/hit") else 0

        def _query_latest_overall_row(self, variants):
            if variants[0].endswith("/seen"):
                return ("https://example.com/seen", False, 1, "2026-03-18 10:00:00", "")
            return None

    runner._make_scraping_validator = lambda: FakeValidator()
    runner._load_coverage_watchlist_rows = lambda: (
        [
            {"source_name": "Hit Source", "source_url": "https://example.com/hit", "priority": "high"},
            {"source_name": "Seen Source", "source_url": "https://example.com/seen", "priority": "medium"},
            {"source_name": "Missed Source", "source_url": "https://example.com/missed", "priority": "high"},
        ],
        "test_watchlist",
    )
    coverage_summary = runner._summarize_coverage_watchlist()
    assert coverage_summary["source_hit_rate_pct"] == 66.67
    assert coverage_summary["event_capture_rate_pct"] == 33.33
    assert coverage_summary["priority_source_hit_rate_pct"] == 50.0
    assert coverage_summary["watchlist_source"] == "test_watchlist"


def test_phase3_holdout_domain_caps_and_guardrails() -> None:
    runner = _build_runner()
    accuracy_replay = {
        "rows": [
            {
                "is_match": True,
                "mismatch_category": "",
                "baseline": {"url": "https://www.redhotswing.com"},
                "baseline_event_id": 1,
            },
            {
                "is_match": False,
                "mismatch_category": "wrong_time",
                "baseline": {"url": "https://vlda.ca/resources/"},
                "baseline_event_id": 2,
            },
            {
                "is_match": True,
                "mismatch_category": "",
                "baseline": {"url": "https://example.org/a"},
                "baseline_event_id": 3,
            },
            {
                "is_match": False,
                "mismatch_category": "wrong_date",
                "baseline": {"url": "https://example.org/b"},
                "baseline_event_id": 4,
            },
            {
                "is_match": True,
                "mismatch_category": "",
                "baseline": {"url": "https://another.org/c"},
                "baseline_event_id": 5,
            },
        ]
    }

    holdout_summary = runner._summarize_holdout_replay(accuracy_replay)
    assert holdout_summary["available"] is True
    assert holdout_summary["replay_urls_seen"] == 2
    assert holdout_summary["replay_url_accuracy_pct"] == 50.0

    domain_capped_summary = runner._summarize_domain_capped_replay(accuracy_replay, per_domain_cap=1)
    assert domain_capped_summary["sampled_rows"] == 4
    assert domain_capped_summary["sampled_domains"] == 4
    assert domain_capped_summary["replay_url_accuracy_pct"] == 75.0

    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-guardrails",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={"coverage_accuracy_pct": 74.0, "replay_accuracy_pct": 70.0, "total_rows": 5, "true_count": 3, "false_count": 2},
        scraping_results={"summary": {"important_urls_checked": 10, "failed_count": 4}, "source_distribution": {"status": "WARNING"}},
        runtime_summary={"pipeline_duration_minutes": 120.0},
        llm_cost_summary={"summary": {"total_usd": 4.0}},
        chatbot_quality_summary={"summary": {"chatbot_response_within_15s_pct": 85.0, "chatbot_answer_correctness_pct": 80.0}},
        classifier_performance_summary={},
        duplicate_summary={"duplicate_rate_per_100_events": 3.0, "severe_duplicate_rate_per_100_events": 2.5},
        coverage_summary={"source_hit_rate_pct": 55.0, "event_capture_rate_pct": 40.0, "watchlist_source": "test"},
        holdout_summary=holdout_summary,
        domain_capped_summary=domain_capped_summary,
    )
    guardrails = runner._evaluate_phase3_guardrails(run_scorecard)
    assert guardrails["status"] == "FAIL"
    assert {item["metric_key"] for item in guardrails["violations"]} == {
        "database_accuracy_min_pct",
        "holdout_replay_url_accuracy_min_pct",
        "coverage_watchlist_source_hit_rate_min_pct",
        "severe_duplicate_rate_max_per_100_events",
        "chatbot_response_within_15s_min_pct",
        "chatbot_answer_correctness_min_pct",
    }
