from __future__ import annotations

from datetime import date, datetime, time
import os
import sys
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(TESTS_DIR, "validation")
if VALIDATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_DIR)

from chatbot_evaluator import generate_chatbot_report
import test_runner as validation_test_runner
from test_runner import ValidationTestRunner
from replay_fetcher import ReplayArtifact


class _FakeDbHandler:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.artifact_calls: list[dict] = []
        self.fb_triage_calls: list[dict] = []
        self.fb_occurrence_calls: list[dict] = []
        self.query_map: dict[str, list[tuple]] = {}
        self.telemetry_integrity_report: dict = {
            "available": True,
            "run_id": "run-123",
            "status": "PASS",
            "violations": [],
            "summary": {"write_attribution_rows": 0},
            "steps": {},
        }

    def record_metric_observation(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def record_validation_run_artifact(self, **kwargs) -> None:
        self.artifact_calls.append(kwargs)

    def record_fb_block_triage_rows(self, **kwargs) -> None:
        self.fb_triage_calls.append(kwargs)

    def record_fb_block_occurrences(self, **kwargs) -> None:
        self.fb_occurrence_calls.append(kwargs)

    def execute_query(self, query, params=None, statement_timeout_ms=None):
        normalized = " ".join(str(query).split())
        for key, value in sorted(self.query_map.items(), key=lambda item: len(item[0]), reverse=True):
            if key in normalized:
                return value
        raise AssertionError(normalized)

    def build_phase1_telemetry_integrity_report(self, run_id: str) -> dict:
        payload = dict(self.telemetry_integrity_report)
        payload["run_id"] = run_id
        return payload


def _build_runner() -> ValidationTestRunner:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {}
    runner.config = {}
    runner.db_handler = None
    return runner


def _stub_code_version(runner: ValidationTestRunner) -> None:
    runner._get_code_version_info = lambda: {"git_commit": "abc123", "branch": "main"}  # type: ignore[method-assign]


def test_build_database_accuracy_manual_review_sample_writes_rows(tmp_path) -> None:
    runner = _build_runner()
    runner.db_handler = _FakeDbHandler()
    runner.db_handler.query_map["FROM events"] = [
        (101, "Friday Social", "2026-04-10", "19:00:00", "2026-04-10", "22:00:00", "Source A", "social dance", "salsa", "Hall A", "https://example.com/a", "desc a"),
        (102, "Saturday Social", "2026-04-11", "20:00:00", "2026-04-11", "23:00:00", "Source B", "live music", "swing", "Hall B", "https://other.example.org/b", "desc b"),
    ]

    original_codex_review_path = validation_test_runner.codex_review_path
    validation_test_runner.codex_review_path = lambda filename: str(tmp_path / filename)
    try:
        sample = runner._build_database_accuracy_manual_review_sample(limit=2)
    finally:
        validation_test_runner.codex_review_path = original_codex_review_path

    assert sample["mode"] == "manual_review"
    assert sample["rows_returned"] == 2
    assert os.path.exists(sample["csv_path"])
    assert sample["rows"][0]["human_label"] == ""
    assert sample["rows"][0]["domain"] == "example.com"


def test_phase1_scorecard_database_accuracy_is_manual_review_only() -> None:
    runner = _build_runner()
    _stub_code_version(runner)
    runner._get_runtime_30d_baseline_hours = lambda **kwargs: {"available": False}  # type: ignore[method-assign]
    runner._get_run_cost_30d_baseline_usd = lambda **kwargs: {"available": False}  # type: ignore[method-assign]

    scorecard = runner._build_phase1_run_scorecard(
        run_id="run-123",
        report_timestamp="2026-03-26T12:00:00",
        accuracy_replay={"status": "DISABLED", "mode": "manual_review", "manual_review_sample": {"rows_returned": 20}},
        scraping_results={"summary": {"total_important_urls": 20, "total_failures": 5}, "source_distribution": {"status": "PASS"}},
        runtime_summary={"pipeline_duration_minutes": 60.0, "pipeline_duration_hours": 1.0},
        llm_cost_summary={"summary": {"total_usd": 2.0}},
        chatbot_quality_summary={"summary": {"chatbot_answer_correctness_pct": 93.0}},
        classifier_performance_summary={},
        event_data_quality_summary={},
        field_accuracy_summary={},
        coverage_summary={},
        scraper_telemetry_summary={"steps": {}},
        dev_summary={},
        holdout_summary={},
        domain_capped_summary={},
        telemetry_integrity_summary={"summary": {}},
    )

    database_accuracy = scorecard["kpis"]["database_accuracy"]
    assert database_accuracy["score"] is None
    assert database_accuracy["summary"]["mode"] == "manual_review"
    assert database_accuracy["summary"]["manual_review_sample_size"] == 20


def test_build_database_accuracy_manual_review_html_renders_review_table() -> None:
    runner = _build_runner()

    html = runner._build_database_accuracy_manual_review_html(
        {
            "mode": "manual_review",
            "rows_returned": 1,
            "csv_path": "/tmp/database_accuracy_manual_review.csv",
            "rows": [
                {
                    "domain": "example.com",
                    "event_name": "Friday Social",
                    "source": "Source A",
                    "start_date": "2026-04-10",
                    "start_time": "19:00:00",
                    "event_type": "social dance",
                    "dance_style": "salsa",
                    "location": "Hall A",
                    "url": "https://example.com/a",
                    "description": "desc a",
                }
            ],
        }
    )

    assert "sampled event rows from the final" in html
    assert "Friday Social" in html
    assert "example.com" in html
    assert "Human Label" in html
    assert "/tmp/database_accuracy_manual_review.csv" in html
    assert "Scoring instructions:" in html
    assert "human_label" in html
    assert "review_notes" in html
    assert "url_archetype_ml_classifier_review.csv" in html


def test_build_classifier_manual_review_html_renders_review_table() -> None:
    runner = _build_runner()

    html = runner._build_classifier_manual_review_html(
        {
            "mode": "manual_review",
            "rows_returned": 1,
            "csv_path": "/tmp/url_archetype_ml_classifier_review.csv",
            "rows": [
                {
                    "sample_bucket": "true_candidate",
                    "source": "Source A",
                    "handled_by": "scraper.py",
                    "url": "https://example.com/a",
                    "classifier_stage": "ml",
                    "classifier_predicted_archetype": "simple_page",
                    "classifier_predicted_owner_step": "scraper.py",
                    "events_written": 1,
                    "survived_to_end": True,
                }
            ],
        }
    )

    assert "classifier correctness" in html
    assert "true_candidate" in html
    assert "Human Truth Archetype" in html
    assert "/tmp/url_archetype_ml_classifier_review.csv" in html
    assert "Scoring instructions:" in html
    assert "human_truth_owner_step" in html
    assert "database_event_accuracy_manual_review.csv" in html
    assert "simple_page" in html
    assert "complicated_page" in html
    assert "scraper.py" in html
    assert "rd_ext.py" in html


def test_infer_latest_pipeline_run_id_reads_archived_pipeline_logs(tmp_path, monkeypatch) -> None:
    runner = _build_runner()
    logs_dir = tmp_path / "logs"
    archived_dir = logs_dir / "logs_20260406_202409"
    archived_dir.mkdir(parents=True)
    (archived_dir / "pipeline_log.txt").write_text(
        "2026-04-06 20:28:22 - INFO - [run_id=20260406-202822-40c67542] [step=pipeline] - start\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    inferred = runner._infer_latest_pipeline_run_id("2026-04-07T13:59:03.325938")

    assert inferred == "20260406-202822-40c67542"


def test_resolve_validation_run_id_prefers_db_telemetry_over_logs() -> None:
    runner = _build_runner()
    runner.db_handler = _FakeDbHandler()
    runner.db_handler.query_map["FROM url_scrape_metrics"] = [
        ("20260406-202822-40c67542", datetime(2026, 4, 6, 20, 45, 0)),
    ]
    runner._infer_latest_pipeline_run_id = lambda timestamp=None: "log-run-id"  # type: ignore[method-assign]

    resolved = runner._resolve_validation_run_id("2026-04-07T13:59:03.325938")

    assert resolved == "20260406-202822-40c67542"


def test_build_completeness_kpis_only_html_includes_top_sources_table() -> None:
    runner = _build_runner()
    runner.db_handler = _FakeDbHandler()
    runner.db_handler.query_map["SELECT source, COUNT(*) AS counted FROM events"] = [
        ("Victoria Latin Dance Association", 226),
        ("Red Hot Swing", 123),
    ]

    html = runner._build_completeness_kpis_only_html(
        {
            "completeness": {
                "checks": [
                    {
                        "name": "Missing Required Sources",
                        "actual": "0",
                        "target": "0",
                        "delta": "+0",
                        "status": "PASS",
                        "details": "All required sources present.",
                    }
                ]
            }
        }
    )

    assert "Top 10 Sources" in html
    assert "Victoria Latin Dance Association" in html
    assert "226" in html


def test_build_phase1_scorecard_html_includes_kpi_calculation_notes() -> None:
    runner = _build_runner()

    html = runner._build_phase1_scorecard_html(
        {
            "kpis": {
                "database_accuracy": {
                    "score": None,
                    "summary": {"manual_review_sample_size": 10},
                },
                "events_coverage": {
                    "score": 97.9,
                    "summary": {"important_urls_checked": 2213, "failed_urls": 46},
                },
                "run_time": {
                    "score": 50.0,
                    "summary": {
                        "pipeline_duration_hours": 2.0,
                        "pipeline_duration_minutes": 120.0,
                        "baseline_30d_average_runtime_hours": 4.0,
                    },
                },
                "run_costs": {
                    "score": 38.7,
                    "summary": {
                        "summary": {
                            "total_usd": 1.4453,
                            "baseline_30d_average_total_usd": 3.7346,
                        }
                    },
                },
                "chatbot_quality": {
                    "score": 76.0,
                    "summary": {"summary": {"chatbot_answer_correctness_pct": 76.0}},
                },
            },
            "overall_score": {"status": "PRELIMINARY"},
            "guardrails": {"status": "PASS"},
            "telemetry_integrity": {"status": "PASS"},
        }
    )

    assert "accurate_rows / labeled_rows * 100" in html
    assert "stays <code>n/a</code> until that sample is scored and persisted" in html
    assert "(important_urls_checked - failed_urls) / important_urls_checked * 100" in html
    assert "pipeline_duration_hours / 30d_average_runtime_hours * 100" in html
    assert "total_usd / 30d_average_total_usd * 100" in html
    assert "Calculated directly from the chatbot answer-correctness percentage" in html


def test_persist_completed_database_event_accuracy_review_records_metric(tmp_path) -> None:
    runner = _build_runner()
    runner.db_handler = _FakeDbHandler()
    csv_path = tmp_path / "database_event_accuracy_manual_review.csv"
    csv_path.write_text(
        "event_id,event_name,url,human_label,review_notes\n"
        "1,Friday Social,https://example.com/a,True,\n"
        "2,Saturday Social,https://other.example.org/b,False,\n",
        encoding="utf-8",
    )

    original_codex_review_path = validation_test_runner.codex_review_path
    validation_test_runner.codex_review_path = lambda filename: str(csv_path if filename == "database_event_accuracy_manual_review.csv" else tmp_path / filename)
    try:
        summary = runner._persist_completed_database_event_accuracy_review()
    finally:
        validation_test_runner.codex_review_path = original_codex_review_path

    assert summary["correctness_pct"] == 50.0
    assert runner.db_handler.calls
    metric_call = runner.db_handler.calls[-1]
    assert metric_call["metric_key"] == "database_event_manual_accuracy_pct"
    assert metric_call["metric_value_numeric"] == 50.0


def test_build_scraper_telemetry_html_uses_images_event_yield_chart_and_detail() -> None:
    runner = _build_runner()
    runner._build_multi_metric_trend_svg = lambda **kwargs: f"<section>{kwargs['title']}</section>"  # type: ignore[method-assign]

    html = runner._build_scraper_telemetry_html(
        {
            "steps": {
                "images": {
                    "total_urls": 10,
                    "access_attempted_count": 8,
                    "access_success_rate_pct": 75.0,
                    "text_extracted_rate_pct": 62.5,
                    "keyword_hit_rate_pct": 50.0,
                    "url_event_hit_rate_pct": 37.5,
                    "urls_with_events_count": 3,
                    "extraction_success_rate_pct": 25.0,
                    "extraction_success_count": 2,
                    "events_written_total": 4,
                    "fallback_usage_rate_pct": 12.5,
                    "fallback_used_count": 1,
                }
            }
        }
    )

    assert "Images Event Yield % Trend" in html
    assert "Images Event Yield Detail" in html
    assert "URLs With Events" in html
    assert "Images OCR / Vision Success % Trend" not in html
    assert "Images OCR / Vision Detail" not in html


def test_phase1_summary_builders_normalize_existing_validation_data() -> None:
    runner = _build_runner()
    _stub_code_version(runner)

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
        llm_activity_summary={
            "top_models": [("gpt-5", 3), ("gpt-5-mini", 1)],
            "top_files": [("scraper_log.txt", 2), ("fb_log.txt", 2)],
        },
    )
    assert llm_cost_summary["summary"]["total_usd"] == 4.0
    assert llm_cost_summary["summary"]["total_requests"] == 15
    assert llm_cost_summary["summary"]["cost_per_successful_replay_url_usd"] == 1.0
    assert llm_cost_summary["by_model"] == {"gpt-5": 3.0, "gpt-5-mini": 1.0}
    assert llm_cost_summary["by_step"] == {"scraper": 2.0, "fb": 2.0}


def test_phase1_llm_cost_summary_rejects_untrusted_provider_totals() -> None:
    runner = _build_runner()

    llm_cost_summary = runner._build_phase1_llm_cost_summary(
        openrouter_cost={
            "available": True,
            "cost_usd": 9.37,
            "requests": 12208,
            "tokens": 36834437,
            "cost_basis": "window_total_api_reset_detected",
            "start_ts": "a",
            "end_ts": "b",
        },
        openai_cost={"available": True, "cost_usd": None, "requests": 8, "tokens": 500, "start_ts": "a", "end_ts": "b"},
        accuracy_replay={},
        llm_activity_summary={
            "top_models": [("openrouter:deepseek/deepseek-v3.2", 1206), ("openrouter:qwen/qwen3-coder", 3)],
            "top_files": [("images_log.txt", 10)],
        },
    )

    assert llm_cost_summary["summary"]["total_usd"] is None
    assert llm_cost_summary["providers"]["openrouter"]["cost_usd"] is None
    assert llm_cost_summary["providers"]["openrouter"]["cost_trust_status"] == "untrusted"


def test_resolve_provider_cost_window_uses_run_day_bounds() -> None:
    runner = _build_runner()

    start_ts, end_ts = runner._resolve_provider_cost_window(
        report_timestamp="2026-04-01T14:40:37",
        pipeline_runtime={"start_ts": "2026-04-01 11:39:17", "end_ts": "2026-04-01 14:20:00"},
    )

    assert start_ts.isoformat(sep=" ") == "2026-04-01 00:00:00"
    assert end_ts.isoformat(sep=" ") == "2026-04-01 14:40:37"


def test_resolve_provider_cost_utc_dates_uses_covered_utc_days() -> None:
    runner = _build_runner()

    utc_dates = runner._resolve_provider_cost_utc_dates(
        report_timestamp="2026-04-01T14:40:37",
        pipeline_runtime={"start_ts": "2026-04-01 11:39:17", "end_ts": "2026-04-01 14:20:00"},
    )

    assert utc_dates == ["2026-04-01"]


def test_build_phase1_telemetry_integrity_summary_uses_db_handler() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    fake_db.telemetry_integrity_report = {
        "available": True,
        "run_id": "placeholder",
        "status": "FAIL",
        "violations": ["step_mismatch:scraper"],
        "summary": {"write_attribution_rows": 12},
        "steps": {"scraper": {"status": "FAIL"}},
    }
    runner.db_handler = fake_db

    summary = runner._build_phase1_telemetry_integrity_summary("run-456")

    assert summary["run_id"] == "run-456"
    assert summary["status"] == "FAIL"
    assert summary["violations"] == ["step_mismatch:scraper"]


def test_phase1_scorecard_events_coverage_uses_scraping_summary_field_names() -> None:
    runner = _build_runner()
    runner._get_code_version_info = lambda: {"git_commit": "abc123", "branch": "main"}  # type: ignore[method-assign]

    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-coverage",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={},
        scraping_results={
            "summary": {
                "total_important_urls": 20,
                "total_failures": 5,
                "whitelist_failures": 1,
                "edge_case_failures": 2,
            },
            "source_distribution": {"status": "PASS"},
        },
        runtime_summary={},
        llm_cost_summary={},
        chatbot_quality_summary={},
        classifier_performance_summary={},
        event_data_quality_summary={},
        field_accuracy_summary={},
        coverage_summary={},
        scraper_telemetry_summary={},
        dev_summary={},
        holdout_summary={},
        domain_capped_summary={},
    )

    assert run_scorecard["kpis"]["events_coverage"]["score"] == 75.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["important_urls_checked"] == 20
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["failed_urls"] == 5


def test_phase1_scorecard_runtime_uses_30d_average_baseline() -> None:
    runner = _build_runner()
    runner._get_code_version_info = lambda: {"git_commit": "abc123", "branch": "main"}  # type: ignore[method-assign]
    runner._get_runtime_30d_baseline_hours = lambda **kwargs: {  # type: ignore[method-assign]
        "available": True,
        "average_runtime_hours": 4.0,
        "runs_used": 6,
        "lookback_days": 30,
    }

    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-runtime",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={},
        scraping_results={},
        runtime_summary={
            "pipeline_duration_hours": 5.0,
            "pipeline_duration_minutes": 300.0,
        },
        llm_cost_summary={},
        chatbot_quality_summary={},
        classifier_performance_summary={},
        event_data_quality_summary={},
        field_accuracy_summary={},
        coverage_summary={},
        scraper_telemetry_summary={},
        dev_summary={},
        holdout_summary={},
        domain_capped_summary={},
    )

    assert run_scorecard["kpis"]["run_time"]["score"] == 125.0
    assert run_scorecard["kpis"]["run_time"]["summary"]["baseline_30d_average_runtime_hours"] == 4.0
    assert run_scorecard["kpis"]["run_time"]["summary"]["baseline_30d_runs_used"] == 6
    assert run_scorecard["kpis"]["run_time"]["summary"]["runtime_vs_30d_average_pct"] == 125.0


def test_phase1_scorecard_run_cost_uses_30d_average_baseline() -> None:
    runner = _build_runner()
    runner._get_code_version_info = lambda: {"git_commit": "abc123", "branch": "main"}  # type: ignore[method-assign]
    runner._get_run_cost_30d_baseline_usd = lambda **kwargs: {  # type: ignore[method-assign]
        "available": True,
        "average_total_usd": 2.0,
        "runs_used": 8,
        "lookback_days": 30,
    }

    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-cost",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={},
        scraping_results={},
        runtime_summary={},
        llm_cost_summary={"summary": {"total_usd": 1.61}},
        chatbot_quality_summary={},
        classifier_performance_summary={},
        event_data_quality_summary={},
        field_accuracy_summary={},
        coverage_summary={},
        scraper_telemetry_summary={},
        dev_summary={},
        holdout_summary={},
        domain_capped_summary={},
    )

    assert run_scorecard["kpis"]["run_costs"]["score"] == 80.5
    assert run_scorecard["kpis"]["run_costs"]["summary"]["summary"]["baseline_30d_average_total_usd"] == 2.0
    assert run_scorecard["kpis"]["run_costs"]["summary"]["summary"]["baseline_30d_runs_used"] == 8
    assert run_scorecard["kpis"]["run_costs"]["summary"]["summary"]["cost_vs_30d_average_pct"] == 80.5


def test_generate_chatbot_report_includes_review_rows() -> None:
    report = generate_chatbot_report(
        [
            {
                "question": "Where can I find beginner tango classes?",
                "category": "classes",
                "interpretation": "Looking for beginner tango classes.",
                "sql_query": "SELECT * FROM events WHERE dance_style ILIKE '%tango%'",
                "execution_success": True,
                "result_count": 3,
                "evaluation": {
                    "score": 88,
                    "reasoning": "Missed the default social-dance filter.",
                    "criteria_matched": ["dance_style"],
                    "criteria_missed": ["timeframe"],
                    "sql_issues": ["missing default filter"],
                    "interpretation_evaluation": {
                        "score": 90,
                        "issues": ["minor ambiguity"],
                        "passed": True,
                    },
                },
            }
        ],
        output_dir="output",
    )

    assert len(report["review_rows"]) == 1
    assert report["review_rows"][0]["question"] == "Where can I find beginner tango classes?"
    assert report["review_rows"][0]["score"] == 88
    assert report["review_rows"][0]["criteria_missed"] == ["timeframe"]
    assert report["review_rows"][0]["sql_issues"] == ["missing default filter"]


def test_build_chatbot_html_includes_row_by_row_scoring_table() -> None:
    runner = _build_runner()
    html = runner._build_chatbot_html(
        {
            "summary": {
                "total_tests": 1,
                "average_score": 88.0,
                "execution_success_rate": 1.0,
                "score_distribution": {
                    "excellent (90-100)": 0,
                    "good (70-89)": 1,
                    "fair (50-69)": 0,
                    "poor (<50)": 0,
                },
            },
            "review_rows": [
                {
                    "question": "Where can I find beginner tango classes?",
                    "sql_query": "SELECT * FROM events WHERE dance_style ILIKE '%tango%'",
                    "score": 88,
                    "reasoning": "Missed the default social-dance filter.",
                    "criteria_missed": ["timeframe"],
                    "sql_issues": ["missing default filter"],
                }
            ],
        }
    )

    assert "Row-by-Row Chatbot Scoring" in html
    assert "Where can I find beginner tango classes?" in html
    assert "Missed the default social-dance filter." in html
    assert "missing default filter" in html


def test_build_phase1_scorecard_html_omits_redundant_summary_bullets() -> None:
    runner = _build_runner()
    html = runner._build_phase1_scorecard_html(
        {
            "overall_score": {"status": "FAIL"},
            "guardrails": {"status": "FAIL"},
            "kpis": {
                "run_time": {"summary": {"pipeline_duration_minutes": 339.57}},
                "run_costs": {"summary": {"summary": {"total_usd": 1.6115}}},
                "chatbot_quality": {
                    "summary": {
                        "summary": {
                            "chatbot_response_within_15s_pct": 91.0,
                            "chatbot_answer_correctness_pct": 93.0,
                        }
                    }
                },
            },
            "telemetry_integrity": {"status": "FAIL"},
            "evaluation_scope": {
                "holdout_summary": {"replay_url_accuracy_pct": 100.0},
                "domain_capped_summary": {"replay_url_accuracy_pct": 66.67},
            },
        }
    )

    assert "Chatbot within 15s: 91.00%" in html
    assert "Domain-capped replay URL accuracy" not in html
    assert "Chatbot correctness" not in html
    assert "Holdout replay URL accuracy" not in html


def test_send_email_notification_attaches_only_comprehensive_report(tmp_path, monkeypatch) -> None:
    runner = _build_runner()
    runner.validation_config = {"reporting": {"output_dir": str(tmp_path)}}

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    html_report = reports_dir / "comprehensive_test_report.html"
    html_report.write_text("<html>report</html>", encoding="utf-8")

    sent_payload: dict = {}

    def _fake_send_report_email(report_summary, attachment_paths, test_type):  # type: ignore[no-untyped-def]
        sent_payload["report_summary"] = report_summary
        sent_payload["attachment_paths"] = attachment_paths
        sent_payload["test_type"] = test_type
        return True

    monkeypatch.setattr(validation_test_runner, "send_report_email", _fake_send_report_email)
    monkeypatch.setattr(validation_test_runner, "reports_path", lambda name: str(html_report))

    runner._send_email_notification(
        {
            "overall_status": "PASS",
            "timestamp": "2026-03-26T12:00:00",
            "chatbot_testing": {"summary": {"total_tests": 1, "execution_success_rate": 1.0, "average_score": 95.0}},
            "scraping_validation": {"summary": {"total_failures": 0, "whitelist_failures": 0}},
        }
    )

    assert sent_payload["attachment_paths"] == [str(html_report)]
    assert sent_payload["test_type"] == "Pre-Commit Validation"


def test_summarize_fb_block_health_includes_private_unavailable_triage(tmp_path, monkeypatch) -> None:
    runner = _build_runner()
    runner._load_coverage_watchlist_rows = lambda: ([{"source_url": "https://www.facebook.com/groups/cubansalsaclub/"}], "test")  # type: ignore[method-assign]
    monkeypatch.chdir(tmp_path)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "fb_log.txt").write_text(
        "\n".join(
            [
                "2026-03-26 08:24:16 - INFO - [run_id=run-1] [step=fb] - fb access check: phase=extract_event_links requested_url=https://www.facebook.com/groups/cubansalsaclub/posts/26145054348414416/events/ current_url=https://www.facebook.com/groups/cubansalsaclub/posts/26145054348414416/events/ state=blocked reason=private_or_unavailable_content",
                "2026-03-26 08:44:07 - INFO - [run_id=run-1] [step=fb] - fb access check: attempt=1 phase=navigate requested_url=https://www.facebook.com/events/909522138040202/ current_url=https://www.facebook.com/events/909522138040202/ state=blocked reason=private_or_unavailable_content",
                "2026-03-26 08:44:14 - INFO - [run_id=run-1] [step=fb] - fb access check: attempt=2 phase=navigate requested_url=https://www.facebook.com/events/909522138040202/ current_url=https://www.facebook.com/events/909522138040202/ state=blocked reason=private_or_unavailable_content",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = runner._summarize_fb_block_health("2026-03-26T10:00:00")

    assert summary["private_unavailable_total_attempts"] == 3
    assert summary["private_unavailable_unique_urls"] == 2
    assert {"category": "group_post_events_tab", "count": 1} in summary["private_unavailable_by_category"]
    assert {"category": "event_detail_page", "count": 1} in summary["private_unavailable_by_category"]
    assert summary["private_unavailable_sources"][0]["source_key"] in {"facebook_event_detail", "group:cubansalsaclub"}
    assert any(entry.get("important_source") is True for entry in summary["private_unavailable_sources"])
    assert any(entry.get("source_key") == "group:cubansalsaclub" for entry in summary["important_private_unavailable_sources"])


def test_build_fb_block_health_html_renders_private_unavailable_triage() -> None:
    runner = _build_runner()

    html = runner._build_fb_block_health_html(
        {
            "explicit_block_count": 0,
            "strike1_count": 0,
            "abort_count": 0,
            "avg_wait_seconds": 0.0,
            "fb_target_urls": 4,
            "instagram_target_urls": 0,
            "target_total_urls": 4,
            "attempted_total_urls": 4,
            "successful_total_urls": 1,
            "success_rate": 0.25,
            "target_coverage_rate": 0.25,
            "throttle_detected": False,
            "fb_successes_before_throttle": 1,
            "numerator_fb_ig": 1,
            "denominator_fb_ig": 4,
            "start_ts": "2026-03-26 08:00:00",
            "end_ts": "2026-03-26 10:00:00",
            "window_hours": 24,
            "private_unavailable_total_attempts": 3,
            "private_unavailable_unique_urls": 2,
            "private_unavailable_by_category": [
                {"category": "event_detail_page", "count": 1},
                {"category": "group_post_events_tab", "count": 1},
            ],
            "private_unavailable_sources": [
                {
                    "source_key": "group:cubansalsaclub",
                    "important_source": True,
                    "category": "group_post_events_tab",
                    "unique_urls": 1,
                    "attempt_count": 1,
                    "sample_url": "https://www.facebook.com/groups/cubansalsaclub/posts/26145054348414416/events/",
                }
            ],
            "important_private_unavailable_sources": [
                {
                    "source_key": "group:cubansalsaclub",
                    "important_source": True,
                    "category": "group_post_events_tab",
                    "unique_urls": 1,
                    "attempt_count": 1,
                    "sample_url": "https://www.facebook.com/groups/cubansalsaclub/posts/26145054348414416/events/",
                }
            ],
        }
    )

    assert "Private/Unavailable Cases:" in html
    assert "Private/Unavailable Triage" in html
    assert "Important Source" in html
    assert "Important Source Triage" in html
    assert "group:cubansalsaclub" in html
    assert "group_post_events_tab" in html


def test_build_image_date_rejections_html_renders_rows() -> None:
    runner = _build_runner()

    html = runner._build_image_date_rejections_html(
        {
            "total_rejections": 2,
            "unique_urls": 1,
            "by_reason": [{"reason": "unresolved_image_date_conflict", "count": 2}],
            "rejections": [
                {
                    "url": "https://www.instagram.com/p/test/#image=abc",
                    "event_name": "Tuesday Night WCS Dance",
                    "start_date": "2026-03-31",
                    "reason": "unresolved_image_date_conflict",
                    "poster_type": "single_event",
                    "detected_date": "2026-03-24",
                    "schedule_dates": [],
                }
            ],
        }
    )

    assert "Total rejected rows:</strong> 2" in html
    assert "Tuesday Night WCS Dance" in html
    assert "unresolved_image_date_conflict" in html
    assert "single_event" in html


def test_summarize_image_date_rejections_reads_structured_log_payload(tmp_path) -> None:
    runner = _build_runner()
    log_path = tmp_path / "images_log.txt"
    log_path.write_text(
        (
            "2026-03-26 09:00:00 [root] INFO: image_date_rejection: "
            "{\"url\":\"https://www.instagram.com/p/test/#image=abc\","
            "\"parent_url\":\"https://www.instagram.com/p/test/\","
            "\"poster_type\":\"single_event\","
            "\"detected_date\":\"2026-03-24\","
            "\"schedule_dates\":[],"
            "\"reason\":\"unresolved_image_date_conflict\","
            "\"event_name\":\"Tuesday Night WCS Dance\","
            "\"start_date\":\"2026-03-31\","
            "\"timestamp\":\"2026-03-26T09:00:00\"}\n"
        ),
        encoding="utf-8",
    )
    runner._get_llm_activity_log_files = lambda: [str(log_path)]  # type: ignore[method-assign]

    summary = runner._summarize_image_date_rejections(
        report_timestamp="2026-03-26T10:00:00",
        pipeline_runtime_summary={
            "start_ts": "2026-03-26 08:00:00",
            "end_ts": "2026-03-26 10:00:00",
        },
    )

    assert summary["total_rejections"] == 1
    assert summary["unique_urls"] == 1
    assert summary["by_reason"] == [{"reason": "unresolved_image_date_conflict", "count": 1}]
    assert summary["rejections"][0]["event_name"] == "Tuesday Night WCS Dance"


def test_build_top_trend_dashboard_separates_classifier_usage_from_accuracy() -> None:
    runner = _build_runner()
    captured_series: list[list[dict]] = []

    def _fake_metric_trend_svg(**kwargs):
        return f"<section>{kwargs['title']}</section>"

    def _fake_multi_metric_trend_svg(**kwargs):
        captured_series.append(list(kwargs.get("series", [])))
        return f"<section>{kwargs['title']}</section>"

    runner._build_metric_trend_svg = _fake_metric_trend_svg  # type: ignore[method-assign]
    runner._build_multi_metric_trend_svg = _fake_multi_metric_trend_svg  # type: ignore[method-assign]

    html = runner._build_top_trend_dashboard_html()

    assert "Classifier Usage" in html
    assert "Database Event Accuracy (Manual Review)" in html
    assert "Classifier Replay Accuracy" not in html
    assert "Classifier Trends" not in html
    assert captured_series
    metric_keys = {str(item.get("metric_key")) for item in captured_series[0] if isinstance(item, dict)}
    assert "classifier_ml_usage_pct" in metric_keys
    assert "classifier_manual_correctness_pct" in metric_keys


def test_build_metric_trend_svg_includes_textual_explanation() -> None:
    runner = _build_runner()
    runner._load_metric_history = lambda metric_key, days=90: [  # type: ignore[method-assign]
        {"timestamp": "2026-03-20T10:00:00", "value": 70.0},
        {"timestamp": "2026-03-24T10:00:00", "value": 75.0},
    ]

    html = runner._build_metric_trend_svg(
        metric_key="validation_replay_accuracy_pct",
        title="Accuracy Trend (DB persisted metric_observations)",
        days=180,
        value_format="percent",
    )

    assert "This graph shows" in html
    assert "validation_replay_accuracy_pct" in html
    assert "75.0%" in html
    assert "increased by 5.0%" in html


def test_build_multi_metric_trend_svg_includes_textual_explanation() -> None:
    runner = _build_runner()

    history = {
        "classifier_ml_usage_pct": [
            {"timestamp": "2026-03-20T10:00:00", "value": 12.0},
            {"timestamp": "2026-03-24T10:00:00", "value": 15.0},
        ],
        "classifier_rule_replay_url_accuracy_pct": [
            {"timestamp": "2026-03-20T10:00:00", "value": 88.0},
            {"timestamp": "2026-03-24T10:00:00", "value": 82.0},
        ],
    }
    runner._load_metric_history = lambda metric_key, days=90: history.get(metric_key, [])  # type: ignore[method-assign]

    html = runner._build_multi_metric_trend_svg(
        title="Classifier Replay Accuracy",
        days=180,
        value_format="percent",
        series=[
            {
                "metric_key": "classifier_ml_usage_pct",
                "label": "ML Usage %",
                "color": "#d94841",
            },
            {
                "metric_key": "classifier_rule_replay_url_accuracy_pct",
                "label": "Rule Replay URL Accuracy %",
                "color": "#2ca02c",
            },
        ],
    )

    assert "This graph compares related metrics" in html
    assert "ML Usage %: 15.0%" in html
    assert "Rule Replay URL Accuracy %: 82.0%" in html
    assert "increased by 3.0%" in html
    assert "decreased by 6.0%" in html


def test_load_metric_history_skips_empty_classifier_usage_placeholder_runs() -> None:
    runner = _build_runner()
    runner.db_handler = SimpleNamespace(
        execute_query=lambda query, params=None: [
            ("2026-03-23T10:00:00", 0.0, "na", '{"stage_counts": {}}'),
            ("2026-03-24T10:00:00", 15.0, "20260324-abc", '{"stage_counts": {"ml": 15, "rule": 85}}'),
        ]
    )

    history = runner._load_metric_history("classifier_ml_usage_pct", days=180)

    assert len(history) == 1
    assert history[0]["value"] == 15.0
    assert history[0]["run_id"] == "20260324-abc"


def test_build_multi_metric_trend_svg_classifier_usage_explains_metric() -> None:
    runner = _build_runner()
    history = {
        "classifier_ml_usage_pct": [
            {"timestamp": "2026-03-20T10:00:00", "value": 21.0},
            {"timestamp": "2026-03-24T10:00:00", "value": 15.0},
        ],
        "classifier_manual_correctness_pct": [
            {"timestamp": "2026-03-24T10:00:00", "value": 75.0, "notes": {"labeled_rows": 8}},
        ],
    }
    runner._load_metric_history = lambda metric_key, days=90: history.get(metric_key, [])  # type: ignore[method-assign]

    html = runner._build_multi_metric_trend_svg(
        title="Classifier Usage",
        days=180,
        value_format="percent",
        series=[
            {
                "metric_key": "classifier_ml_usage_pct",
                "label": "ML Usage %",
                "color": "#d94841",
            },
            {
                "metric_key": "classifier_manual_correctness_pct",
                "label": "Manual Correctness %",
                "color": "#1f77b4",
            },
        ],
    )

    assert "share of classified URLs handled by the ML classifier stage" in html
    assert "classifying URLs/pages for routing and handling decisions in the scraping pipeline" in html
    assert "ml_classified_urls / total_classified_urls * 100" in html
    assert "runs with no classifier data are excluded rather than plotted as 0%" in html
    assert "Manual review correctness from the scored CSV is 75.0%" in html
    assert "Manual Correctness %: 75.0%" in html
    assert "previous run with classifier data" in html


def test_build_metric_trend_svg_database_event_accuracy_explains_metric() -> None:
    runner = _build_runner()
    history = [
        {"timestamp": "2026-03-24T10:00:00", "value": 80.0, "notes": {"labeled_rows": 10}},
        {"timestamp": "2026-04-01T10:00:00", "value": 90.0, "notes": {"labeled_rows": 10}},
    ]
    runner._load_metric_history = lambda metric_key, days=90: history if metric_key == "database_event_manual_accuracy_pct" else []  # type: ignore[method-assign]

    html = runner._build_metric_trend_svg(
        metric_key="database_event_manual_accuracy_pct",
        title="Database Event Accuracy (Manual Review)",
        days=180,
        value_format="percent",
    )

    assert "share of manually reviewed event rows judged accurate" in html
    assert "accurate_rows / labeled_rows * 100" in html
    assert "based on 10 labeled row(s)" in html
    assert "90.0%" in html


def test_generate_html_report_omits_replay_based_sections(tmp_path) -> None:
    runner = _build_runner()
    runner.validation_config = {"reporting": {"output_dir": str(tmp_path)}}

    report_path = tmp_path / "comprehensive_test_report.html"

    original_reports_path = validation_test_runner.reports_path
    original_codex_review_path = validation_test_runner.codex_review_path
    original_chatbot_path = validation_test_runner.chatbot_path
    validation_test_runner.reports_path = lambda filename: str(report_path if filename == "comprehensive_test_report.html" else tmp_path / filename)
    validation_test_runner.codex_review_path = lambda filename: str(tmp_path / filename)
    validation_test_runner.chatbot_path = lambda filename: str(tmp_path / filename)
    try:
        runner._summarize_address_alias_audit = lambda: {}  # type: ignore[method-assign]
        runner._summarize_llm_provider_activity = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._summarize_llm_extraction_quality = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._summarize_chatbot_performance = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._summarize_chatbot_metrics_sync = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._summarize_scraper_network_health = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._summarize_fb_block_health = lambda timestamp=None: {"run_id": "run-123"}  # type: ignore[method-assign]
        runner._summarize_fb_ig_url_funnel = lambda timestamp=None, summary=None: {}  # type: ignore[method-assign]
        runner._summarize_suspicious_deletes = lambda timestamp=None: {}  # type: ignore[method-assign]
        runner._infer_latest_pipeline_run_id = lambda timestamp=None: "run-123"  # type: ignore[method-assign]
        runner._summarize_pipeline_runtime = lambda timestamp=None, run_id=None: {}  # type: ignore[method-assign]
        runner._summarize_scraper_step_telemetry = lambda run_id=None: {"steps": {}}  # type: ignore[method-assign]
        runner._build_phase1_telemetry_integrity_summary = lambda run_id=None: {"status": "PASS"}  # type: ignore[method-assign]
        runner._summarize_openrouter_run_cost = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._summarize_openai_run_cost = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._build_phase1_runtime_summary = lambda summary=None: {}  # type: ignore[method-assign]
        runner._build_phase1_llm_cost_summary = lambda **kwargs: {"summary": {}}  # type: ignore[method-assign]
        runner._build_phase1_chatbot_quality_summary = lambda **kwargs: {"summary": {}}  # type: ignore[method-assign]
        runner._summarize_event_data_quality = lambda: {}  # type: ignore[method-assign]
        runner._summarize_field_accuracy = lambda summary=None: {}  # type: ignore[method-assign]
        runner._summarize_coverage_watchlist = lambda: {}  # type: ignore[method-assign]
        runner._summarize_dev_replay = lambda summary=None: {}  # type: ignore[method-assign]
        runner._summarize_holdout_replay = lambda summary=None: {}  # type: ignore[method-assign]
        runner._summarize_domain_capped_replay = lambda summary=None: {}  # type: ignore[method-assign]
        runner._summarize_domain_evaluation = lambda summary=None: {}  # type: ignore[method-assign]
        runner._summarize_image_date_rejections = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._build_phase1_run_scorecard = lambda **kwargs: {"guardrails": {}, "overall_score": {"value": 1.0}}  # type: ignore[method-assign]
        runner._evaluate_phase3_guardrails = lambda scorecard=None: {"status": "PASS"}  # type: ignore[method-assign]
        runner._build_recommendations_input = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._build_recommendation_plan = lambda **kwargs: {"prioritized_recommendations": []}  # type: ignore[method-assign]
        runner._build_codex_review_bundle = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._persist_phase1_scorecard_metrics = lambda **kwargs: None  # type: ignore[method-assign]
        runner._persist_scraper_step_telemetry_metrics = lambda **kwargs: None  # type: ignore[method-assign]
        runner._persist_fb_block_triage = lambda **kwargs: None  # type: ignore[method-assign]
        runner._persist_validation_artifacts = lambda **kwargs: None  # type: ignore[method-assign]
        runner._summarize_reliability_scorecard = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        runner._extract_reliability_issues = lambda *args, **kwargs: []  # type: ignore[method-assign]
        runner._evaluate_reliability_gates = lambda scorecard=None: {}  # type: ignore[method-assign]
        runner._build_optimization_plan = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        runner._update_and_summarize_reliability_history = lambda output_dir=None, reliability_scorecard=None: {}  # type: ignore[method-assign]
        runner._update_reliability_issue_registry = lambda output_dir=None, issues=None: (issues or [], {})  # type: ignore[method-assign]
        runner._build_action_queue = lambda *args, **kwargs: {}  # type: ignore[method-assign]
        runner._summarize_run_control_panel = lambda **kwargs: {}  # type: ignore[method-assign]
        runner._build_top_trend_dashboard_html = lambda runtime_summary=None: "<div>dashboard</div>"  # type: ignore[method-assign]
        runner._build_scraper_telemetry_html = lambda summary=None: "<div>telemetry</div>"  # type: ignore[method-assign]
        runner._build_database_accuracy_manual_review_html = lambda sample=None: "<div>manual review</div>"  # type: ignore[method-assign]
        runner._build_image_date_rejections_html = lambda summary=None: "<div>image date rejections</div>"  # type: ignore[method-assign]
        runner._build_completeness_kpis_only_html = lambda panel=None: "<div>completeness</div>"  # type: ignore[method-assign]
        runner._build_phase1_scorecard_html = lambda scorecard=None: "<div>scorecard</div>"  # type: ignore[method-assign]
        runner._build_phase2_integrity_html = lambda summary=None: "<div>phase 2</div>"  # type: ignore[method-assign]
        runner._build_recommendation_plan_html = lambda plan=None: "<div>recommendation plan</div>"  # type: ignore[method-assign]

        runner._generate_html_report(
            {
                "timestamp": "2026-03-26T12:00:00",
                "overall_status": "PASS",
                "accuracy_replay": {"mode": "manual_review"},
            }
        )
    finally:
        validation_test_runner.reports_path = original_reports_path
        validation_test_runner.codex_review_path = original_codex_review_path
        validation_test_runner.chatbot_path = original_chatbot_path

    html = report_path.read_text(encoding="utf-8")
    assert "Phase 3 Honest Evaluation" not in html
    assert "Domain Evaluation" not in html
    assert "Parser Improvement Workflow" not in html
    assert "<h2>7. Recommendation Plan</h2>" in html


def test_build_runtime_step_breakdown_html_limits_to_seven_rows() -> None:
    runner = _build_runner()

    html = runner._build_runtime_step_breakdown_html(
        {
            "step_spans": [
                {"log_file": "scraper_log.txt", "duration_minutes": 19.12},
                {"log_file": "fb_log.txt", "duration_minutes": 18.0},
                {"log_file": "images_log.txt", "duration_minutes": 17.0},
                {"log_file": "rd_ext_log.txt", "duration_minutes": 16.0},
                {"log_file": "ebs_log.txt", "duration_minutes": 15.0},
                {"log_file": "emails_log.txt", "duration_minutes": 14.0},
                {"log_file": "gs_log.txt", "duration_minutes": 13.0},
                {"log_file": "read_pdfs_log.txt", "duration_minutes": 12.0},
            ]
        }
    )

    assert "<th>Step</th><th>Run Time</th>" in html
    assert "scraper" in html
    assert "gs" in html
    assert "read_pdfs" not in html


def test_build_top_trend_dashboard_includes_runtime_step_breakdown() -> None:
    runner = _build_runner()

    def _fake_metric_trend_svg(**kwargs):
        return f"<section>{kwargs['title']}</section>"

    def _fake_multi_metric_trend_svg(**kwargs):
        return f"<section>{kwargs['title']}</section>"

    runner._build_metric_trend_svg = _fake_metric_trend_svg  # type: ignore[method-assign]
    runner._build_multi_metric_trend_svg = _fake_multi_metric_trend_svg  # type: ignore[method-assign]

    html = runner._build_top_trend_dashboard_html(
        {
            "step_spans": [
                {"log_file": "scraper_log.txt", "duration_minutes": 19.12},
                {"log_file": "fb_log.txt", "duration_minutes": 18.0},
            ]
        }
    )

    assert "Current-run step timings below are based on per-log-file spans" in html
    assert "<th>Step</th><th>Run Time</th>" in html
    assert "scraper" in html


def test_social_replay_preserves_source_url_instead_of_llm_mentioned_url() -> None:
    runner = _build_runner()

    class _FakeFbScraper:
        def navigate_and_maybe_login(self, _url: str) -> bool:
            return True

        def extract_event_text(self, _url: str, assume_navigated: bool = True) -> str:
            _ = assume_navigated
            return "poster text"

    runner._get_replay_fb_scraper = lambda: _FakeFbScraper()  # type: ignore[method-assign]
    runner.llm_handler = SimpleNamespace(
        generate_prompt=lambda url, text, prompt_type: ("prompt", "event_extraction"),
        query_llm=lambda url, prompt, schema_type: '{"events":[]}',
        extract_and_parse_json=lambda response, url, schema_type: [
            {
                "event_name": "salsa & bachata party",
                "start_date": "2026-03-29",
                "start_time": "13:00:00",
                "source": "la bodeguita",
                "location": "grote marktstraat",
                "url": "https://www.labodeguita.nl",
            }
        ],
    )

    payload = runner._fetch_replay_events_for_social_url("https://www.instagram.com/p/DVn5jSriOv9")

    assert payload["ok"] is True
    assert payload["events"][0]["url"] == "https://www.instagram.com/p/DVn5jSriOv9"
    assert payload["events"][0]["raw"]["mentioned_url"] == "https://www.labodeguita.nl"


def test_social_replay_routes_instagram_image_child_url_to_image_replay_helper() -> None:
    runner = _build_runner()
    child_url = "https://www.instagram.com/p/DVZsZicAc9y#image=abc123"
    runner.llm_handler = SimpleNamespace(
        generate_prompt=lambda url, text, prompt_type: ("prompt", "event_extraction"),
        query_llm=lambda url, prompt, schema_type=None: "",
        extract_and_parse_json=lambda response, url, schema_type: [],
    )
    runner._fetch_replay_events_for_instagram_image_url = lambda url: {
        "ok": True,
        "category": "",
        "details": "instagram_image_replay",
        "events": [
            {
                "event_name": "Country Wednesdays",
                "start_date": "2026-03-25",
                "start_time": "18:30:00",
                "source": "The Valencia Club",
                "location": "2162 Taylor Rd, Penryn, CA 95663",
                "url": url,
                "raw": {},
            }
        ],
    }

    payload = runner._fetch_replay_events_for_social_url(child_url)

    assert payload["ok"] is True
    assert payload["details"] == "instagram_image_replay"
    assert payload["events"][0]["url"] == child_url


def test_compare_replay_row_rejects_social_url_drift_before_field_mismatch() -> None:
    runner = _build_runner()

    comparison = runner._compare_replay_row(
        baseline_row={
            "event_name": "la bodeguita brings salsa & bachata back to the streets of the hague",
            "start_date": "2026-03-29",
            "start_time": "13:00:00",
            "source": "la bodeguita",
            "location": "de bijenkorf",
            "url": "https://www.instagram.com/p/DVn5jSriOv9",
        },
        replay_payload={
            "ok": True,
            "events": [
                {
                    "event_name": "salsa & bachata party",
                    "start_date": "2026-03-29",
                    "start_time": "13:00:00",
                    "source": "la bodeguita",
                    "location": "grote marktstraat",
                    "url": "https://www.labodeguita.nl",
                    "raw": {},
                }
            ],
        },
        strict_time_match=True,
    )

    assert comparison["is_match"] is False
    assert comparison["category"] == "wrong_replay_source_url"


def test_compare_replay_row_rejects_wrong_event_from_same_listing_page() -> None:
    runner = _build_runner()

    comparison = runner._compare_replay_row(
        baseline_row={
            "event_name": "sunday blues services",
            "start_date": "2026-03-29",
            "start_time": "16:00:00",
            "source": "livevictoria.com",
            "location": "the wicket hall (strathcona hotel)",
            "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
        },
        replay_payload={
            "ok": True,
            "events": [
                {
                    "event_name": "Flamenco Tablao - March 29, 2026",
                    "start_date": "2026-03-29",
                    "start_time": "19:00:00",
                    "source": "livevictoria.com",
                    "location": "The Mint",
                    "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                    "raw": {},
                },
                {
                    "event_name": "Victoria Symphony Presents Kluxen & Brantelid",
                    "start_date": "2026-03-29",
                    "start_time": "14:30:00",
                    "source": "livevictoria.com",
                    "location": "Royal Theatre",
                    "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                    "raw": {},
                },
            ],
        },
        strict_time_match=True,
    )

    assert comparison["is_match"] is False
    assert comparison["category"] == "wrong_replay_event_selection"


def test_compare_replay_row_prefers_same_page_candidate_with_matching_time() -> None:
    runner = _build_runner()

    comparison = runner._compare_replay_row(
        baseline_row={
            "event_name": "sunday blues services",
            "start_date": "2026-03-29",
            "start_time": "16:00:00",
            "source": "livevictoria.com",
            "location": "studio 919",
            "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
        },
        replay_payload={
            "ok": True,
            "events": [
                {
                    "event_name": "Flamenco Tablao - March 29, 2026",
                    "start_date": "2026-03-29",
                    "start_time": "19:00:00",
                    "source": "livevictoria.com",
                    "location": "The Mint",
                    "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                    "raw": {},
                },
                {
                    "event_name": "Sunday Blues Services",
                    "start_date": "2026-03-29",
                    "start_time": "16:00:00",
                    "source": "livevictoria.com",
                    "location": "Studio 919",
                    "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                    "raw": {},
                },
            ],
        },
        strict_time_match=True,
    )

    assert comparison["is_match"] is True
    assert comparison["replay"]["event_name"] == "sunday blues services"


def test_compare_replay_row_accepts_recurring_weekly_social_schedule_shift() -> None:
    runner = _build_runner()

    comparison = runner._compare_replay_row(
        baseline_row={
            "event_name": "live music",
            "start_date": "2026-03-30",
            "start_time": "14:00:00",
            "day_of_week": "Monday",
            "source": "bin 39: wine bar",
            "location": "bin 39: wine bar, ca",
            "url": "https://www.instagram.com/p/DPHME2UEVjy",
            "description": "Live music Monday-Wednesday 2-5pm Thursday-Sunday 1-7pm",
        },
        replay_payload={
            "ok": True,
            "events": [
                {
                    "event_name": "live music",
                    "start_date": "2026-03-23",
                    "start_time": "14:00:00",
                    "day_of_week": "Monday",
                    "source": "bin39winebar",
                    "location": "bin39winebar courtyard",
                    "url": "https://www.instagram.com/p/DPHME2UEVjy",
                    "raw": {
                        "description": "Monday-Wednesday 2-5pm Thursday-Sunday 1-7pm",
                    },
                }
            ],
        },
        strict_time_match=True,
    )

    assert comparison["is_match"] is True
    assert comparison["details"] == "recurring_event_schedule_match"


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
                "chatbot_p95_latency_seconds": 13.4,
            }
        },
        coverage_summary={
            "source_hit_rate_pct": 84.0,
            "event_capture_rate_pct": 72.0,
            "missed_event_rate_manual_audit_pct": 12.5,
        },
        dev_summary={
            "replay_url_accuracy_pct": 79.0,
        },
        holdout_summary={
            "replay_url_accuracy_pct": 77.0,
        },
        domain_capped_summary={
            "replay_url_accuracy_pct": 74.0,
        },
        event_data_quality_summary={
            "invalid_event_rate_pct": 1.5,
            "stale_event_rate_pct": 2.25,
        },
        field_accuracy_summary={
            "date_pct": 90.0,
            "time_pct": 80.0,
            "location_pct": 70.0,
            "source_pct": 95.0,
            "address_id_pct": 88.0,
            "dance_style_pct": 76.0,
            "description_pct": 64.0,
        },
    )

    metric_keys = [call["metric_key"] for call in fake_db.calls]
    assert metric_keys == [
        "phase1_pipeline_duration_minutes",
        "phase1_total_llm_cost_usd",
        "phase1_chatbot_response_within_15s_pct",
        "phase1_chatbot_answer_correctness_pct",
        "phase1_chatbot_p95_latency_seconds",
        "coverage_watchlist_source_hit_rate_pct",
        "coverage_watchlist_event_capture_rate_pct",
        "missed_event_rate_manual_audit_pct",
        "dev_replay_url_accuracy_pct",
        "holdout_replay_url_accuracy_pct",
        "domain_capped_replay_url_accuracy_pct",
        "invalid_event_rate_pct",
        "stale_event_rate_pct",
        "field_accuracy_date_pct",
        "field_accuracy_time_pct",
        "field_accuracy_location_pct",
        "field_accuracy_source_pct",
        "field_accuracy_address_id_pct",
        "field_accuracy_dance_style_pct",
        "field_accuracy_description_pct",
    ]
    assert all(call["run_id"] == "run-456" for call in fake_db.calls)
    assert all(isinstance(call["window_end"], datetime) for call in fake_db.calls)


def test_summarize_scraper_step_telemetry_normalizes_all_scrapers() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db
    fake_db.query_map["FROM url_scrape_metrics"] = [
        {
            "step_norm": "scraper",
            "total_urls": 10,
            "access_attempted_count": 9,
            "access_success_count": 9,
            "text_extracted_count": 8,
            "keywords_found_count": 6,
            "extraction_attempted_count": 8,
            "extraction_success_count": 5,
            "urls_with_events_count": 5,
            "events_written_total": 7,
            "ocr_attempted_count": 0,
            "ocr_success_count": 0,
            "vision_attempted_count": 0,
            "vision_success_count": 0,
            "fallback_used_count": 0,
        },
        {
            "step_norm": "images",
            "total_urls": 5,
            "access_attempted_count": 4,
            "access_success_count": 4,
            "text_extracted_count": 3,
            "keywords_found_count": 2,
            "extraction_attempted_count": 4,
            "extraction_success_count": 2,
            "urls_with_events_count": 2,
            "events_written_total": 3,
            "ocr_attempted_count": 4,
            "ocr_success_count": 3,
            "vision_attempted_count": 2,
            "vision_success_count": 1,
            "fallback_used_count": 2,
        },
    ]

    summary = runner._summarize_scraper_step_telemetry("run-telemetry")

    assert summary["available"] is True
    assert summary["steps"]["scraper"]["access_success_rate_pct"] == 100.0
    assert summary["steps"]["scraper"]["text_extracted_rate_pct"] == 88.89
    assert summary["steps"]["scraper"]["keyword_hit_rate_pct"] == 75.0
    assert summary["steps"]["scraper"]["extraction_success_rate_pct"] == 62.5
    assert summary["steps"]["images"]["text_extracted_rate_pct"] == 75.0
    assert summary["steps"]["images"]["keyword_hit_rate_pct"] == 66.67
    assert summary["steps"]["images"]["ocr_success_rate_pct"] == 75.0
    assert summary["steps"]["images"]["vision_success_rate_pct"] == 50.0
    assert summary["steps"]["images"]["fallback_usage_rate_pct"] == 50.0


def test_summarize_scraper_step_telemetry_prefers_handled_by_over_pipeline_step() -> None:
    runner = _build_runner()

    class QueryCapturingDb:
        def __init__(self) -> None:
            self.query_text = ""

        def execute_query(self, query, params=None):
            self.query_text = str(query)
            return [
                {
                    "step_norm": "images",
                    "total_urls": 5,
                    "access_attempted_count": 4,
                    "access_success_count": 4,
                    "text_extracted_count": 3,
                    "keywords_found_count": 2,
                    "extraction_attempted_count": 4,
                    "extraction_success_count": 2,
                    "urls_with_events_count": 2,
                    "events_written_total": 3,
                    "ocr_attempted_count": 4,
                    "ocr_success_count": 3,
                    "vision_attempted_count": 2,
                    "vision_success_count": 1,
                    "fallback_used_count": 2,
                },
                {
                    "step_norm": "rd_ext",
                    "total_urls": 7,
                    "access_attempted_count": 6,
                    "access_success_count": 6,
                    "text_extracted_count": 6,
                    "keywords_found_count": 5,
                    "extraction_attempted_count": 6,
                    "extraction_success_count": 4,
                    "urls_with_events_count": 4,
                    "events_written_total": 9,
                    "ocr_attempted_count": 0,
                    "ocr_success_count": 0,
                    "vision_attempted_count": 0,
                    "vision_success_count": 0,
                    "fallback_used_count": 0,
                },
            ]

    runner.db_handler = QueryCapturingDb()

    summary = runner._summarize_scraper_step_telemetry("run-telemetry")

    assert "REPLACE(COALESCE(handled_by, ''), '.py', '')" in runner.db_handler.query_text
    assert summary["steps"]["images"]["total_urls"] == 5
    assert summary["steps"]["rd_ext"]["total_urls"] == 7


def test_persist_scraper_step_telemetry_metrics_records_trend_keys() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db

    runner._persist_scraper_step_telemetry_metrics(
        run_id="run-telemetry",
        runtime_summary={
            "start_ts": "2026-03-18 10:00:00",
            "end_ts": "2026-03-18 12:00:00",
        },
        scraper_telemetry_summary={
            "available": True,
            "steps": {
                "rd_ext": {
                    "access_success_rate_pct": 80.0,
                    "text_extracted_rate_pct": 70.0,
                    "keyword_hit_rate_pct": 60.0,
                    "extraction_success_rate_pct": 50.0,
                    "url_event_hit_rate_pct": 40.0,
                    "ocr_success_rate_pct": None,
                    "vision_success_rate_pct": None,
                    "fallback_usage_rate_pct": None,
                },
                "images": {
                    "access_success_rate_pct": 90.0,
                    "text_extracted_rate_pct": 85.0,
                    "keyword_hit_rate_pct": 75.0,
                    "extraction_success_rate_pct": 65.0,
                    "url_event_hit_rate_pct": 55.0,
                    "ocr_success_rate_pct": 88.0,
                    "vision_success_rate_pct": 44.0,
                    "fallback_usage_rate_pct": 22.0,
                },
            },
        },
    )

    metric_keys = {call["metric_key"] for call in fake_db.calls}
    assert "scraper_rd_ext_access_success_rate_pct" in metric_keys
    assert "scraper_rd_ext_extraction_success_rate_pct" in metric_keys
    assert "scraper_images_access_success_rate_pct" in metric_keys
    assert "scraper_images_ocr_success_rate_pct" in metric_keys
    assert "scraper_images_vision_success_rate_pct" in metric_keys


def test_build_multi_metric_trend_svg_renders_single_point_series() -> None:
    runner = _build_runner()
    runner._load_metric_history = lambda metric_key, days=120: [  # type: ignore[method-assign]
        {"timestamp": "2026-03-23T05:54:12", "value": 23.6}
    ]

    html = runner._build_multi_metric_trend_svg(
        title="Scraper Access Success % Trend",
        days=180,
        value_format="percent",
        series=[
            {"metric_key": "scraper_ebs_access_success_rate_pct", "label": "ebs", "color": "#d62728"},
        ],
    )

    assert "<circle " in html
    assert "Single run of history; showing point marker only." in html
    assert "2026-03-23" in html


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


def test_persist_fb_block_triage_writes_relational_rows_and_metrics() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db

    runner._persist_fb_block_triage(
        run_id="run-fb",
        fb_block_summary={
            "start_ts": "2026-03-26 08:00:00",
            "end_ts": "2026-03-26 10:00:00",
            "private_unavailable_total_attempts": 5,
            "private_unavailable_unique_urls": 3,
            "private_unavailable_by_category": [
                {"category": "event_detail_page", "count": 2},
                {"category": "group_post_events_tab", "count": 1},
            ],
            "private_unavailable_sources": [
                {
                    "source_key": "facebook_event_detail",
                    "category": "event_detail_page",
                    "unique_urls": 2,
                    "attempt_count": 4,
                    "sample_url": "https://www.facebook.com/events/909522138040202/",
                }
            ],
            "private_unavailable_occurrences": [
                {
                    "requested_url": "https://www.facebook.com/events/909522138040202/",
                    "source_key": "facebook_event_detail",
                    "category": "event_detail_page",
                    "occurrence_ts": "2026-03-26 08:44:07",
                }
            ],
        },
    )

    assert len(fake_db.fb_triage_calls) == 1
    assert len(fake_db.fb_occurrence_calls) == 1
    assert fake_db.fb_triage_calls[0]["run_id"] == "run-fb"
    assert fake_db.fb_triage_calls[0]["blocked_reason"] == "private_or_unavailable_content"
    assert fake_db.fb_occurrence_calls[0]["rows"][0]["requested_url"] == "https://www.facebook.com/events/909522138040202/"
    metric_keys = [call["metric_key"] for call in fake_db.calls]
    assert "fb_private_unavailable_attempts" in metric_keys
    assert "fb_private_unavailable_unique_urls" in metric_keys
    assert "fb_private_unavailable_event_detail_page_unique_urls" in metric_keys
    assert "fb_private_unavailable_group_post_events_tab_unique_urls" in metric_keys


def test_build_top_trend_dashboard_includes_fb_blocked_content_trend() -> None:
    runner = _build_runner()

    def _history(metric_key, days=180):  # type: ignore[no-untyped-def]
        mapping = {
            "fb_private_unavailable_attempts": [{"timestamp": "2026-03-25T10:00:00", "value": 5.0}],
            "fb_private_unavailable_unique_urls": [{"timestamp": "2026-03-25T10:00:00", "value": 3.0}],
            "fb_private_unavailable_event_detail_page_unique_urls": [{"timestamp": "2026-03-25T10:00:00", "value": 2.0}],
            "fb_private_unavailable_group_post_events_tab_unique_urls": [{"timestamp": "2026-03-25T10:00:00", "value": 1.0}],
        }
        return mapping.get(metric_key, [])

    runner._load_metric_history = _history  # type: ignore[method-assign]

    html = runner._build_top_trend_dashboard_html(runtime_summary={})

    assert "Facebook Blocked Content Trend" in html
    assert "Blocked Attempts" in html
    assert "Unique Blocked URLs" in html


def test_phase2_coverage_summary_uses_watchlist_and_manual_audit() -> None:
    runner = _build_runner()

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
    runner._summarize_manual_coverage_audit = lambda: {
        "available": True,
        "manual_audit_source": "manual_coverage_audit_csv",
        "sample_size": 2,
        "expected_present_count": 2,
        "captured_count": 1,
        "missed_count": 1,
        "missed_event_rate_manual_audit_pct": 50.0,
        "sample_missed_events": [{"event_name": "Missed Event"}],
    }
    runner._summarize_new_source_discovery = lambda watchlist_rows: {
        "available": True,
        "new_source_discovery_count": 2,
        "sample_new_source_domains": ["new.example.org", "other.example.org"],
        "watchlist_domains_total": 3,
        "error": "",
    }
    coverage_summary = runner._summarize_coverage_watchlist()
    assert coverage_summary["source_hit_rate_pct"] == 66.67
    assert coverage_summary["event_capture_rate_pct"] == 33.33
    assert coverage_summary["priority_source_hit_rate_pct"] == 50.0
    assert coverage_summary["missed_event_rate_manual_audit_pct"] == 50.0
    assert coverage_summary["watchlist_source"] == "test_watchlist"
    assert coverage_summary["new_source_discovery_count"] == 2


def test_manual_coverage_audit_summary() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db
    runner._load_manual_coverage_audit_rows = lambda: (
        [
            {
                "source_name": "Audit A",
                "source_url": "https://example.com/a",
                "event_name": "Friday Social",
                "start_date": "2026-04-10",
                "expected_present": True,
                "active": True,
            },
            {
                "source_name": "Audit B",
                "source_url": "https://example.com/b",
                "event_name": "Saturday Salsa",
                "start_date": "2026-04-11",
                "expected_present": True,
                "active": True,
            },
        ],
        "manual_coverage_audit_csv",
    )

    def execute_query(query, params=None):
        if params and params.get("source_url") == "https://example.com/a":
            return [(1,)]
        return [(0,)]

    fake_db.execute_query = execute_query
    summary = runner._summarize_manual_coverage_audit()
    assert summary["available"] is True
    assert summary["captured_count"] == 1
    assert summary["missed_count"] == 1
    assert summary["missed_event_rate_manual_audit_pct"] == 50.0


def test_load_coverage_watchlist_rows_uses_repo_root_not_cwd(tmp_path, monkeypatch) -> None:
    runner = _build_runner()
    watchlists_dir = tmp_path / "data" / "watchlists"
    watchlists_dir.mkdir(parents=True)
    (watchlists_dir / "coverage_watchlist.csv").write_text(
        "source_name,source_url,source_type,priority,expected_frequency,coverage_region,active\n"
        "Test Source,https://example.com,venue_site,high,weekly,Greater Victoria,true\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(validation_test_runner, "repo_root", str(tmp_path))
    monkeypatch.chdir("/tmp")

    rows, source_label = runner._load_coverage_watchlist_rows()

    assert source_label == "coverage_watchlist_csv"
    assert len(rows) == 1
    assert rows[0]["source_url"] == "https://example.com"


def test_load_manual_coverage_audit_rows_uses_repo_root_not_cwd(tmp_path, monkeypatch) -> None:
    runner = _build_runner()
    evaluation_dir = tmp_path / "data" / "evaluation"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "manual_coverage_audit.csv").write_text(
        "source_name,source_url,event_name,start_date,expected_present,active,notes\n"
        "Audit Source,https://example.com,Friday Social,2026-04-10,true,true,ok\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(validation_test_runner, "repo_root", str(tmp_path))
    monkeypatch.chdir("/tmp")

    rows, source_label = runner._load_manual_coverage_audit_rows()

    assert source_label == "manual_coverage_audit_csv"
    assert len(rows) == 1
    assert rows[0]["event_name"] == "Friday Social"


def test_new_source_discovery_summary() -> None:
    runner = _build_runner()

    class DiscoveryDb:
        def execute_query(self, query, params=None):
            return [
                ("https://watch.example.org/event-1",),
                ("https://new.example.org/event-2",),
                ("https://www.other.example.org/event-3",),
            ]

    runner.db_handler = DiscoveryDb()
    summary = runner._summarize_new_source_discovery(
        [
            {"source_url": "https://watch.example.org/calendar"},
        ]
    )
    assert summary["available"] is True
    assert summary["new_source_discovery_count"] == 2
    assert summary["sample_new_source_domains"] == ["new.example.org", "other.example.org"]


def test_field_accuracy_summary() -> None:
    runner = _build_runner()
    class AddressDb:
        def lookup_raw_location(self, location):
            return None
        def quick_address_lookup(self, location):
            return 101 if location == "hall a" else 202 if location == "hall c" else None
    runner.db_handler = AddressDb()
    accuracy_replay = {
        "rows": [
            {
                "baseline": {"start_date": "2026-04-10", "start_time": "19:00:00", "location": "Hall A", "source": "source-a", "address_id": 101, "dance_style": "salsa, bachata", "description": "A great social dance night."},
                "replay": {"start_date": "2026-04-10", "start_time": "19:00:00", "location": "hall a", "source": "source-a", "dance_style": "bachata / salsa", "description": "A great social dance night."},
            },
            {
                "baseline": {"start_date": "2026-04-11", "start_time": "20:00:00", "location": "Hall B", "source": "source-b", "address_id": 201, "dance_style": "west coast swing", "description": "Weekly workshop and social."},
                "replay": {"start_date": "2026-04-12", "start_time": "20:30:00", "location": "hall c", "source": "source-c", "dance_style": "salsa", "description": "Different content entirely."},
            },
            {
                "baseline": {"start_date": "2026-04-13", "start_time": "", "location": "", "source": "unknown", "address_id": None, "dance_style": "", "description": ""},
                "replay": {"start_date": "2026-04-13", "start_time": "", "location": "", "source": "unknown", "dance_style": "", "description": ""},
            },
        ]
    }
    summary = runner._summarize_field_accuracy(accuracy_replay)
    assert summary["available"] is True
    assert summary["date_pct"] == 66.67
    assert summary["time_pct"] == 50.0
    assert summary["location_pct"] == 50.0
    assert summary["source_pct"] == 50.0
    assert summary["address_id_pct"] == 50.0
    assert summary["dance_style_pct"] == 50.0
    assert summary["description_pct"] == 50.0


def test_event_data_quality_summary() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    fake_db.query_map = {
        "SELECT COUNT(*) FROM events": [(40,)],
        "COALESCE(NULLIF(TRIM(event_name), ''), '') = ''": [(2,)],
        "COALESCE(end_date, start_date) < CURRENT_DATE": [(3,)],
    }
    runner.db_handler = fake_db

    summary = runner._summarize_event_data_quality()
    assert summary["available"] is True
    assert summary["invalid_event_rate_pct"] == 5.0
    assert summary["stale_event_rate_pct"] == 7.5


def test_domain_evaluation_and_codex_review_bundle() -> None:
    runner = _build_runner()
    accuracy_replay = {
        "rows": [
            {"is_match": True, "mismatch_category": "", "baseline": {"url": "https://a.example.org/1"}},
            {"is_match": False, "mismatch_category": "wrong_time", "baseline": {"url": "https://a.example.org/2"}},
            {"is_match": True, "mismatch_category": "", "baseline": {"url": "https://b.example.org/3"}},
            {"is_match": False, "mismatch_category": "wrong_date", "baseline": {"url": "https://b.example.org/3"}},
            {"is_match": False, "mismatch_category": "wrong_location", "baseline": {"url": "https://c.example.org/4"}},
        ]
    }
    domain_summary = runner._summarize_domain_evaluation(accuracy_replay)
    assert domain_summary["available"] is True
    assert domain_summary["domain_count"] == 3
    assert domain_summary["total_urls"] == 4
    assert domain_summary["top_domains"][0]["domain"] == "a.example.org"
    assert domain_summary["top_domains"][0]["replay_url_accuracy_pct"] == 50.0
    assert domain_summary["worst_domains"][0]["domain"] == "c.example.org"

    run_scorecard = {
        "run_id": "run-123",
        "overall_score": {"status": "FAIL"},
        "guardrails": {"status": "FAIL", "violations": [{"detail": "Guardrail fail"}]},
        "recommendations_input": {"top_regressions": ["Guardrail fail"]},
    }
    recommendation_plan = {
        "plan_version": "v1",
        "top_issues": [{"issue_type": "guardrail_violation"}],
    }
    bundle = runner._build_codex_review_bundle(
        run_scorecard=run_scorecard,
        accuracy_replay_summary=accuracy_replay,
        classifier_performance_summary={"status": "OK"},
        coverage_summary={"source_hit_rate_pct": 80.0},
        dev_summary={"replay_url_accuracy_pct": 66.0},
        holdout_summary={"replay_url_accuracy_pct": 70.0},
        domain_capped_summary={"replay_url_accuracy_pct": 75.0},
        domain_evaluation_summary=domain_summary,
        recommendation_plan=recommendation_plan,
    )
    assert bundle["bundle_version"] == "v1"
    assert bundle["run_id"] == "run-123"
    assert bundle["artifacts"]["domain_evaluation_summary"]["domain_count"] == 3
    assert bundle["artifacts"]["dev_summary"]["replay_url_accuracy_pct"] == 66.0
    assert bundle["artifacts"]["recommendation_plan"]["plan_version"] == "v1"


def test_persist_validation_artifacts_makes_payload_json_safe() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    runner.db_handler = fake_db

    runner._persist_validation_artifacts(
        "run-safe",
        {
            "codex_review_bundle": {
                "generated_on": date(2026, 4, 6),
                "generated_at": datetime(2026, 4, 6, 23, 35, 54),
                "window_end_time": time(23, 35, 54),
                "nested": {"items": [date(2026, 4, 7)]},
            }
        },
    )

    assert len(fake_db.artifact_calls) == 1
    payload = fake_db.artifact_calls[0]["artifact_payload"]
    assert payload["generated_on"] == "2026-04-06"
    assert payload["generated_at"].startswith("2026-04-06T23:35:54")
    assert payload["window_end_time"] == "23:35:54"
    assert payload["nested"]["items"] == ["2026-04-07"]


def test_phase3_holdout_domain_caps_and_guardrails() -> None:
    runner = _build_runner()
    _stub_code_version(runner)
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

    original_load_dev_urls = validation_test_runner.load_dev_urls
    validation_test_runner.load_dev_urls = lambda: {
        "https://example.org/a",
        "https://another.org/c",
    }
    try:
        dev_summary = runner._summarize_dev_replay(accuracy_replay)
    finally:
        validation_test_runner.load_dev_urls = original_load_dev_urls
    assert dev_summary["available"] is True
    assert dev_summary["replay_urls_seen"] == 2
    assert dev_summary["replay_url_accuracy_pct"] == 100.0

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
        chatbot_quality_summary={"summary": {"chatbot_response_within_15s_pct": 85.0, "chatbot_answer_correctness_pct": 80.0, "chatbot_user_visible_error_rate_pct": 5.0}},
        classifier_performance_summary={},
        event_data_quality_summary={"invalid_event_rate_pct": 2.0, "stale_event_rate_pct": 6.0, "total_events": 50},
        field_accuracy_summary={"date_pct": 70.0, "time_pct": 60.0, "location_pct": 50.0, "source_pct": 80.0},
        coverage_summary={"source_hit_rate_pct": 55.0, "event_capture_rate_pct": 40.0, "watchlist_source": "test"},
        scraper_telemetry_summary={},
        dev_summary=dev_summary,
        holdout_summary=holdout_summary,
        domain_capped_summary=domain_capped_summary,
    )
    guardrails = runner._evaluate_phase3_guardrails(run_scorecard)
    assert guardrails["status"] == "FAIL"
    assert {item["metric_key"] for item in guardrails["violations"]} == {
        "database_accuracy_min_pct",
        "holdout_replay_url_accuracy_min_pct",
        "coverage_watchlist_source_hit_rate_min_pct",
        "events_coverage_min_pct",
        "stale_event_rate_max_pct",
        "chatbot_response_within_15s_min_pct",
        "chatbot_answer_correctness_min_pct",
        "chatbot_user_visible_error_rate_max_pct",
    }


def test_phase3_guardrails_fail_when_telemetry_integrity_fails() -> None:
    runner = _build_runner()
    run_scorecard = {
        "kpis": {
            "database_accuracy": {"summary": {"replay_url_accuracy_pct": 90.0, "stale_event_rate_pct": 1.0}},
            "events_coverage": {"summary": {"watchlist_source_hit_rate_pct": 90.0, "watchlist_event_capture_rate_pct": 90.0}},
            "chatbot_quality": {"summary": {"summary": {"chatbot_response_within_15s_pct": 95.0, "chatbot_answer_correctness_pct": 90.0, "chatbot_user_visible_error_rate_pct": 1.0}}},
        },
        "evaluation_scope": {"holdout_summary": {"replay_url_accuracy_pct": 90.0}},
        "telemetry_integrity": {"status": "FAIL", "violations": ["step_mismatch:scraper"]},
    }

    guardrails = runner._evaluate_phase3_guardrails(run_scorecard)

    assert guardrails["status"] == "FAIL"
    assert any(item["metric_key"] == "phase1_telemetry_integrity" for item in guardrails["violations"])


def test_run_delta_summary_compares_previous_run_and_holdout_baseline() -> None:
    runner = _build_runner()
    fake_db = _FakeDbHandler()
    fake_db.query_map["FROM validation_run_artifacts"] = [
        (
            "run-122",
            """{
                "run_id": "run-122",
                "run_timestamp_utc": "2026-03-17T12:00:00",
                "overall_score": {"value": 80.0},
                "evaluation_scope": {"dev_summary": {"replay_url_accuracy_pct": 72.0}, "holdout_summary": {"replay_url_accuracy_pct": 74.0}},
                "kpis": {
                    "database_accuracy": {"summary": {"replay_url_accuracy_pct": 81.0}},
                    "events_coverage": {"summary": {"watchlist_source_hit_rate_pct": 70.0, "watchlist_event_capture_rate_pct": 66.0, "missed_event_rate_manual_audit_pct": 18.0}},
                    "run_time": {"summary": {"pipeline_duration_minutes": 180.0, "run_processed_url_count": 9, "run_inserted_event_count": 36}},
                    "run_costs": {"summary": {"summary": {"total_usd": 5.0}}},
                    "chatbot_quality": {"summary": {"summary": {"chatbot_response_within_15s_pct": 85.0, "chatbot_answer_correctness_pct": 82.0}}}
                }
            }""",
            datetime(2026, 3, 17, 12, 0, 0),
        )
    ]
    runner.db_handler = fake_db
    current_run_scorecard = {
        "run_id": "run-123",
        "overall_score": {"value": 84.5},
        "evaluation_scope": {"dev_summary": {"replay_url_accuracy_pct": 76.0}, "holdout_summary": {"replay_url_accuracy_pct": 78.0}},
        "kpis": {
            "database_accuracy": {"summary": {"replay_url_accuracy_pct": 84.0}},
            "events_coverage": {"summary": {"watchlist_source_hit_rate_pct": 75.0, "watchlist_event_capture_rate_pct": 68.0, "missed_event_rate_manual_audit_pct": 12.0}},
            "run_time": {"summary": {"pipeline_duration_minutes": 160.0, "run_processed_url_count": 12, "run_inserted_event_count": 40}},
            "run_costs": {"summary": {"summary": {"total_usd": 4.0}}},
            "chatbot_quality": {"summary": {"summary": {"chatbot_response_within_15s_pct": 90.0, "chatbot_answer_correctness_pct": 88.0}}},
        },
    }

    run_delta_summary = runner._build_run_delta_summary(current_run_scorecard)

    assert run_delta_summary["available"] is True
    assert run_delta_summary["previous_run"]["baseline_run_id"] == "run-122"
    assert run_delta_summary["previous_run"]["summary"]["delta_overall_score"] == 4.5
    assert run_delta_summary["holdout_baseline"]["summary"]["delta_holdout_replay_url_accuracy_pct"] == 4.0
    assert any(item["metric_key"] == "pipeline_duration_minutes" and item["direction"] == "improved" for item in run_delta_summary["previous_run"]["metric_deltas"])


def test_recommendation_plan_uses_existing_scorecard_signals() -> None:
    runner = _build_runner()
    run_scorecard = {
        "run_id": "run-plan",
        "overall_score": {"status": "FAIL"},
        "guardrails": {
            "status": "FAIL",
            "violations": [
                {"detail": "Telemetry integrity guardrail failed"},
                {"detail": "Chatbot correctness below minimum"},
            ],
        },
    }
    recommendation_plan = runner._build_recommendation_plan(
        run_scorecard=run_scorecard,
        classifier_performance_summary={
            "stage_details": [{"stage": "ml", "replay_url_accuracy_pct": 60.0, "replay_url_count": 8}],
        },
        coverage_summary={
            "missed_sources": [{"source_name": "Watchlist Venue", "source_url": "https://venue.example.org"}],
        },
        dev_summary={"replay_url_accuracy_pct": 65.0, "replay_urls_seen": 4, "matched_urls": 2, "mismatched_urls": 2},
        runtime_summary={
            "step_spans": [{"log_file": "images_log.txt", "duration_minutes": 80.0}],
        },
        llm_cost_summary={
            "by_provider": {"openai_usd": 4.2, "openrouter_usd": 1.1},
        },
        domain_evaluation_summary={
            "worst_domains": [{"domain": "bad.example.org", "replay_url_accuracy_pct": 25.0}],
        },
    )
    assert recommendation_plan["plan_version"] == "v1"
    assert recommendation_plan["summary"]["has_guardrail_failure"] is True
    assert len(recommendation_plan["top_issues"]) == 3
    assert recommendation_plan["top_issues"][0]["issue_type"] == "guardrail_violation"
    assert recommendation_plan["top_issues"][0]["title"] == "Chatbot correctness below minimum"
    assert any(
        "chatbot_evaluation_report.json" in action
        for action in recommendation_plan["top_issues"][0]["recommended_actions"]
    )
    assert all(
        item["title"] != "Telemetry integrity guardrail failed"
        for item in recommendation_plan["top_issues"]
    )
    assert any(
        item["issue_type"] == "runtime_bottleneck"
        and any("image_date_resolution_drop_summary" in action for action in item["recommended_actions"])
        for item in recommendation_plan["top_issues"]
    )


def test_build_parser_improvement_workflow_html_surfaces_ordered_fix_plan() -> None:
    runner = _build_runner()
    accuracy_replay_summary = {
        "rows": [
            {
                "is_match": False,
                "mismatch_category": "wrong_date",
                "mismatch_details": "core field mismatch",
                "baseline": {
                    "event_name": "Sunday Blues Services",
                    "url": "https://www.debrhymerband.com/shows",
                },
                "replay": {
                    "event_name": "Blues Jam",
                    "url": "https://www.debrhymerband.com/shows",
                },
            },
            {
                "is_match": False,
                "mismatch_category": "wrong_replay_event_selection",
                "mismatch_details": "listing page chose wrong event",
                "baseline": {
                    "event_name": "Sunday Blues Services",
                    "url": "https://livevictoria.com/calendar/music",
                },
                "replay": {
                    "event_name": "Flamenco Tablao",
                    "url": "https://livevictoria.com/calendar/music",
                },
            },
        ]
    }
    recommendation_plan = {
        "top_issues": [
            {"issue_type": "domain_regression", "title": "Investigate replay regression on livevictoria.com"},
        ]
    }
    action_queue = {
        "items": [
            {
                "title": "Fix replay matching for listing pages",
                "reason": "replay rows are pairing the wrong event from scraper output",
                "suggested_change": "tighten parser/replay matcher",
                "acceptance_test": "replay row matches on event_name and time",
            }
        ]
    }
    domain_evaluation_summary = {
        "worst_domains": [
            {"domain": "livevictoria.com"},
            {"domain": "www.debrhymerband.com"},
        ]
    }

    html = runner._build_parser_improvement_workflow_html(
        accuracy_replay_summary=accuracy_replay_summary,
        recommendation_plan=recommendation_plan,
        action_queue=action_queue,
        domain_evaluation_summary=domain_evaluation_summary,
    )

    assert "Recommended first self-improving subsystem" in html
    assert "Ordered Parser Workflow" in html
    assert "Work parser/replay mismatches before runtime or cost tuning." in html
    assert "python tests/validation/test_runner.py || true" in html
    assert "Top Parser Mismatch Categories" in html
    assert "Current queued actions" not in html
    assert "wrong_replay_event_selection" in html
    assert "https://livevictoria.com/calendar/music" in html
    assert "Likely Fix Location" in html
    assert "Replay matcher in tests/validation/test_runner.py" in html


def test_classifier_performance_summary_includes_stage_domain_details() -> None:
    runner = _build_runner()

    class PerfDb:
        def __init__(self) -> None:
            self.metric_calls: list[dict] = []

        def execute_query(self, query, params=None):
            return [
                {
                    "link": "https://a.example.org/1",
                    "archetype": "simple_page",
                    "classification_stage": "ml",
                    "classification_confidence": 0.8,
                    "classification_owner_step": "scraper.py",
                    "classification_subtype": "detail",
                    "replay_row_count": 2,
                    "replay_true_count": 2,
                    "replay_url_success": True,
                },
                {
                    "link": "https://a.example.org/2",
                    "archetype": "simple_page",
                    "classification_stage": "ml",
                    "classification_confidence": 0.6,
                    "classification_owner_step": "scraper.py",
                    "classification_subtype": "detail",
                    "replay_row_count": 1,
                    "replay_true_count": 0,
                    "replay_url_success": False,
                },
                {
                    "link": "https://b.example.org/3",
                    "archetype": "simple_page",
                    "classification_stage": "rule",
                    "classification_confidence": 0.9,
                    "classification_owner_step": "scraper.py",
                    "classification_subtype": "detail",
                    "replay_row_count": 1,
                    "replay_true_count": 1,
                    "replay_url_success": True,
                },
            ]

        def record_metric_observation(self, **kwargs):
            self.metric_calls.append(kwargs)

    runner.db_handler = PerfDb()
    summary = runner._summarize_classifier_performance("run-1")
    assert summary["status"] == "OK"
    assert any(item["stage"] == "ml" and item["domain"] == "a.example.org" and item["replay_url_accuracy_pct"] == 50.0 for item in summary["stage_domain_details"])


def test_accuracy_replay_rows_html_excludes_email_like_rows() -> None:
    runner = _build_runner()
    replay_data = {
        "coverage_accuracy_pct": 100.0,
        "replay_accuracy_pct": 100.0,
        "total_rows": 2,
        "true_count": 2,
        "false_count": 0,
        "rows": [
            {
                "is_match": True,
                "baseline": {
                    "event_name": "Email Event",
                    "url": "djdancingdean@39679421.mailchimpapp.com",
                },
                "replay": {
                    "event_name": "Email Event",
                    "url": "djdancingdean@39679421.mailchimpapp.com",
                },
                "mismatch_category": "",
                "mismatch_details": "",
                "action_taken": "",
            },
            {
                "is_match": True,
                "baseline": {
                    "event_name": "Web Event",
                    "url": "https://example.org/event",
                },
                "replay": {
                    "event_name": "Web Event",
                    "url": "https://example.org/event",
                },
                "mismatch_category": "",
                "mismatch_details": "",
                "action_taken": "",
            },
        ],
    }
    html = runner._build_accuracy_replay_rows_html(replay_data)
    assert "Email Event" not in html
    assert "Web Event" in html
    full_html = runner._build_accuracy_replay_html(replay_data)
    assert "2/2" in full_html
    assert "False Rows" in full_html


def test_accuracy_replay_assessment_excludes_email_rows_before_sampling() -> None:
    runner = _build_runner()
    runner.config = {
        "testing": {
            "validation": {
                "accuracy_replay": {
                    "enabled": True,
                    "query_text": "test",
                    "max_events": 20,
                    "strict_time_match": True,
                    "trend_days": 30,
                }
            }
        }
    }

    class FakeExecutor:
        def __init__(self, config, db_handler) -> None:
            pass

        def execute_test_question(self, question_dict):
            return {
                "execution_success": True,
                "sql_query": "SELECT * FROM fake",
            }

    class FakeDb:
        def execute_query(self, query):
            rows = []
            for idx in range(1, 22):
                url = f"https://example.org/event-{idx}"
                if idx == 5:
                    url = "djdancingdean@39679421.mailchimpapp.com"
                rows.append(
                    {
                        "event_name": f"Event {idx}",
                        "start_date": "2026-04-01",
                        "start_time": "19:00:00",
                        "source": "source",
                        "location": "Hall",
                        "url": url,
                    }
                )
            return rows

        def record_accuracy_replay_result(self, **kwargs):
            return None

        def record_metric_observation(self, **kwargs):
            return None

    runner.db_handler = FakeDb()
    runner._load_metric_history = lambda metric_key, days=90: []
    runner._infer_latest_pipeline_run_id = lambda timestamp: "run-1"
    runner._fetch_replay_events_for_url = lambda url: {"ok": True, "category": "", "details": "", "events": []}
    runner._compare_replay_row = lambda baseline_row, replay_payload, strict_time_match: {
        "is_match": True,
        "category": "match",
        "details": "",
        "baseline": baseline_row,
        "replay": baseline_row,
    }
    runner._resolve_baseline_event_id = lambda baseline: None

    original_executor = validation_test_runner.ChatbotTestExecutor
    validation_test_runner.ChatbotTestExecutor = FakeExecutor
    try:
        summary = runner._run_accuracy_replay_assessment({"timestamp": "2026-03-18T12:00:00"})
    finally:
        validation_test_runner.ChatbotTestExecutor = original_executor

    assert summary["total_rows"] == 20
    assert summary["true_count"] == 20
    assert all("mailchimpapp.com" not in str((row.get("baseline") or {}).get("url", "")) for row in summary["rows"])


def test_fetch_replay_events_for_url_uses_rd_ext_for_listing_pages(monkeypatch) -> None:
    runner = _build_runner()
    runner.config_path = "config/config.yaml"

    class FakeLlm:
        def generate_prompt(self, url, text, prompt_type):
            return "prompt", "schema"

        def query_llm(self, url, prompt, schema_type=None):
            return '{"events":[{"event_name":"Bill Francis – Story & Song","start_date":"2026-03-23","start_time":"17:30","source":"The Loft Pub Victoria","location":"The Loft Pub Victoria"}]}'

        def extract_and_parse_json(self, response, url, schema_type):
            return [
                {
                    "event_name": "Bill Francis – Story & Song",
                    "start_date": "2026-03-23",
                    "start_time": "17:30",
                    "source": "The Loft Pub Victoria",
                    "location": "The Loft Pub Victoria",
                }
            ]

    class FakeResponse:
        status_code = 200
        text = '<html><body><a href="/events/2026-03-23/">Bill Francis</a></body></html>'

    runner.llm_handler = FakeLlm()
    runner._fetch_replay_events_via_rd_ext = lambda url: {
        "ok": True,
        "category": "",
        "details": "rd_ext_replay",
        "events": [
            {
                "event_name": "Bill Francis – Story & Song",
                "start_date": "2026-03-23",
                "start_time": "17:30:00",
                "source": "The Loft Pub Victoria",
                "location": "The Loft Pub Victoria",
                "url": "https://loftpubvictoria.com/events",
                "raw": {"replay_child_url": "https://loftpubvictoria.com/events/2026-03-23/"},
            }
        ],
    }

    monkeypatch.setattr(validation_test_runner.requests, "get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(
        validation_test_runner,
        "classify_page_with_confidence",
        lambda **kwargs: {"owner_step": "rd_ext.py", "archetype": "incomplete_event"},
    )

    payload = runner._fetch_replay_events_for_url("https://loftpubvictoria.com/events")

    assert payload["ok"] is True
    assert payload["details"] == "rd_ext_replay"
    assert payload["events"][0]["start_date"] == "2026-03-23"
    assert payload["events"][0]["url"] == "https://loftpubvictoria.com/events"
    assert payload["events"][0]["raw"]["replay_child_url"] == "https://loftpubvictoria.com/events/2026-03-23/"


def test_extract_replay_events_from_artifact_uses_same_rd_ext_routing() -> None:
    runner = _build_runner()
    runner.llm_handler = SimpleNamespace(
        generate_prompt=lambda url, text, prompt_type: ("prompt", "schema"),
        query_llm=lambda url, prompt, schema_type=None: "",
        extract_and_parse_json=lambda response, url, schema_type: [],
    )
    runner._fetch_replay_events_via_rd_ext = lambda url: {
        "ok": True,
        "category": "",
        "details": "rd_ext_replay",
        "events": [
            {
                "event_name": "Bill Francis – Story & Song",
                "start_date": "2026-03-23",
                "start_time": "17:30:00",
                "source": "The Loft Pub Victoria",
                "location": "The Loft Pub Victoria",
                "url": "https://loftpubvictoria.com/events",
                "raw": {"replay_child_url": "https://loftpubvictoria.com/events/2026-03-23/"},
            }
        ],
    }
    artifact = ReplayArtifact(
        source_url="https://loftpubvictoria.com/events",
        fetch_method="requests",
        artifact_type="raw_html",
        body_text='<html><body><a href="/events/2026-03-23/">Bill Francis</a></body></html>',
        links=["https://loftpubvictoria.com/events/2026-03-23/"],
    )
    original_classifier = validation_test_runner.classify_page_with_confidence
    validation_test_runner.classify_page_with_confidence = lambda **kwargs: {"owner_step": "rd_ext.py"}
    try:
        payload = runner._extract_replay_events_from_artifact(artifact)
    finally:
        validation_test_runner.classify_page_with_confidence = original_classifier

    assert payload["ok"] is True
    assert payload["details"] == "rd_ext_replay"
    assert payload["events"][0]["raw"]["replay_child_url"] == "https://loftpubvictoria.com/events/2026-03-23/"


def test_fetch_replay_events_for_eventbrite_organizer_follows_detail_links(monkeypatch) -> None:
    runner = _build_runner()
    organizer_url = "https://www.eventbrite.ca/o/silent-dj-victoria-31599471691"
    detail_url = "https://www.eventbrite.ca/e/full-pink-moon-circle-beach-dance-tickets-9876543210"

    class FakeResponse:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text
            self.status_code = 200

    def fake_get(url: str, *args, **kwargs):
        if url == organizer_url:
            return FakeResponse(
                organizer_url,
                f'<html><body><a href="{detail_url}">Full PINK MOON Circle & Beach-dance</a></body></html>',
            )
        if url == detail_url:
            return FakeResponse(
                detail_url,
                "<html><body>Full PINK MOON Circle & Beach-dance Wed, Apr 1, 7:30 PM South end of Willows Beach</body></html>",
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(validation_test_runner.requests, "get", fake_get)
    runner.llm_handler = SimpleNamespace(
        generate_prompt=lambda url, text, prompt_type: ("prompt", "event_extraction"),
        query_llm=lambda url, prompt, schema_type=None: '{"events":[]}',
        extract_and_parse_json=lambda response, url, schema_type: [
            {
                "event_name": "full pink moon circle & beach-dance",
                "start_date": "2026-04-01",
                "start_time": "19:30:00",
                "source": "eventbrite - silent dj victoria",
                "location": "willows, victoria, bc",
                "url": detail_url,
            }
        ] if url == detail_url else [],
    )

    payload = runner._fetch_replay_events_for_url(organizer_url)

    assert payload["ok"] is True
    assert payload["details"] == "eventbrite_organizer_detail_replay"
    assert payload["events"][0]["url"] == organizer_url
    assert payload["events"][0]["raw"]["replay_child_url"] == detail_url
