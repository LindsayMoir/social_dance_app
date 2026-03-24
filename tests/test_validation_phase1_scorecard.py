from __future__ import annotations

from datetime import datetime
import os
import sys
from types import SimpleNamespace

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_DIR = os.path.join(TESTS_DIR, "validation")
if VALIDATION_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_DIR)

import test_runner as validation_test_runner
from test_runner import ValidationTestRunner


class _FakeDbHandler:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.artifact_calls: list[dict] = []
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

    def execute_query(self, query, params=None):
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
            "problem_categories": [
                {"name": "Weekend Calculation", "count": 3},
                {"name": "Hallucinated Venue", "count": 1},
            ],
        },
    )
    assert chatbot_quality_summary["summary"]["chatbot_response_within_15s_pct"] == 90.0
    assert chatbot_quality_summary["summary"]["chatbot_answer_correctness_pct"] == 88.0
    assert chatbot_quality_summary["summary"]["chatbot_sql_validity_pct"] == 95.0
    assert chatbot_quality_summary["summary"]["chatbot_hallucination_rate_pct"] == 25.0
    assert chatbot_quality_summary["summary"]["chatbot_fallback_rate_pct"] == 20.0
    assert chatbot_quality_summary["summary"]["chatbot_p95_latency_seconds"] == 14.2
    assert chatbot_quality_summary["summary"]["chatbot_user_visible_error_rate_pct"] == 10.0

    dev_summary = {
        "available": True,
        "dev_version": "v1",
        "dev_urls_total": 3,
        "replay_urls_seen": 2,
        "matched_urls": 1,
        "mismatched_urls": 1,
        "replay_url_accuracy_pct": 50.0,
    }
    run_scorecard = runner._build_phase1_run_scorecard(
        run_id="run-123",
        report_timestamp="2026-03-18T12:00:00",
        accuracy_replay={"coverage_accuracy_pct": 82.5, "replay_accuracy_pct": 77.0, "total_rows": 10, "true_count": 8, "false_count": 2},
        scraping_results={"summary": {"important_urls_checked": 20, "failed_count": 3, "whitelist_failures": 1, "edge_case_failures": 0}, "source_distribution": {"status": "PASS"}},
        runtime_summary=runtime_summary,
        llm_cost_summary=llm_cost_summary,
        chatbot_quality_summary=chatbot_quality_summary,
        classifier_performance_summary={
            "total_classified_urls": 8,
            "ml_usage_pct": 25.0,
            "stage_counts": {"rule": 6, "ml": 2},
            "stage_details": [{"stage": "rule", "replay_url_accuracy_pct": 80.0}],
        },
        duplicate_summary={
            "duplicate_rate_per_100_events": 4.5,
            "severe_duplicate_rate_per_100_events": 1.5,
        },
        event_data_quality_summary={
            "invalid_event_rate_pct": 1.25,
            "stale_event_rate_pct": 2.5,
            "total_events": 40,
        },
        field_accuracy_summary={
            "date_pct": 90.0,
            "time_pct": 80.0,
            "location_pct": 70.0,
            "source_pct": 95.0,
            "address_id_pct": 85.0,
            "dance_style_pct": 75.0,
            "description_pct": 65.0,
        },
        coverage_summary={
            "source_hit_rate_pct": 80.0,
            "event_capture_rate_pct": 60.0,
            "missed_event_rate_manual_audit_pct": 20.0,
            "new_source_discovery_count": 4,
            "watchlist_source": "coverage_watchlist_csv",
        },
        dev_summary=dev_summary,
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
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["invalid_event_rate_pct"] == 1.25
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["stale_event_rate_pct"] == 2.5
    assert run_scorecard["kpis"]["database_accuracy"]["field_accuracy"]["date_pct"] == 90.0
    assert run_scorecard["kpis"]["database_accuracy"]["field_accuracy"]["time_pct"] == 80.0
    assert run_scorecard["kpis"]["database_accuracy"]["field_accuracy"]["address_id_pct"] == 85.0
    assert run_scorecard["kpis"]["database_accuracy"]["field_accuracy"]["dance_style_pct"] == 75.0
    assert run_scorecard["kpis"]["database_accuracy"]["field_accuracy"]["description_pct"] == 65.0
    assert run_scorecard["kpis"]["database_accuracy"]["summary"]["classifier_effect"]["ml_usage_pct"] == 25.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["watchlist_source_hit_rate_pct"] == 80.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["missed_event_rate_manual_audit_pct"] == 20.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["new_source_discovery_count"] == 4
    assert run_scorecard["evaluation_scope"]["uses_holdout"] is True
    assert run_scorecard["evaluation_scope"]["holdout_summary"]["replay_url_accuracy_pct"] == 78.0
    assert run_scorecard["evaluation_scope"]["uses_dev_split"] is True
    assert run_scorecard["evaluation_scope"]["dev_version"] == "v1"
    assert run_scorecard["evaluation_scope"]["dev_summary"]["replay_url_accuracy_pct"] == 50.0
    assert run_scorecard["kpis"]["events_coverage"]["summary"]["dev_replay_url_accuracy_pct"] == 50.0
    assert run_scorecard["kpis"]["run_time"]["summary"]["urls_processed_per_minute"] == 0.0667
    assert run_scorecard["kpis"]["run_costs"]["summary"]["summary"]["cost_per_processed_url_usd"] == 0.5
    assert run_scorecard["kpis"]["run_costs"]["summary"]["summary"]["cost_per_inserted_event_usd"] == 0.1
    assert run_scorecard["code_version"] == {"git_commit": "abc123", "branch": "main"}
    assert run_scorecard["scorecard_version"] == "phase4"
    assert run_scorecard["telemetry_integrity"] == {}
    assert run_scorecard["comparison_summary"] == {}
    assert "top_regressions" not in run_scorecard["recommendations_input"]
    assert run_scorecard["overall_score"]["status"] == "PRELIMINARY"


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
        duplicate_summary={
            "duplicate_rate_per_100_events": 3.2,
            "severe_duplicate_rate_per_100_events": 1.1,
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
        run_delta_summary={
            "previous_run": {"summary": {"delta_overall_score": 1.5}},
            "holdout_baseline": {"summary": {"delta_holdout_replay_url_accuracy_pct": 2.0}},
        },
    )

    metric_keys = [call["metric_key"] for call in fake_db.calls]
    assert metric_keys == [
        "phase1_pipeline_duration_minutes",
        "phase1_total_llm_cost_usd",
        "phase1_chatbot_response_within_15s_pct",
        "phase1_chatbot_answer_correctness_pct",
        "phase1_chatbot_p95_latency_seconds",
        "duplicate_rate_per_100_events",
        "severe_duplicate_rate_per_100_events",
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
        "overall_score_delta_vs_previous_run",
        "holdout_replay_url_accuracy_delta_vs_baseline",
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
    assert summary["steps"]["scraper"]["access_success_rate_pct"] == 90.0
    assert summary["steps"]["scraper"]["extraction_success_rate_pct"] == 62.5
    assert summary["steps"]["images"]["ocr_success_rate_pct"] == 75.0
    assert summary["steps"]["images"]["vision_success_rate_pct"] == 50.0
    assert summary["steps"]["images"]["fallback_usage_rate_pct"] == 40.0


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
        duplicate_summary={"duplicate_rate_per_100_events": 2.0},
        coverage_summary={"source_hit_rate_pct": 80.0},
        dev_summary={"replay_url_accuracy_pct": 66.0},
        holdout_summary={"replay_url_accuracy_pct": 70.0},
        domain_capped_summary={"replay_url_accuracy_pct": 75.0},
        domain_evaluation_summary=domain_summary,
        run_delta_summary={"previous_run": {"available": True}},
        recommendation_plan=recommendation_plan,
    )
    assert bundle["bundle_version"] == "v1"
    assert bundle["run_id"] == "run-123"
    assert bundle["artifacts"]["domain_evaluation_summary"]["domain_count"] == 3
    assert bundle["artifacts"]["dev_summary"]["replay_url_accuracy_pct"] == 66.0
    assert bundle["artifacts"]["run_delta_summary"]["previous_run"]["available"] is True
    assert bundle["artifacts"]["recommendation_plan"]["plan_version"] == "v1"


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
        duplicate_summary={"duplicate_rate_per_100_events": 3.0, "severe_duplicate_rate_per_100_events": 2.5},
        event_data_quality_summary={"invalid_event_rate_pct": 2.0, "stale_event_rate_pct": 6.0, "total_events": 50},
        field_accuracy_summary={"date_pct": 70.0, "time_pct": 60.0, "location_pct": 50.0, "source_pct": 80.0},
        coverage_summary={"source_hit_rate_pct": 55.0, "event_capture_rate_pct": 40.0, "watchlist_source": "test"},
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
        "severe_duplicate_rate_max_per_100_events",
        "stale_event_rate_max_pct",
        "chatbot_response_within_15s_min_pct",
        "chatbot_answer_correctness_min_pct",
        "chatbot_user_visible_error_rate_max_pct",
    }


def test_phase3_guardrails_fail_when_telemetry_integrity_fails() -> None:
    runner = _build_runner()
    run_scorecard = {
        "kpis": {
            "database_accuracy": {"summary": {"replay_url_accuracy_pct": 90.0, "severe_duplicate_rate_per_100_events": 0.5, "stale_event_rate_pct": 1.0}},
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
                    "database_accuracy": {"summary": {"replay_url_accuracy_pct": 81.0, "duplicate_rate_per_100_events": 5.0}},
                    "events_coverage": {"summary": {"watchlist_source_hit_rate_pct": 70.0, "watchlist_event_capture_rate_pct": 66.0, "missed_event_rate_manual_audit_pct": 18.0}},
                    "run_time": {"summary": {"pipeline_duration_minutes": 180.0, "urls_processed_per_minute": 0.05, "events_inserted_per_minute": 0.2}},
                    "run_costs": {"summary": {"summary": {"total_usd": 5.0, "cost_per_processed_url_usd": 0.4, "cost_per_inserted_event_usd": 0.1}}},
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
            "database_accuracy": {"summary": {"replay_url_accuracy_pct": 84.0, "duplicate_rate_per_100_events": 4.0}},
            "events_coverage": {"summary": {"watchlist_source_hit_rate_pct": 75.0, "watchlist_event_capture_rate_pct": 68.0, "missed_event_rate_manual_audit_pct": 12.0}},
            "run_time": {"summary": {"pipeline_duration_minutes": 160.0, "urls_processed_per_minute": 0.06, "events_inserted_per_minute": 0.25}},
            "run_costs": {"summary": {"summary": {"total_usd": 4.0, "cost_per_processed_url_usd": 0.3, "cost_per_inserted_event_usd": 0.09}}},
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
        "guardrails": {"status": "FAIL", "violations": [{"detail": "Replay URL accuracy below minimum"}]},
    }
    recommendation_plan = runner._build_recommendation_plan(
        run_scorecard=run_scorecard,
        classifier_performance_summary={
            "stage_details": [{"stage": "ml", "replay_url_accuracy_pct": 60.0, "replay_url_count": 8}],
        },
        duplicate_summary={
            "top_duplicate_domains": [{"domain": "dup.example.org", "duplicate_groups": 3}],
        },
        coverage_summary={
            "missed_sources": [{"source_name": "Watchlist Venue", "source_url": "https://venue.example.org"}],
        },
        dev_summary={"replay_url_accuracy_pct": 65.0, "replay_urls_seen": 4, "matched_urls": 2, "mismatched_urls": 2},
        runtime_summary={
            "step_spans": [{"log_file": "scraper_log.txt", "duration_minutes": 80.0}],
        },
        llm_cost_summary={
            "by_provider": {"openai_usd": 4.2, "openrouter_usd": 1.1},
        },
        domain_evaluation_summary={
            "worst_domains": [{"domain": "bad.example.org", "replay_url_accuracy_pct": 25.0}],
        },
        run_delta_summary={
            "previous_run": {"top_regressions": [{"label": "Overall Score", "delta": -3.0}]},
            "holdout_baseline": {"top_regressions": [{"label": "Holdout Replay URL Accuracy %", "delta": -5.0}]},
        },
    )
    assert recommendation_plan["plan_version"] == "v1"
    assert recommendation_plan["summary"]["has_guardrail_failure"] is True
    assert len(recommendation_plan["top_issues"]) == 3
    assert recommendation_plan["top_issues"][0]["issue_type"] == "guardrail_violation"
    assert "recommended_actions" in recommendation_plan["top_issues"][0]


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
