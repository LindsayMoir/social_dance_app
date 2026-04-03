import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "validation"))

from test_runner import ValidationTestRunner


def test_run_control_panel_includes_completeness_from_source_distribution(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {}

    panel = runner._summarize_run_control_panel(
        output_dir=str(tmp_path),
        run_id="test-run-id",
        reliability_scorecard={"url_level_score": 90.0, "metrics": {"chatbot_average_score": 90.0}},
        llm_activity={"pressure_level": "LOW", "pressure_reasons": []},
        llm_quality={
            "total_attempts": 10,
            "successful_urls": 5,
            "too_short_urls": 0,
            "total_urls": 5,
            "hard_failure_rate": 0.01,
        },
        scraping_results={
            "source_distribution": {
                "status": "FAIL",
                "missing_sources": ["Eventbrite"],
                "top_10_percentage": 96.0,
                "top_10_total": 120,
                "total_events": 130,
                "total_sources": 25,
                "trend_monitoring": {
                    "alerts": [
                        {"severity": "critical"},
                        {"severity": "warning"},
                    ],
                    "history_runs_used": 6,
                },
            }
        },
        fb_ig_funnel={
            "events_over_passed_rate": 0.50,
            "urls_with_events": 5,
            "urls_passed_for_scraping": 10,
        },
        scraper_network={"exception_rate": 0.10},
        pipeline_runtime={
            "runtime_hours": 2.0,
            "start_ts": "2026-03-13 01:00:00",
            "end_ts": "2026-03-13 03:00:00",
        },
        chatbot_performance={"query_latency_ms": {"p95": 10000.0}},
        openrouter_cost={"cost_usd": 1.0, "requests": 10, "tokens": 1000},
        openai_cost={"cost_usd": 0.0, "requests": 0, "tokens": 0},
        accuracy_replay={"coverage_accuracy_pct": 92.0, "true_count": 12, "total_rows": 13},
    )

    assert "completeness" in panel
    assert panel["completeness"]["status"] == "FAIL"
    check_names = [str(c.get("name", "")) for c in panel["completeness"]["checks"]]
    assert "Missing Required Sources" in check_names
    assert "Top-Source Trend Alerts" in check_names
    assert "Top 10 Source Concentration" in check_names

    html = runner._build_run_control_panel_html(panel)
    assert "Completeness Status" in html
    assert "Completeness KPIs" in html


def test_top_source_trend_alerts_are_advisory_not_fail(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {}
    runner.db_handler = None

    panel = runner._summarize_run_control_panel(
        output_dir=str(tmp_path),
        run_id="test-run-id",
        reliability_scorecard={"url_level_score": 90.0, "metrics": {"chatbot_average_score": 90.0}},
        llm_activity={"pressure_level": "LOW", "pressure_reasons": []},
        llm_quality={
            "total_attempts": 10,
            "successful_urls": 5,
            "too_short_urls": 0,
            "total_urls": 5,
            "hard_failure_rate": 0.01,
        },
        scraping_results={
            "source_distribution": {
                "status": "PASS",
                "missing_sources": [],
                "top_10_percentage": 80.0,
                "top_10_total": 120,
                "total_events": 130,
                "total_sources": 25,
                "trend_monitoring": {
                    "alerts": [
                        {"severity": "critical"},
                        {"severity": "critical"},
                    ],
                    "history_runs_used": 14,
                },
            }
        },
        fb_ig_funnel={
            "events_over_passed_rate": 0.50,
            "urls_with_events": 5,
            "urls_passed_for_scraping": 10,
        },
        scraper_network={"exception_rate": 0.10},
        pipeline_runtime={
            "runtime_hours": 2.0,
            "start_ts": "2026-03-13 01:00:00",
            "end_ts": "2026-03-13 03:00:00",
        },
        chatbot_performance={"query_latency_ms": {"p95": 10000.0}},
        openrouter_cost={"cost_usd": 1.0, "requests": 10, "tokens": 1000},
        openai_cost={"cost_usd": 0.0, "requests": 0, "tokens": 0},
        accuracy_replay={"coverage_accuracy_pct": 92.0, "true_count": 12, "total_rows": 13},
    )

    checks = panel["completeness"]["checks"]
    trend_check = next(c for c in checks if c["name"] == "Top-Source Trend Alerts")
    assert trend_check["status"] == "WARN"
    assert "advisory only" in trend_check["details"]


def test_run_control_panel_rejects_untrusted_provider_cost_totals(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {}
    runner.db_handler = None

    panel = runner._summarize_run_control_panel(
        output_dir=str(tmp_path),
        run_id="test-run-id",
        reliability_scorecard={"url_level_score": 90.0, "metrics": {"chatbot_average_score": 90.0}},
        llm_activity={"pressure_level": "LOW", "pressure_reasons": []},
        llm_quality={
            "total_attempts": 1209,
            "successful_urls": 1113,
            "too_short_urls": 0,
            "total_urls": 1169,
            "hard_failure_rate": 0.0,
        },
        scraping_results={"source_distribution": {"status": "PASS", "missing_sources": [], "top_10_percentage": 80.0, "top_10_total": 10, "total_events": 100, "trend_monitoring": {"alerts": []}}},
        fb_ig_funnel={"events_over_passed_rate": 0.5, "urls_with_events": 5, "urls_passed_for_scraping": 10},
        scraper_network={"exception_rate": 0.10},
        pipeline_runtime={"runtime_hours": 2.0, "start_ts": "2026-04-01 11:39:17", "end_ts": "2026-04-01 14:40:37"},
        chatbot_performance={"query_latency_ms": {"p95": 10000.0}},
        openrouter_cost={
            "cost_usd": 9.373552,
            "requests": 12208,
            "tokens": 36834437,
            "endpoint_used": "https://openrouter.ai/api/v1/activity",
            "cost_basis": "window_total_api_reset_detected",
        },
        openai_cost={"cost_usd": None, "requests": 8, "tokens": 63694},
        accuracy_replay={},
    )

    assert panel["simple_summary"]["cost_usd"] is None
    assert panel["simple_summary"]["openrouter_cost_usd"] is None
    openrouter_check = next(check for check in panel["cost"]["checks"] if check["name"] == "OpenRouter Run Cost (USD)")
    assert openrouter_check["actual"] == "N/A"
    assert "provider_requests=12208" in openrouter_check["details"]
