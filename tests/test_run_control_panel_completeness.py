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
    )

    assert "completeness" in panel
    assert panel["completeness"]["status"] == "FAIL"
    check_names = [str(c.get("name", "")) for c in panel["completeness"]["checks"]]
    assert "Source Distribution Check Status" in check_names
    assert "Missing Required Sources" in check_names
    assert "Top-Source Trend Alerts" in check_names
    assert "Top 10 Source Concentration" in check_names

    html = runner._build_run_control_panel_html(panel)
    assert "Completeness Status" in html
    assert "Completeness KPIs" in html
