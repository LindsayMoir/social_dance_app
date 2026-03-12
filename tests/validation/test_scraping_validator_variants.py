import pandas as pd
import sys

sys.path.insert(0, "tests/validation")

from scraping_validator import ScrapingValidator


class _DummyDB:
    def execute_query(self, _query, params):
        link_params = [v for k, v in params.items() if k.startswith("l")]
        # Simulate DB row existing only for trailing-slash form.
        if "https://vlda.ca/resources/" in link_params:
            return [
                ("https://vlda.ca/resources/", "Victoria Latin Dance Association", True, 1, None, "salsa"),
            ]
        return []


def test_check_scraping_failures_includes_trailing_slash_variant():
    config = {
        "testing": {
            "validation": {
                "scraping": {
                    "days_back": 7,
                    "whitelist_days_back": 60,
                    "consecutive_failures_threshold": 2,
                    "whitelist_consecutive_failures_threshold": 3,
                }
            }
        }
    }
    validator = ScrapingValidator(_DummyDB(), config)

    important_urls = pd.DataFrame(
        [
            {
                "url": "https://vlda.ca/resources/",
                "source": "Victoria Latin Dance Association",
                "importance_type": "whitelist",
                "hit_ratio": None,
            }
        ]
    )

    failures = validator.check_scraping_failures(important_urls)
    assert failures.empty


class _DummyDBChildSuccess:
    def execute_query(self, query, params):
        # Base URL query (link variants): return recent irrelevant rows.
        if "FROM urls" in query and "WHERE (" in query and "link =" in query and "SELECT link, source, relevant" in query:
            return [
                ("https://gotothecoda.com/calendar", "The Coda", False, 1, None, "swing"),
                ("https://gotothecoda.com/calendar", "The Coda", False, 1, None, "swing"),
            ]

        # Child-success query (parent_url variants): return one relevant child row.
        if "SELECT COUNT(*)" in query and "parent_url" in query and "relevant = true" in query:
            return [(1,)]

        return []


def test_check_scraping_failures_suppresses_base_failure_when_children_succeed():
    config = {
        "testing": {
            "validation": {
                "scraping": {
                    "days_back": 7,
                    "whitelist_days_back": 60,
                    "consecutive_failures_threshold": 2,
                    "whitelist_consecutive_failures_threshold": 3,
                }
            }
        }
    }
    validator = ScrapingValidator(_DummyDBChildSuccess(), config)

    important_urls = pd.DataFrame(
        [
            {
                "url": "https://gotothecoda.com/calendar",
                "source": "The Coda",
                "importance_type": "edge_case",
                "hit_ratio": None,
            }
        ]
    )

    failures = validator.check_scraping_failures(important_urls)
    assert failures.empty


def test_generate_report_categorizes_not_attempted_reasons(tmp_path):
    config = {
        "testing": {
            "validation": {
                "reporting": {
                    "output_dir": str(tmp_path),
                },
                "scraping": {
                    "days_back": 7,
                    "whitelist_days_back": 60,
                    "consecutive_failures_threshold": 2,
                    "whitelist_consecutive_failures_threshold": 3,
                },
            }
        }
    }
    validator = ScrapingValidator(_DummyDB(), config)

    log_path = tmp_path / "scraper_log.txt"
    log_path.write_text(
        "\n".join(
            [
                "2026-03-03 10:00:00 - INFO - should_process_url: URL https://skip-me.com/ does not meet criteria for processing, skipping it.",
                "2026-03-03 10:01:00 - INFO - parse(): URL run limit reached with 1 scraper-owned whitelist roots still unattempted (fb_owned=8, non_text=1); skipping non-whitelist link: https://run-limit.com/path",
                "2026-03-03 10:02:00 - INFO - {'finish_reason': 'URL run limit reached'}",
            ]
        ),
        encoding="utf-8",
    )
    validator._log_files = [str(log_path)]

    failures_df = pd.DataFrame(
        [
            {
                "url": "https://skip-me.com/",
                "source": "Skip Site",
                "failure_type": "not_attempted",
                "importance": "high_performer",
            },
            {
                "url": "https://run-limit.com/path",
                "source": "Run Limit Site",
                "failure_type": "not_attempted",
                "importance": "high_performer",
            },
            {
                "url": "https://unknown-reason.com/",
                "source": "Unknown Site",
                "failure_type": "not_attempted",
                "importance": "high_performer",
            },
        ]
    )

    report = validator.generate_report(failures_df)
    breakdown = report["summary"]["not_attempted_reason_breakdown"]
    categories = breakdown["categories"]
    context = breakdown["run_limit_whitelist_context"]

    assert breakdown["total_not_attempted"] == 3
    assert breakdown["global_url_run_limit_reached"] is True
    assert context["pending_scraper_owned_roots_max"] == 1
    assert context["fb_owned_roots_max"] == 8
    assert context["non_text_roots_max"] == 1
    assert categories["explicit_should_process_url_skip"] == 1
    assert categories["explicit_url_run_limit_skip"] == 1
    assert categories["unattributed_with_global_run_limit"] == 1
    assert categories["other_or_unknown"] == 0


def test_generate_report_excludes_prescrape_should_process_skips_from_total(tmp_path):
    config = {
        "testing": {
            "validation": {
                "reporting": {
                    "output_dir": str(tmp_path),
                },
                "scraping": {
                    "days_back": 7,
                    "whitelist_days_back": 60,
                    "consecutive_failures_threshold": 2,
                    "whitelist_consecutive_failures_threshold": 3,
                },
            }
        }
    }
    validator = ScrapingValidator(_DummyDB(), config)

    log_path = tmp_path / "scraper_log.txt"
    log_path.write_text(
        "2026-03-03 10:00:00 - INFO - should_process_url: URL https://skip-me.com/ does not meet criteria for processing, skipping it.\n",
        encoding="utf-8",
    )
    validator._log_files = [str(log_path)]

    failures_df = pd.DataFrame(
        [
            {
                "url": "https://skip-me.com/",
                "source": "Skip Site",
                "failure_type": "not_attempted",
                "importance": "high_performer",
            },
            {
                "url": "https://counted-failure.com/",
                "source": "Counted Site",
                "failure_type": "marked_irrelevant",
                "importance": "high_performer",
            },
        ]
    )

    important_urls_df = pd.DataFrame(
        [
            {"url": "https://skip-me.com/"},
            {"url": "https://counted-failure.com/"},
            {"url": "https://successful-site.com/"},
        ]
    )
    report = validator.generate_report(failures_df, important_urls_df)
    summary = report["summary"]

    assert summary["total_failures_raw"] == 2
    assert summary["pre_scrape_skipped_failures_excluded"] == 1
    assert summary["total_failures"] == 1
    assert summary["post_scrape_failures"] == 1
    assert summary["keyword_failures_after_scrape"] == 1
    assert summary["attempted_url_denominator"] == 2
    assert summary["attempted_failure_rate"] == 0.5
