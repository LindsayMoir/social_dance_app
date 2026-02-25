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
