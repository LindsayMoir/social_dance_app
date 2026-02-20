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
