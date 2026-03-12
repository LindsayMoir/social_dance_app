import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rd_ext import validate_edge_case_social_url_ownership


def test_validate_edge_case_social_url_ownership_detects_social_rows() -> None:
    df = pd.DataFrame(
        {
            "source": ["a", "b", "c"],
            "keywords": ["dance", "dance", "dance"],
            "url": [
                "https://example.com/events",
                "https://www.facebook.com/events/123",
                "https://www.instagram.com/p/abc",
            ],
            "multiple": ["no", "no", "no"],
        }
    )
    assert validate_edge_case_social_url_ownership(df) == 2


def test_validate_edge_case_social_url_ownership_allows_non_social_rows() -> None:
    df = pd.DataFrame(
        {
            "source": ["a", "b"],
            "keywords": ["dance", "dance"],
            "url": [
                "https://example.com/events",
                "https://other.org/calendar",
            ],
            "multiple": ["no", "yes"],
        }
    )
    assert validate_edge_case_social_url_ownership(df) == 0
