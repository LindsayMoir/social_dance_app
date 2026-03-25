import pandas as pd
import sys

sys.path.insert(0, "src")

from db import DatabaseHandler


def test_extract_us_postal_code_requires_us_signal() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)

    assert handler.extract_us_postal_code("2162 Taylor Rd, Penryn, CA 95663") == "95663"
    assert handler.extract_us_postal_code("Austin, TX 78701") == "78701"
    assert handler.extract_us_postal_code("Victoria, BC V8V 2A2") is None
    assert handler.extract_us_postal_code("12345 attendees expected") is None


def test_drop_us_postal_code_events_filters_us_rows_only() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {"event_name": "US Row", "location": "2162 Taylor Rd, Penryn, CA 95663"},
            {"event_name": "CA Row", "location": "759 Yates St, Victoria, BC V8W 1L8"},
        ]
    )

    filtered = handler._drop_us_postal_code_events(df, context="test")

    assert filtered["event_name"].tolist() == ["CA Row"]
