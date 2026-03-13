import datetime
import sys

import pandas as pd

sys.path.insert(0, "src")

from utils.chatbot_metrics_sync_utils import (
    count_nullish_datetime_values,
    safe_db_target_label,
    sanitize_records_for_sql,
)


def test_sanitize_records_for_sql_converts_nat_and_nan_to_none():
    df = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "started_at": pd.NaT,
                "finished_at": pd.Timestamp("2026-03-10T10:00:00"),
                "duration_ms": float("nan"),
                "sql_snippet": None,
            }
        ]
    )

    records = sanitize_records_for_sql(df)
    assert len(records) == 1
    row = records[0]
    assert row["request_id"] == "r1"
    assert row["started_at"] is None
    assert isinstance(row["finished_at"], datetime.datetime)
    assert row["duration_ms"] is None
    assert row["sql_snippet"] is None


def test_count_nullish_datetime_values_counts_missing_entries():
    df = pd.DataFrame(
        [
            {"started_at": pd.NaT, "finished_at": pd.Timestamp("2026-03-10T10:00:00")},
            {"started_at": pd.Timestamp("2026-03-10T09:00:00"), "finished_at": pd.NaT},
        ]
    )
    counts = count_nullish_datetime_values(df, ["started_at", "finished_at", "created_at"])
    assert counts["started_at"] == 1
    assert counts["finished_at"] == 1
    assert counts["created_at"] == 0


def test_safe_db_target_label_redacts_credentials():
    url = "postgresql://my-host.example.com:5432/mydb"
    label = safe_db_target_label(url)
    assert label == "my-host.example.com/mydb"
