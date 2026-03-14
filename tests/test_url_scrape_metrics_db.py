from datetime import datetime

import pandas as pd
import sys

sys.path.insert(0, "src")
from db import DatabaseHandler


def test_ensure_url_scrape_metrics_table_executes_expected_queries() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    queries: list[str] = []

    def _exec(query: str, params=None):
        queries.append(query)
        return []

    handler.execute_query = _exec  # type: ignore[attr-defined]

    handler.ensure_url_scrape_metrics_table()

    joined = "\n".join(queries)
    assert "CREATE TABLE IF NOT EXISTS url_scrape_metrics" in joined
    assert "idx_url_scrape_metrics_run_id" in joined
    assert "idx_url_scrape_metrics_link" in joined
    assert "idx_url_scrape_metrics_time_stamp" in joined


def test_write_url_scrape_metric_writes_single_row(monkeypatch) -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.conn = object()

    captured: dict = {}

    def _fake_to_sql(self, name, con, if_exists, index):  # type: ignore[no-untyped-def]
        captured["name"] = name
        captured["con"] = con
        captured["if_exists"] = if_exists
        captured["index"] = index
        captured["df"] = self.copy()

    monkeypatch.setattr(pd.DataFrame, "to_sql", _fake_to_sql, raising=True)

    handler.write_url_scrape_metric(
        {
            "run_id": "run-123",
            "step_name": "scraper",
            "link": "https://example.com/events",
            "parent_url": "https://example.com",
            "source": "Example Source",
            "keywords": ["dance", "salsa"],
            "archetype": "simple_page",
            "extraction_attempted": True,
            "extraction_succeeded": True,
            "extraction_skipped": False,
            "decision_reason": "llm_positive",
            "links_discovered": 5,
            "links_followed": 2,
            "time_stamp": datetime(2026, 3, 13, 12, 0, 0),
        }
    )

    assert captured["name"] == "url_scrape_metrics"
    assert captured["if_exists"] == "append"
    assert captured["index"] is False

    df = captured["df"]
    assert len(df) == 1
    row = df.iloc[0].to_dict()
    assert row["run_id"] == "run-123"
    assert row["step_name"] == "scraper"
    assert row["archetype"] == "simple_page"
    assert row["decision_reason"] == "llm_positive"
    assert row["keywords"] == "dance, salsa"
    assert row["links_discovered"] == 5
    assert row["links_followed"] == 2
