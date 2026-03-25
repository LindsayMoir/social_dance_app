import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import fb as fb_module
from db import DatabaseHandler
from fb import FacebookEventScraper


class _HistoryReuseDB(DatabaseHandler):
    def __init__(self, history_rows_by_url: dict[str, list[tuple]]) -> None:
        self.config = {"clean_up": {"old_events": 30}}
        self.conn = object()
        self._history_rows_by_url = history_rows_by_url

    def execute_query(self, query, params=None):
        if "FROM events_history" not in str(query):
            return []
        return self._history_rows_by_url.get((params or {}).get("url"), [])


def _history_row(url: str, start_date_value: date) -> tuple:
    return (
        501,
        401,
        "Future Event",
        "salsa",
        "Description",
        "Friday",
        start_date_value,
        start_date_value,
        "19:00:00",
        "22:00:00",
        "Source",
        "Venue",
        "$10",
        url,
        "social dance",
        7,
        datetime.now() - timedelta(days=1),
    )


def test_maybe_reuse_static_event_detail_from_history_reuses_future_event(monkeypatch) -> None:
    url = "https://www.facebook.com/events/1234567890123456/"
    db = _HistoryReuseDB({url: [_history_row(url, date.today() + timedelta(days=10))]})
    captured = {}

    def _refresh_address(event_payload):
        refreshed = dict(event_payload)
        refreshed["address_id"] = 99
        refreshed["location"] = "Resolved Venue, 123 Main St, Victoria, BC, CA"
        return refreshed

    db.process_event_address = _refresh_address

    def _capture_to_sql(self, name, con, if_exists="append", index=False, method=None):
        captured["name"] = name
        captured["row"] = self.iloc[0].to_dict()

    monkeypatch.setattr(pd.DataFrame, "to_sql", _capture_to_sql, raising=True)

    result = db.maybe_reuse_static_event_detail_from_history(url=url, rescrape_window_days=7)

    assert result["reused"] is True
    assert result["reason"] == "history_reuse_static_event_detail"
    assert result["history_kind"] == "facebook_event_detail"
    assert captured["name"] == "events"
    assert captured["row"]["url"] == url
    assert captured["row"]["address_id"] == 99
    assert captured["row"]["location"] == "Resolved Venue, 123 Main St, Victoria, BC, CA"


def test_maybe_reuse_static_event_detail_from_history_respects_rescrape_window(monkeypatch) -> None:
    url = "https://www.facebook.com/events/1234567890123456/"
    db = _HistoryReuseDB({url: [_history_row(url, date.today() + timedelta(days=3))]})
    called = {"count": 0}

    def _capture_to_sql(self, name, con, if_exists="append", index=False, method=None):
        called["count"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", _capture_to_sql, raising=True)

    result = db.maybe_reuse_static_event_detail_from_history(url=url, rescrape_window_days=7)

    assert result["reused"] is False
    assert result["reason"] == "no_upcoming_history_outside_rescrape_window"
    assert called["count"] == 0


def test_maybe_reuse_static_event_detail_from_history_rejects_us_zip_rows(monkeypatch) -> None:
    url = "https://www.facebook.com/events/1234567890123456/"
    db = _HistoryReuseDB({url: [_history_row(url, date.today() + timedelta(days=10))]})
    called = {"count": 0}

    def _refresh_address(event_payload):
        refreshed = dict(event_payload)
        refreshed["address_id"] = 42
        refreshed["location"] = "2162 Taylor Rd, Penryn, CA 95663"
        return refreshed

    def _capture_to_sql(self, name, con, if_exists="append", index=False, method=None):
        called["count"] += 1

    db.process_event_address = _refresh_address
    monkeypatch.setattr(pd.DataFrame, "to_sql", _capture_to_sql, raising=True)

    result = db.maybe_reuse_static_event_detail_from_history(url=url, rescrape_window_days=7)

    assert result["reused"] is False
    assert result["reason"] == "history_payload_disqualified"
    assert called["count"] == 0


def test_process_fb_url_reuses_history_before_navigation(monkeypatch) -> None:
    class DummyDB:
        def __init__(self) -> None:
            self.rows = []
            self.metrics = []

        def maybe_reuse_static_event_detail_from_history(self, *, url: str, rescrape_window_days: int = 7):
            return {
                "reused": True,
                "reason": "history_reuse_static_event_detail",
                "history_kind": "facebook_event_detail",
                "event_count": 1,
                "events_history_event_id": 501,
                "events_history_original_event_id": 401,
                "history_time_stamp": datetime.now().isoformat(),
                "start_date": (date.today() + timedelta(days=14)).isoformat(),
                "days_until_event": 14,
                "url": url,
            }

        def write_url_to_db(self, row):
            self.rows.append(row)

        def write_url_scrape_metric(self, metric):
            self.metrics.append(metric)

    dummy_db = DummyDB()
    monkeypatch.setattr(fb_module, "db_handler", dummy_db, raising=False)

    scraper = FacebookEventScraper.__new__(FacebookEventScraper)
    scraper.events_written_to_db = 0
    scraper.urls_visited = set()
    scraper.fb_run_abort_requested = False
    scraper.fb_run_abort_reason = ""
    scraper.normalize_facebook_url = lambda u: u

    def _should_not_navigate(_url):
        raise AssertionError("navigate_and_maybe_login should not be called when history is reused")

    scraper.navigate_and_maybe_login = _should_not_navigate

    scraper.process_fb_url(
        "https://www.facebook.com/events/1234567890123456/",
        "",
        "fb",
        "salsa",
    )

    assert scraper.events_written_to_db == 1
    assert "https://www.facebook.com/events/1234567890123456/" in scraper.urls_visited
    assert dummy_db.rows
    assert dummy_db.rows[0][7] == "history_reuse_static_event_detail"
    assert dummy_db.metrics
    assert dummy_db.metrics[0]["classification_stage"] == "history_reuse"
