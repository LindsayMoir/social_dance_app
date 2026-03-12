import sys
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, 'src')
from db import DatabaseHandler


class DummyDB(DatabaseHandler):
    def __init__(self):
        cfg = {
            'input': {'urls': 'data/urls'},
            'constants': {'black_list_domains': 'data/other/black_list_domains.csv'},
            'clean_up': {'old_events': 3},
        }
        # Minimal init without heavy DB connections
        self.config = cfg
        self.blacklisted_domains = set()
        self.urls_df = pd.DataFrame(columns=['link','parent_url','source','keywords','relevant','crawl_try','time_stamp'])
        self.urls_gb = pd.DataFrame(columns=['link','hit_ratio','crawl_try'])
        self._whitelist_set = set()
        self._should_process_decision_counters = Counter()
        self._history_start_dates = {}

    def execute_query(self, query, params=None):
        if "FROM events_history" in str(query):
            key = (params or {}).get("url")
            if key in self._history_start_dates:
                return [(self._history_start_dates[key],)]
        return []


def test_whitelist_prefix_match_subpath():
    db = DummyDB()
    db._whitelist_set = {db._normalize_for_compare('https://latindancecanada.com/')}
    assert db.is_whitelisted_url('https://latindancecanada.com/events/calendar') is True
    assert db.should_process_url('https://latindancecanada.com/events/calendar') is True


def test_whitelist_exact_match():
    db = DummyDB()
    db._whitelist_set = {db._normalize_for_compare('https://vlda.ca/resources/')}
    assert db.is_whitelisted_url('https://vlda.ca/resources/') is True
    assert db.should_process_url('https://vlda.ca/resources/') is True


def test_non_whitelisted_uses_history():
    db = DummyDB()
    assert db.should_process_url('https://example.com/page') is True


def test_should_skip_stale_facebook_event_detail_url_when_already_seen():
    db = DummyDB()
    url = "https://www.facebook.com/events/1234567890123456/"
    db.urls_df = pd.DataFrame(
        [{"link": db.normalize_url(url), "parent_url": "", "source": "fb", "keywords": [], "relevant": False, "crawl_try": 1, "time_stamp": datetime.now()}]
    )
    db.urls_gb = pd.DataFrame(
        [{"link": db.normalize_url(url), "hit_ratio": 0.5, "crawl_try": 1}]
    )
    db._history_start_dates[db.normalize_url(url)] = datetime.now().date() - timedelta(days=2)

    assert db.should_process_url(url) is False
    counts = db.get_should_process_decision_counts()
    assert counts.get("skip_stale_facebook_event_detail", 0) >= 1


def test_should_skip_stale_eventbrite_event_detail_url_when_already_seen():
    db = DummyDB()
    url = "https://www.eventbrite.ca/e/latin-vibes-party-tickets-1982755373697?aff=ebdssbdestsearch"
    norm_url = db.normalize_url(url)
    db.urls_df = pd.DataFrame(
        [{"link": norm_url, "parent_url": "", "source": "ebs", "keywords": [], "relevant": False, "crawl_try": 1, "time_stamp": datetime.now()}]
    )
    db.urls_gb = pd.DataFrame(
        [{"link": norm_url, "hit_ratio": 0.5, "crawl_try": 1}]
    )
    db._history_start_dates[norm_url] = datetime.now().date() - timedelta(days=3)

    assert db.should_process_url(url) is False
    counts = db.get_should_process_decision_counts()
    assert counts.get("skip_stale_eventbrite_event_detail", 0) >= 1


def test_should_not_apply_stale_static_skip_to_non_static_page_urls():
    db = DummyDB()
    url = "https://example.com/events"
    norm_url = db.normalize_url(url)
    db.urls_df = pd.DataFrame(
        [{"link": norm_url, "parent_url": "", "source": "site", "keywords": [], "relevant": False, "crawl_try": 1, "time_stamp": datetime.now()}]
    )
    db.urls_gb = pd.DataFrame(
        [{"link": norm_url, "hit_ratio": 0.5, "crawl_try": 1}]
    )
    db._history_start_dates[norm_url] = datetime.now().date() - timedelta(days=10)

    # Non-static pages should continue through existing history rules.
    assert db.should_process_url(url) is True
