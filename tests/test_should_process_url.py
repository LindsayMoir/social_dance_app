import sys
import pandas as pd

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

    def execute_query(self, *a, **k):
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
