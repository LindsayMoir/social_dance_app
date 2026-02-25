import sys

sys.path.insert(0, "src")

import fb as fb_module
from fb import should_use_fb_checkpoint
from fb import FacebookEventScraper
from fb import canonicalize_facebook_url, is_facebook_login_redirect, is_non_content_facebook_url
from fb import sanitize_facebook_seed_urls
from fb import classify_facebook_access_state
import pandas as pd


def test_should_use_fb_checkpoint_true_only_when_enabled_and_not_render():
    cfg = {"checkpoint": {"fb_urls_cp_status": True}}
    assert should_use_fb_checkpoint(cfg, is_render=False) is True


def test_should_use_fb_checkpoint_false_on_render_or_disabled():
    cfg_enabled = {"checkpoint": {"fb_urls_cp_status": True}}
    cfg_disabled = {"checkpoint": {"fb_urls_cp_status": False}}
    assert should_use_fb_checkpoint(cfg_enabled, is_render=True) is False
    assert should_use_fb_checkpoint(cfg_disabled, is_render=False) is False


def test_canonicalize_facebook_url_unwraps_login_redirect():
    wrapped = "https://es-la.facebook.com/login/?next=https%3A%2F%2Fwww.facebook.com%2Fgroups%2Fvictoriawesties%2F"
    assert canonicalize_facebook_url(wrapped) == "https://www.facebook.com/groups/victoriawesties/"


def test_is_facebook_login_redirect_detects_login_urls():
    wrapped = "https://www.facebook.com/login/?next=%2Fgroups%2Fvictoriawesties%2F"
    plain = "https://www.facebook.com/groups/victoriawesties/"
    assert is_facebook_login_redirect(wrapped) is True
    assert is_facebook_login_redirect(plain) is False


def test_canonicalize_facebook_url_unwraps_recover_redirect():
    wrapped = "https://www.facebook.com/recover/initiate/?next=https%3A%2F%2Fwww.facebook.com%2Fgroups%2Falivetango%2F"
    assert canonicalize_facebook_url(wrapped) == "https://www.facebook.com/groups/alivetango/"


def test_is_non_content_facebook_url_flags_sharer_dialog_recover():
    assert is_non_content_facebook_url("https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fexample.com") is True
    assert is_non_content_facebook_url("https://www.facebook.com/dialog/send?app_id=1&link=https%3A%2F%2Fexample.com") is True
    assert is_non_content_facebook_url("https://www.facebook.com/recover/initiate/?next=%2Fgroups%2Fabc%2F") is True
    assert is_non_content_facebook_url("https://www.facebook.com/groups/victoriawesties/") is False


def test_sanitize_facebook_seed_urls_canonicalizes_filters_and_dedupes():
    df = pd.DataFrame(
        {
            "link": [
                "https://es-la.facebook.com/login/?next=https%3A%2F%2Fwww.facebook.com%2Fgroups%2Fvictoriawesties%2F",
                "https://www.facebook.com/groups/victoriawesties/",
                "https://www.facebook.com/dialog/send?app_id=1&link=https%3A%2F%2Fexample.com",
                "https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fexample.com",
            ],
            "parent_url": ["a", "b", "c", "d"],
            "source": ["s", "s", "s", "s"],
            "keywords": ["k", "k", "k", "k"],
        }
    )

    cleaned, stats = sanitize_facebook_seed_urls(df)
    assert cleaned["link"].tolist() == ["https://www.facebook.com/groups/victoriawesties/"]
    assert stats["input_rows"] == 4
    assert stats["output_rows"] == 1
    assert stats["canonicalized_rows"] >= 1
    assert stats["non_content_rows_dropped"] == 2
    assert stats["duplicate_rows_dropped"] == 1


def test_classify_facebook_access_state_detects_login():
    state = classify_facebook_access_state(
        "https://www.facebook.com/login/?next=%2Fevents%2F123%2F",
        "Log in to Facebook",
    )
    assert state == "login"


def test_classify_facebook_access_state_detects_blocked():
    state = classify_facebook_access_state(
        "https://www.facebook.com/events/123/",
        "You are temporarily blocked from using this feature. Try again later.",
    )
    assert state == "blocked"


def test_classify_facebook_access_state_detects_ok():
    state = classify_facebook_access_state(
        "https://www.facebook.com/events/123/",
        "Event details and comments",
    )
    assert state == "ok"


def test_classify_facebook_access_state_does_not_false_positive_on_generic_login_text():
    state = classify_facebook_access_state(
        "https://www.facebook.com/groups/examplegroup/",
        "Some footer text saying log in to facebook to comment",
    )
    assert state == "ok"


def test_driver_fb_urls_marks_base_relevant_when_child_events_yield(monkeypatch):
    class DummyDB:
        def __init__(self):
            self.conn = object()
            self.rows = []

        def should_process_url(self, _url):
            return True

        def write_url_to_db(self, row):
            self.rows.append(row)

    dummy_db = DummyDB()
    monkeypatch.setattr(fb_module, "db_handler", dummy_db, raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(
        fb_module.pd,
        "read_sql",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "link": "https://www.facebook.com/groups/1634269246863069/",
                    "parent_url": "",
                    "source": "Victoria Kizomba Lovers Group",
                    "keywords": "kizomba",
                }
            ]
        ),
    )

    scraper = FacebookEventScraper.__new__(FacebookEventScraper)
    scraper.config = {"crawling": {"urls_run_limit": 10}, "checkpoint": {"fb_urls_write_every": 10}}
    scraper.urls_visited = set()
    scraper.events_written_to_db = 0
    scraper._ensure_authenticated_or_raise = lambda: None
    scraper.normalize_facebook_url = lambda u: u
    scraper.extract_event_links = lambda _u: ["https://www.facebook.com/events/1018836737121988/"]

    def process_fb_url(url, _parent_url, _source, _keywords):
        if "/events/" in url:
            scraper.events_written_to_db += 1

    scraper.process_fb_url = process_fb_url

    scraper.driver_fb_urls()

    assert any(
        row[0] == "https://www.facebook.com/groups/1634269246863069/" and row[4] is True
        for row in dummy_db.rows
    )


def test_driver_fb_urls_does_not_mark_base_relevant_when_no_events_yield(monkeypatch):
    class DummyDB:
        def __init__(self):
            self.conn = object()
            self.rows = []

        def should_process_url(self, _url):
            return True

        def write_url_to_db(self, row):
            self.rows.append(row)

    dummy_db = DummyDB()
    monkeypatch.setattr(fb_module, "db_handler", dummy_db, raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(
        fb_module.pd,
        "read_sql",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "link": "https://www.facebook.com/groups/1634269246863069/",
                    "parent_url": "",
                    "source": "Victoria Kizomba Lovers Group",
                    "keywords": "kizomba",
                }
            ]
        ),
    )

    scraper = FacebookEventScraper.__new__(FacebookEventScraper)
    scraper.config = {"crawling": {"urls_run_limit": 10}, "checkpoint": {"fb_urls_write_every": 10}}
    scraper.urls_visited = set()
    scraper.events_written_to_db = 0
    scraper._ensure_authenticated_or_raise = lambda: None
    scraper.normalize_facebook_url = lambda u: u
    scraper.extract_event_links = lambda _u: []
    scraper.process_fb_url = lambda *_args, **_kwargs: None

    scraper.driver_fb_urls()

    assert not any(
        row[0] == "https://www.facebook.com/groups/1634269246863069/" and row[4] is True
        for row in dummy_db.rows
    )
