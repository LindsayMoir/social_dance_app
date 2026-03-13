import sys
import types
from contextlib import contextmanager

sys.path.insert(0, "src")

import credential_validator as cv
from credential_validator import _load_facebook_group_probe_urls


def test_load_facebook_group_probe_urls_filters_and_limits(tmp_path):
    whitelist = tmp_path / "aaa_urls.csv"
    gs_urls = tmp_path / "gs_urls.csv"
    whitelist.write_text(
        "link\n"
        "https://www.facebook.com/groups/alivetango/\n"
        "https://www.facebook.com/groups/alivetango/\n"
        "https://www.facebook.com/events/123/\n",
        encoding="utf-8",
    )
    gs_urls.write_text(
        "link\n"
        "https://www.facebook.com/groups/TangoVanIsle/\n"
        "https://www.facebook.com/groups/alivetango/\n",
        encoding="utf-8",
    )

    cfg = {
        "input": {"gs_urls": str(gs_urls)},
        "testing": {
            "validation": {
                "scraping": {
                    "whitelist_file": str(whitelist),
                }
            }
        }
    }

    urls = _load_facebook_group_probe_urls(cfg, limit=2)
    assert len(urls) == 2
    assert set(urls) == {
        "https://www.facebook.com/groups/alivetango",
        "https://www.facebook.com/groups/TangoVanIsle",
    }


def test_load_facebook_group_probe_urls_returns_empty_when_missing_file():
    cfg = {
        "input": {"gs_urls": "also/missing.csv"},
        "testing": {
            "validation": {
                "scraping": {
                    "whitelist_file": "does/not/exist.csv",
                }
            }
        }
    }
    assert _load_facebook_group_probe_urls(cfg, limit=10) == []


def test_validate_facebook_probe_failures_are_warning_when_not_enforced(monkeypatch):
    @contextmanager
    def _noop_headless(_headless):
        yield

    class _DummyPage:
        url = "https://www.facebook.com/"

        def goto(self, *_args, **_kwargs):
            return None

    class _DummyBrowser:
        def close(self):
            return None

    class _DummyPlaywright:
        def stop(self):
            return None

    class _DummyScraper:
        def __init__(self, *args, **kwargs):
            self.page = _DummyPage()
            self.browser = _DummyBrowser()
            self.playwright = _DummyPlaywright()

        def navigate_and_maybe_login(self, _url, max_attempts=1):
            return False

    monkeypatch.setattr(cv, "_temporary_headless_config", _noop_headless)
    monkeypatch.setattr(cv, "_load_facebook_group_probe_urls", lambda *_a, **_k: ["https://www.facebook.com/groups/a"])
    monkeypatch.setattr(
        cv,
        "config",
        {
            "testing": {
                "validation": {
                    "scraping": {
                        "facebook_group_probe_enforce": False,
                        "facebook_group_probe_retry_once": False,
                        "facebook_group_manual_review": False,
                    }
                }
            }
        },
    )
    monkeypatch.setitem(sys.modules, "fb", types.SimpleNamespace(FacebookEventScraper=_DummyScraper))

    result = cv.validate_facebook(headless=True, check_timeout_seconds=1)
    assert result["valid"] is True
    assert result["error"] is None
    assert "warning" in result


def test_validate_facebook_probe_failures_fail_when_enforced(monkeypatch):
    @contextmanager
    def _noop_headless(_headless):
        yield

    class _DummyPage:
        url = "https://www.facebook.com/"

        def goto(self, *_args, **_kwargs):
            return None

    class _DummyBrowser:
        def close(self):
            return None

    class _DummyPlaywright:
        def stop(self):
            return None

    class _DummyScraper:
        def __init__(self, *args, **kwargs):
            self.page = _DummyPage()
            self.browser = _DummyBrowser()
            self.playwright = _DummyPlaywright()

        def navigate_and_maybe_login(self, _url, max_attempts=1):
            return False

    monkeypatch.setattr(cv, "_temporary_headless_config", _noop_headless)
    monkeypatch.setattr(cv, "_load_facebook_group_probe_urls", lambda *_a, **_k: ["https://www.facebook.com/groups/a"])
    monkeypatch.setattr(
        cv,
        "config",
        {
            "testing": {
                "validation": {
                    "scraping": {
                        "facebook_group_probe_enforce": True,
                        "facebook_group_probe_retry_once": False,
                        "facebook_group_manual_review": False,
                    }
                }
            }
        },
    )
    monkeypatch.setitem(sys.modules, "fb", types.SimpleNamespace(FacebookEventScraper=_DummyScraper))

    result = cv.validate_facebook(headless=True, check_timeout_seconds=1)
    assert result["valid"] is False
    assert "Facebook group access failed" in (result["error"] or "")
