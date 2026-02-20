import sys

sys.path.insert(0, "src")

from credential_validator import _load_facebook_group_probe_urls


def test_load_facebook_group_probe_urls_filters_and_limits(tmp_path):
    whitelist = tmp_path / "aaa_urls.csv"
    whitelist.write_text(
        "link\n"
        "https://www.facebook.com/groups/alivetango/\n"
        "https://www.facebook.com/groups/alivetango/\n"
        "https://www.facebook.com/groups/TangoVanIsle/\n"
        "https://www.facebook.com/events/123/\n",
        encoding="utf-8",
    )

    cfg = {
        "testing": {
            "validation": {
                "scraping": {
                    "whitelist_file": str(whitelist),
                }
            }
        }
    }

    urls = _load_facebook_group_probe_urls(cfg, limit=1)
    assert urls == ["https://www.facebook.com/groups/alivetango"]


def test_load_facebook_group_probe_urls_returns_empty_when_missing_file():
    cfg = {
        "testing": {
            "validation": {
                "scraping": {
                    "whitelist_file": "does/not/exist.csv",
                }
            }
        }
    }
    assert _load_facebook_group_probe_urls(cfg, limit=10) == []
