import sys

sys.path.insert(0, "src")

from fb import should_use_fb_checkpoint
from fb import canonicalize_facebook_url, is_facebook_login_redirect, is_non_content_facebook_url
from fb import sanitize_facebook_seed_urls
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
