import sys

sys.path.insert(0, "src")

from fb import should_use_fb_checkpoint
from fb import canonicalize_facebook_url, is_facebook_login_redirect


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
