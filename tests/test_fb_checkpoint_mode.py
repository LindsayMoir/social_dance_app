import sys

sys.path.insert(0, "src")

from fb import should_use_fb_checkpoint


def test_should_use_fb_checkpoint_true_only_when_enabled_and_not_render():
    cfg = {"checkpoint": {"fb_urls_cp_status": True}}
    assert should_use_fb_checkpoint(cfg, is_render=False) is True


def test_should_use_fb_checkpoint_false_on_render_or_disabled():
    cfg_enabled = {"checkpoint": {"fb_urls_cp_status": True}}
    cfg_disabled = {"checkpoint": {"fb_urls_cp_status": False}}
    assert should_use_fb_checkpoint(cfg_enabled, is_render=True) is False
    assert should_use_fb_checkpoint(cfg_disabled, is_render=False) is False
