"""Unit tests for the live browser-extraction comparison runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "compare_browser_extraction.py"
SPEC = importlib.util.spec_from_file_location("compare_browser_extraction", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def test_load_targets_requires_five_urls_for_every_platform(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.csv"
    urls_file.write_text(
        "platform,url,note\nfacebook,https://www.facebook.com/events/1/,one\n",
        encoding="utf-8",
    )

    try:
        comparison.load_targets(urls_file, pages_per_platform=1)
    except ValueError as exc:
        assert "instagram, eventbrite" in str(exc)
    else:
        raise AssertionError("Expected missing platform validation")


def test_load_targets_caps_each_platform_independently(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.csv"
    urls_file.write_text(
        "platform,url,note\n"
        "facebook,https://www.facebook.com/events/1/,first\n"
        "facebook,https://www.facebook.com/events/2/,second\n"
        "instagram,https://www.instagram.com/p/one/,first\n"
        "eventbrite,https://www.eventbrite.com/e/one,first\n",
        encoding="utf-8",
    )

    targets = comparison.load_targets(urls_file, pages_per_platform=1)

    assert [(target.platform, target.note) for target in targets] == [
        ("facebook", "first"),
        ("instagram", "first"),
        ("eventbrite", "first"),
    ]


def test_load_targets_can_select_only_instagram(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.csv"
    urls_file.write_text(
        "platform,url,note\n"
        "facebook,https://www.facebook.com/events/1/,facebook\n"
        "instagram,https://www.instagram.com/p/one/,instagram\n",
        encoding="utf-8",
    )

    targets = comparison.load_targets(urls_file, pages_per_platform=1, platforms=("instagram",))

    assert [(target.platform, target.note) for target in targets] == [("instagram", "instagram")]


def test_dom_text_from_html_matches_instagram_script_removal() -> None:
    html = "<html><body><script>ignore()</script><style>.x {}</style><p>Dance tonight</p></body></html>"

    assert comparison.dom_text_from_html("instagram", html) == "Dance tonight"
    assert comparison.dom_text_from_html("facebook", html) == "Dance tonight"


def test_detect_access_state_flags_platform_shells() -> None:
    assert comparison.detect_access_state("facebook", "You must log in to continue.") == "login_required"
    assert (
        comparison.detect_access_state("facebook", "Continue with saved account")
        == "login_dialog_obscuring_content"
    )
    assert (
        comparison.detect_access_state("instagram", "This content is unavailable")
        == "content_unavailable"
    )
    assert comparison.detect_access_state("eventbrite", "Salsa social at 8 PM") == "event_content_visible"


def test_detect_access_state_flags_facebook_device_login() -> None:
    assert comparison.detect_access_state("facebook", "device-based/login") == "login_required"
