import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fb import sanitize_facebook_event_text_for_extraction


def test_sanitize_fb_text_trims_suggested_events_tail() -> None:
    raw = (
        "Live Music Sessions at Crafthouse Vic West Wednesday, December 24, 2025 at 7:30 PM "
        "184 Wilson Street, Victoria, BC Suggested events Saturday SBK Dance Social March 21, 2026 "
        "Saturday SBK Dance Social April 18, 2026"
    )
    cleaned = sanitize_facebook_event_text_for_extraction(raw)
    assert "Suggested events" not in cleaned
    assert "March 21, 2026" not in cleaned
    assert "December 24, 2025" in cleaned


def test_sanitize_fb_text_handles_empty_input() -> None:
    assert sanitize_facebook_event_text_for_extraction("") == ""
    assert sanitize_facebook_event_text_for_extraction(None) == ""
