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


def test_sanitize_fb_text_prefers_date_near_last_title_occurrence() -> None:
    raw = (
        "(20+) Live Music Sessions at Crafthouse Vic West | Facebook "
        "Recommended events Saturday 21 March 2026 from 20:00-23:59 Saturday SBK Dance Social March 21, 2026 "
        "Categories Classics Comedy Crafts Dance "
        "24 Wednesday 24 December 2025 from 19:30-19:31 "
        "Live Music Sessions at Crafthouse Vic West #100 - 184 Wilson Street, Victoria, BC, Canada "
        "About Discussion More About Discussion Live Music Sessions at Crafthouse Vic West Details"
    )
    cleaned = sanitize_facebook_event_text_for_extraction(raw)
    assert "Wednesday 24 December 2025 from 19:30-19:31" in cleaned
    assert "Saturday 21 March 2026 from 20:00-23:59" not in cleaned


def test_sanitize_fb_text_drops_relative_today_when_absolute_header_exists() -> None:
    raw = (
        "(2) WoW Dance Co Tuesday Night WCS Dance | Facebook "
        "Today from 19:30-21:45 WoW Dance Co Tuesday Night WCS Dance "
        "Eastern Star Hall Chapters No 5 & No 17 "
        "24 Tuesday 24 March 2026 from 19:30-21:45 "
        "WoW Dance Co Tuesday Night WCS Dance "
        "Eastern Star Hall Chapters No 5 & No 17 "
        "Details Victoria, British Columbia "
    )
    cleaned = sanitize_facebook_event_text_for_extraction(raw)
    assert "Tuesday 24 March 2026 from 19:30-21:45" in cleaned
    assert "Today from 19:30-21:45" not in cleaned
