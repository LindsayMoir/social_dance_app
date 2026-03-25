import sys

sys.path.insert(0, "src")

from rd_ext import is_calendar_export_url


def test_is_calendar_export_url_detects_ics_and_outlook_exports():
    assert is_calendar_export_url("https://example.com/events/123?ical=1") is True
    assert is_calendar_export_url("https://example.com/events/123?outlook-ical=1") is True
    assert is_calendar_export_url("https://example.com/calendar/feed/basic.ics") is True


def test_is_calendar_export_url_does_not_flag_normal_event_detail_pages():
    assert is_calendar_export_url("https://example.com/events/123") is False
    assert is_calendar_export_url("https://example.com/show/abc") is False
