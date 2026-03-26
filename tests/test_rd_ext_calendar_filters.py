import sys

sys.path.insert(0, "src")

from rd_ext import is_calendar_export_url


def test_is_calendar_export_url_detects_ics_and_outlook_exports():
    assert is_calendar_export_url("https://example.com/events/123?ical=1") is True
    assert is_calendar_export_url("https://example.com/events/123?outlook-ical=1") is True
    assert is_calendar_export_url("https://example.com/calendar/feed/basic.ics") is True
    assert is_calendar_export_url("https://www.google.com/calendar/event?action=TEMPLATE&text=Social+Dance") is True
    assert is_calendar_export_url("https://calendar.google.com/calendar/render?cid=webcal%3A%2F%2Fexample.com%2Ffeed.ics") is True


def test_is_calendar_export_url_does_not_flag_normal_event_detail_pages():
    assert is_calendar_export_url("https://example.com/events/123") is False
    assert is_calendar_export_url("https://example.com/show/abc") is False


def test_scraper_marks_google_login_calendar_redirect_as_unsupported():
    from scraper import EventSpider

    assert EventSpider._is_unsupported_google_calendar_candidate(
        "https://accounts.google.com/ServiceLogin/identifier"
        "?continue=https://calendar.google.com/calendar/r"
    ) is True
