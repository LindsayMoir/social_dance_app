import pandas as pd
import sys

sys.path.insert(0, "src")

from db import DatabaseHandler
from scraper import EventSpider


def test_non_calendar_url_is_rejected_by_calendar_like_guard():
    assert EventSpider._is_google_calendar_like_url("https://www.youtube.com/embed/FBg_E-ePokU") is False


def test_decode_calendar_id_without_src_returns_none():
    spider = EventSpider.__new__(EventSpider)
    assert spider.decode_calendar_id("https://www.youtube.com/embed/FBg_E-ePokU") is None


def test_google_calendar_template_share_url_is_unsupported_candidate():
    assert EventSpider._is_unsupported_google_calendar_candidate(
        "https://www.google.com/calendar/event?action=TEMPLATE&text=Social+Dance"
    ) is True


def test_google_calendar_render_webcal_share_url_is_unsupported_candidate():
    assert EventSpider._is_unsupported_google_calendar_candidate(
        "https://calendar.google.com/calendar/render?cid=webcal%3A%2F%2Fexample.com%2Ffeed.ics"
    ) is True


def test_google_calendar_group_calendar_feed_remains_supported():
    spider = EventSpider.__new__(EventSpider)
    candidates = spider._expand_calendar_url_candidates(
        "https://calendar.google.com/calendar/embed?src=test%40group.calendar.google.com"
    )
    assert candidates == [
        "https://calendar.google.com/calendar/embed?src=test%40group.calendar.google.com"
    ]


def test_convert_datetime_fields_handles_missing_canonical_columns():
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame({"Start_Date": ["2026-03-01"], "Start_Time": ["19:00:00"]})

    handler._convert_datetime_fields(df)

    assert "start_date" in df.columns
    assert "end_date" in df.columns
    assert "start_time" in df.columns
    assert "end_time" in df.columns
