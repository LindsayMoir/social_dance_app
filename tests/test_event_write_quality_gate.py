import sys

import pandas as pd


sys.path.insert(0, "src")

from db import DatabaseHandler


def test_event_write_quality_gate_drops_explicitly_nonlocal_social_image_event() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Salsa Weekend",
                "start_date": "2026-05-15",
                "start_time": "20:00",
                "location": "Warsaw, Poland",
                "description": "SocialDanceTV salsa party in Poland.",
                "source": "instagram_keyword_search",
                "url": "https://www.instagram.com/p/example/#image=one",
            }
        ]
    )

    filtered, reason = handler._filter_event_rows_for_write_quality(
        df,
        source="instagram_keyword_search",
        url="https://www.instagram.com/p/example/#image=one",
        parent_url="https://www.instagram.com/p/example/",
        context="test",
    )

    assert filtered.empty
    assert reason == "out_of_scope_location"


def test_event_write_quality_gate_drops_social_image_without_locality_signal() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Dance Conference",
                "start_date": "2026-06-10",
                "start_time": "19:00",
                "location": "Grand Ballroom",
                "description": "Kizomba, semba, and urban kiz workshops.",
                "source": "instagram_keyword_search",
                "url": "https://www.instagram.com/p/example/#image=two",
            }
        ]
    )

    filtered, reason = handler._filter_event_rows_for_write_quality(
        df,
        source="instagram_keyword_search",
        url="https://www.instagram.com/p/example/#image=two",
        parent_url="https://www.instagram.com/p/example/",
        context="test",
    )

    assert filtered.empty
    assert reason == "social_image_missing_locality"


def test_event_write_quality_gate_keeps_social_image_with_local_venue_signal() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Bachata Classes",
                "start_date": "2026-05-07",
                "start_time": "19:00",
                "location": "The Loft Pub, Victoria, BC",
                "description": "Beginner bachata class.",
                "source": "instagram_keyword_search",
                "url": "https://www.instagram.com/p/example/#image=three",
            }
        ]
    )

    filtered, reason = handler._filter_event_rows_for_write_quality(
        df,
        source="instagram_keyword_search",
        url="https://www.instagram.com/p/example/#image=three",
        parent_url="https://www.instagram.com/p/example/",
        context="test",
    )

    assert len(filtered) == 1
    assert reason is None


def test_event_write_quality_gate_keeps_social_image_with_metro_vancouver_signal() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "BC Swing Dance Social",
                "start_date": "2026-08-29",
                "start_time": "20:00",
                "location": "Bonsor Recreation Complex, Burnaby, BC",
                "description": "West Coast Swing social dance.",
                "source": "instagram_keyword_search",
                "url": "https://www.instagram.com/p/example/#image=burnaby",
            }
        ]
    )

    filtered, reason = handler._filter_event_rows_for_write_quality(
        df,
        source="instagram_keyword_search",
        url="https://www.instagram.com/p/example/#image=burnaby",
        parent_url="https://www.instagram.com/p/example/",
        context="test",
    )

    assert len(filtered) == 1
    assert reason is None


def test_event_write_quality_gate_drops_rows_without_minimum_event_data() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_name": "Dance",
                "start_date": "",
                "location": "",
                "description": "Dance keywords but no actual event details.",
                "source": "instagram_keyword_search",
                "url": "https://www.instagram.com/p/example/#image=four",
            }
        ]
    )

    filtered, reason = handler._filter_event_rows_for_write_quality(
        df,
        source="instagram_keyword_search",
        url="https://www.instagram.com/p/example/#image=four",
        parent_url="https://www.instagram.com/p/example/",
        context="test",
    )

    assert filtered.empty
    assert reason == "insufficient_event_data"
