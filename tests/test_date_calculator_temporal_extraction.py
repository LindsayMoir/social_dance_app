import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from date_calculator import calculate_date_range, extract_temporal_phrase
from date_calculator import resolve_temporal_from_text


def test_extract_prefers_explicit_date_over_weekday_phrase() -> None:
    phrase = extract_temporal_phrase("Next Wednesday. That would be March 18, 2026.")
    assert phrase == "march 18, 2026"


def test_next_weekday_uses_next_week_semantics() -> None:
    result = calculate_date_range("next wednesday", "2026-03-09")
    assert result["start_date"] == "2026-03-18"
    assert result["end_date"] == "2026-03-18"


def test_explicit_month_day_date_parses() -> None:
    result = calculate_date_range("march 18, 2026", "2026-03-09")
    assert result["start_date"] == "2026-03-18"
    assert result["end_date"] == "2026-03-18"


def test_explicit_day_month_date_parses() -> None:
    result = calculate_date_range("18 march 2026", "2026-03-09")
    assert result["start_date"] == "2026-03-18"
    assert result["end_date"] == "2026-03-18"


def test_before_wrapper_with_explicit_date() -> None:
    result = calculate_date_range("before march 18, 2026", "2026-03-09")
    assert result["end_date"] == "2026-03-17"


def test_extract_first_week_of_month_phrase() -> None:
    phrase = extract_temporal_phrase("Where can I dance the first week of April?")
    assert phrase == "the first week of april"


def test_first_week_of_month_range() -> None:
    result = calculate_date_range("first week of april", "2026-03-09")
    assert result["start_date"] == "2026-04-01"
    assert result["end_date"] == "2026-04-07"


def test_first_week_of_next_month_range() -> None:
    result = calculate_date_range("first week of next month", "2026-03-09")
    assert result["start_date"] == "2026-04-01"
    assert result["end_date"] == "2026-04-07"


def test_between_date_range_expression() -> None:
    result = calculate_date_range("between march 3, 2026 and march 9, 2026", "2026-03-01")
    assert result["start_date"] == "2026-03-03"
    assert result["end_date"] == "2026-03-09"


def test_resolve_temporal_from_full_text() -> None:
    result = resolve_temporal_from_text("Where can I dance the first week of next month?", "2026-03-09")
    assert result is not None
    assert result["temporal_phrase"] == "the first week of next month"
    assert result["start_date"] == "2026-04-01"
    assert result["end_date"] == "2026-04-07"


def test_extract_upcoming_week_phrase() -> None:
    phrase = extract_temporal_phrase("What social dances are happening in the upcoming week?")
    assert phrase == "upcoming week"


def test_upcoming_week_range_is_forward_looking() -> None:
    result = calculate_date_range("upcoming week", "2026-03-09")
    assert result["start_date"] == "2026-03-09"
    assert result["end_date"] == "2026-03-15"


def test_extract_upcoming_30_days_phrase() -> None:
    phrase = extract_temporal_phrase("Show me upcoming 30 days of dance events.")
    assert phrase == "upcoming 30 days"


def test_upcoming_30_days_range_is_forward_looking() -> None:
    result = calculate_date_range("upcoming 30 days", "2026-03-09")
    assert result["start_date"] == "2026-03-09"
    assert result["end_date"] == "2026-04-07"


def test_upcoming_events_defaults_to_next_30_days() -> None:
    result = calculate_date_range("upcoming events", "2026-03-09")
    assert result["start_date"] == "2026-03-09"
    assert result["end_date"] == "2026-04-07"


def test_live_bands_no_longer_raises_and_defaults_forward() -> None:
    result = calculate_date_range("live bands", "2026-03-09")
    assert result["start_date"] == "2026-03-09"
    assert result["end_date"] == "2026-04-07"
