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
