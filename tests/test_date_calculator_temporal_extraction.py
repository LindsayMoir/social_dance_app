import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from date_calculator import calculate_date_range, extract_temporal_phrase


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
