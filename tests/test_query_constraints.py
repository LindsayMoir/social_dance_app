import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from query_constraints import build_sql_from_constraints, derive_constraints_from_text


def test_clarification_not_just_style_clears_specific_style() -> None:
    base = derive_constraints_from_text("show me salsa events next wednesday", "2026-03-09")
    updated = derive_constraints_from_text(
        "all dance events not just salsa",
        "2026-03-09",
        base_constraints=base,
        is_clarification=True,
    )
    assert updated["all_styles"] is True
    assert updated["include_styles"] == []
    assert updated["start_date"] == "2026-03-18"


def test_constraints_build_sql_keeps_date_and_no_style_filter_when_all_styles() -> None:
    constraints = {
        "start_date": "2026-03-18",
        "end_date": "2026-03-18",
        "all_styles": True,
        "include_styles": [],
        "exclude_styles": [],
    }
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "start_date >= '2026-03-18'" in sql_l
    assert "start_date <= '2026-03-18'" in sql_l
    assert "dance_style ilike" not in sql_l


def test_derive_constraints_first_week_of_april() -> None:
    constraints = derive_constraints_from_text("Where can I dance the first week of April?", "2026-03-09")
    assert constraints["start_date"] == "2026-04-01"
    assert constraints["end_date"] == "2026-04-07"
    assert constraints["location_terms"] == []


def test_first_week_of_april_not_used_as_location_filter() -> None:
    constraints = derive_constraints_from_text(
        "Please give me all of the dance events in the first week of april",
        "2026-03-09",
    )
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "location ilike '%first week of april%'" not in sql_l
    assert "source ilike '%first week of april%'" not in sql_l


def test_derive_constraints_first_week_of_next_month() -> None:
    constraints = derive_constraints_from_text("Where can I dance the first week of next month?", "2026-03-09")
    assert constraints["start_date"] == "2026-04-01"
    assert constraints["end_date"] == "2026-04-07"


def test_clarification_not_just_single_day_does_not_narrow_week_range() -> None:
    base = derive_constraints_from_text("Where can I dance during the first week of april", "2026-03-09")
    updated = derive_constraints_from_text(
        "The first week, not just April 1.",
        "2026-03-09",
        base_constraints=base,
        is_clarification=True,
    )
    assert updated["start_date"] == "2026-04-01"
    assert updated["end_date"] == "2026-04-07"


def test_clarification_merges_live_music_and_coda_location() -> None:
    base = derive_constraints_from_text("Please tell me what is playing at coda tonight", "2026-03-13")
    updated = derive_constraints_from_text(
        "Include live music events please at the coda.",
        "2026-03-13",
        base_constraints=base,
        is_clarification=True,
    )
    assert updated["start_date"] == "2026-03-13"
    assert updated["include_event_types"] and "live music" in updated["include_event_types"]
    assert updated["location_terms"] and any("coda" in t for t in updated["location_terms"])

    sql = build_sql_from_constraints(updated)
    assert sql is not None
    sql_l = sql.lower()
    assert "event_type ilike '%live music%'" in sql_l
    assert "location ilike '%coda%'" in sql_l or "source ilike '%coda%'" in sql_l
