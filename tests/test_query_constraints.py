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


def test_derive_constraints_first_week_of_next_month() -> None:
    constraints = derive_constraints_from_text("Where can I dance the first week of next month?", "2026-03-09")
    assert constraints["start_date"] == "2026-04-01"
    assert constraints["end_date"] == "2026-04-07"
