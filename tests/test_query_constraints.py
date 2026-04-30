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


def test_derive_constraints_recurring_weekday_city_style_and_limit() -> None:
    constraints = derive_constraints_from_text(
        "Where can I dance on Tuesdays in Victoria for West Coast Swing? Show me 5.",
        "2026-03-09",
    )
    assert constraints["weekday_filters"] == [1]
    assert constraints["is_recurring_weekday"] is True
    assert "west coast swing" in constraints["include_styles"]
    assert "victoria" in constraints["city_terms"]
    assert "victoria" in constraints["location_terms"]
    assert constraints["limit"] == 5
    assert constraints["start_date"] == ""
    assert constraints["end_date"] == ""


def test_build_sql_from_constraints_includes_weekday_and_url_location() -> None:
    constraints = derive_constraints_from_text(
        "Where can I dance on Tuesdays in Victoria for West Coast Swing?",
        "2026-03-09",
    )
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "extract(dow from start_date) in (1)" in sql_l
    assert "dance_style ilike '%west coast swing%'" in sql_l
    assert "location ilike '%victoria%'" in sql_l
    assert "source ilike '%victoria%'" in sql_l
    assert "url ilike '%victoria%'" in sql_l
    assert "victorias" not in sql_l


def test_build_sql_from_constraints_matches_singular_venue_for_plural_query() -> None:
    constraints = derive_constraints_from_text(
        "bachata classes at Method Studios",
        "2026-03-09",
    )
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "dance_style ilike '%bachata%'" in sql_l
    assert "event_type ilike '%class%'" in sql_l
    assert "start_date >= '2026-03-09'" in sql_l
    assert "start_date <= '2026-04-07'" in sql_l
    assert "location ilike '%method studios%'" in sql_l
    assert "location ilike '%method studio%'" in sql_l


def test_concrete_venue_class_query_without_timeframe_gets_default_window() -> None:
    constraints = derive_constraints_from_text(
        "bachata classes at The Loft",
        "2026-04-30",
    )
    assert constraints["temporal_phrase"] == "upcoming 30 days"
    assert constraints["start_date"] == "2026-04-30"
    assert constraints["end_date"] == "2026-05-29"
    assert "bachata" in constraints["include_styles"]
    assert constraints["include_event_types"] == ["class"]
    assert any("loft" in term for term in constraints["location_terms"])
    assert constraints["clarification_needed"] is False

    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "start_date >= '2026-04-30'" in sql_l
    assert "start_date <= '2026-05-29'" in sql_l
    assert "dance_style ilike '%bachata%'" in sql_l
    assert "event_type ilike '%class%'" in sql_l
    assert "location ilike '%loft%'" in sql_l


def test_generic_query_without_timeframe_does_not_get_default_window() -> None:
    constraints = derive_constraints_from_text("Where can I dance?", "2026-04-30")
    assert constraints["start_date"] == ""
    assert constraints["end_date"] == ""
    assert constraints["temporal_phrase"] == ""
    assert build_sql_from_constraints(constraints) is None


def test_build_sql_from_constraints_matches_plural_venue_for_singular_query() -> None:
    constraints = derive_constraints_from_text(
        "salsa at Method Studio",
        "2026-03-09",
    )
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    sql_l = sql.lower()
    assert "location ilike '%method studio%'" in sql_l
    assert "location ilike '%method studios%'" in sql_l


def test_build_sql_from_constraints_uses_explicit_limit() -> None:
    constraints = derive_constraints_from_text(
        "Give me 7 salsa events in Victoria this week",
        "2026-03-09",
    )
    sql = build_sql_from_constraints(constraints)
    assert sql is not None
    assert sql.lower().endswith("limit 7")


def test_derive_constraints_marks_singular_weekday_as_needing_clarification() -> None:
    constraints = derive_constraints_from_text(
        "Where can I dance on Tuesday in Victoria for salsa?",
        "2026-03-09",
    )
    assert constraints["clarification_needed"] is True
    assert constraints["clarification_message"] == "Do you mean next Tuesday or every Tuesday?"
    assert "weekday_scope" in constraints["ambiguity_reasons"]


def test_refinement_also_style_broadens_existing_styles() -> None:
    base = derive_constraints_from_text("show me salsa events in Victoria this week", "2026-03-09")
    updated = derive_constraints_from_text(
        "also bachata",
        "2026-03-09",
        base_constraints=base,
    )
    assert "salsa" in updated["include_styles"]
    assert "bachata" in updated["include_styles"]
    assert updated["start_date"] == base["start_date"]
    assert "victoria" in updated["location_terms"]


def test_refinement_only_event_type_replaces_previous_event_type() -> None:
    base = derive_constraints_from_text("show me salsa social dances in Victoria", "2026-03-09")
    updated = derive_constraints_from_text(
        "only classes",
        "2026-03-09",
        base_constraints=base,
    )
    assert updated["include_event_types"] == ["class"]
    assert "salsa" in updated["include_styles"]


def test_refinement_excluding_style_preserves_other_constraints() -> None:
    base = derive_constraints_from_text("show me salsa or bachata events in Victoria this week", "2026-03-09")
    updated = derive_constraints_from_text(
        "not salsa",
        "2026-03-09",
        base_constraints=base,
    )
    assert "salsa" in updated["exclude_styles"]
    assert "salsa" not in updated["include_styles"]
    assert "bachata" in updated["include_styles"]
    assert "victoria" in updated["location_terms"]
