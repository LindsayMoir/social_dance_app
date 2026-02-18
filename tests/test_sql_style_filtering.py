import re

import sys
sys.path.insert(0, 'src')
from utils.sql_filters import enforce_dance_style


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def test_adds_west_coast_swing_filter_when_requested():
    user = "Where can I dance west coast swing this week?"
    sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= '2026-02-16' AND start_date <= '2026-02-22' "
        "ORDER BY start_date, start_time LIMIT 30"
    )
    out = enforce_dance_style(sql, user)
    assert "dance_style ilike '%west coast swing%'" in out.lower()
    assert "dance_style ilike '%wcs%'" in out.lower()
    # Ensure insertion before ORDER BY
    assert out.lower().find("dance_style ilike") < out.lower().find("order by")


def test_does_not_add_style_when_user_did_not_specify():
    user = "Where can I dance this week?"
    sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= '2026-02-16' AND start_date <= '2026-02-22' "
        "ORDER BY start_date, start_time LIMIT 30"
    )
    out = enforce_dance_style(sql, user)
    assert out == sql  # unchanged


def test_idempotent_when_sql_already_has_style():
    user = "Where can I dance west coast swing this week?"
    sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= '2026-02-16' AND start_date <= '2026-02-22' "
        "AND (dance_style ilike '%west coast swing%' OR dance_style ilike '%wcs%') "
        "ORDER BY start_date, start_time LIMIT 30"
    )
    out = enforce_dance_style(sql, user)
    # Should not duplicate or modify existing style group
    assert out == sql


def test_multiple_styles_or_group():
    user = "Show me salsa or bachata this weekend"
    sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= '2026-02-20' AND start_date <= '2026-02-22' "
        "ORDER BY start_date, start_time LIMIT 30"
    )
    out = enforce_dance_style(sql, user)
    low = out.lower()
    assert "dance_style ilike '%salsa%'" in low
    assert "dance_style ilike '%bachata%'" in low
    # Ensure both in the same OR group
    assert "( dance_style ilike '%salsa%' or dance_style ilike '%bachata%' )" in low.replace("  ", " ")


def test_abbreviation_ecs_and_full_name():
    user = "ecs next week"
    sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= '2026-02-23' AND start_date <= '2026-03-01' "
        "ORDER BY start_date, start_time LIMIT 30"
    )
    out = enforce_dance_style(sql, user)
    low = out.lower()
    assert "dance_style ilike '%east coast swing%'" in low
    assert "dance_style ilike '%ecs%'" in low
