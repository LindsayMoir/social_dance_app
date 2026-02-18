import os
import sys
from typing import List

import pytest
import yaml

# Use src on path like other tests
sys.path.insert(0, 'src')

from utils.sql_filters import enforce_dance_style  # type: ignore
from db import DatabaseHandler  # type: ignore
from sqlalchemy import text  # type: ignore


@pytest.mark.integration
def test_wcs_query_runs_and_filters_results_if_any():
    # Initialize real DB handler using app's normal path
    import os
    with open(os.path.join('config','config.yaml'),'r') as f:
        cfg = yaml.safe_load(f)
    dbh = DatabaseHandler(cfg)
    engine = dbh.conn

    # Base SQL for upcoming 21 days without style
    base_sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= CURRENT_DATE AND start_date <= CURRENT_DATE + INTERVAL '21 days' "
        "ORDER BY start_date, start_time LIMIT 50"
    )

    # Enforce dance style for West Coast Swing
    user_text = "Where can I dance west coast swing this week?"
    sql = enforce_dance_style(base_sql, user_text)

    # Execute and validate
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()
    # If rows exist, ensure they match the style filter
    if rows:
        # dance_style is column index 2 per SELECT order
        styles: List[str] = [str(r[2] or '').lower() for r in rows]
        assert all(('west coast swing' in s) or ('wcs' in s) for s in styles)


@pytest.mark.integration
def test_no_style_query_runs_without_adding_style_filter():
    import os
    with open(os.path.join('config','config.yaml'),'r') as f:
        cfg = yaml.safe_load(f)
    dbh = DatabaseHandler(cfg)
    engine = dbh.conn

    base_sql = (
        "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, start_time, end_time, source, url, price, description, location "
        "FROM events WHERE start_date >= CURRENT_DATE AND start_date <= CURRENT_DATE + INTERVAL '14 days' "
        "ORDER BY start_date, start_time LIMIT 20"
    )

    user_text = "Where can I dance this week?"  # No style specified
    sql = enforce_dance_style(base_sql, user_text)

    # Ensure we did not add a style filter
    assert 'dance_style ILIKE' not in sql and 'dance_style ilike' not in sql

    # Execute and ensure it runs
    with engine.connect() as conn:
        _ = conn.execute(text(sql)).fetchall()
