#!/usr/bin/env python3
"""
Backfill day_of_week from start_date for events and events_history.

This utility is intentionally non-destructive for dates:
- It never edits start_date/end_date.
- It only normalizes/replaces day_of_week when missing/invalid/inconsistent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import os
import sys
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from db_config import get_database_config

load_dotenv()


@dataclass(frozen=True)
class RepairResult:
    table_name: str
    affected_rows: int


DAY_OF_WEEK_CASE = """
CASE EXTRACT(DOW FROM start_date)::int
    WHEN 0 THEN 'Sunday'
    WHEN 1 THEN 'Monday'
    WHEN 2 THEN 'Tuesday'
    WHEN 3 THEN 'Wednesday'
    WHEN 4 THEN 'Thursday'
    WHEN 5 THEN 'Friday'
    WHEN 6 THEN 'Saturday'
END
"""


def repair_table_day_of_week(connection: Any, table_name: str) -> RepairResult:
    update_sql = f"""
        UPDATE {table_name}
        SET day_of_week = {DAY_OF_WEEK_CASE}
        WHERE start_date IS NOT NULL
          AND (
                day_of_week IS NULL
             OR TRIM(day_of_week) = ''
             OR LOWER(TRIM(day_of_week)) NOT IN (
                    'monday', 'tuesday', 'wednesday', 'thursday',
                    'friday', 'saturday', 'sunday'
                )
             OR day_of_week <> {DAY_OF_WEEK_CASE}
          );
    """
    result = connection.execute(text(update_sql))
    return RepairResult(table_name=table_name, affected_rows=result.rowcount or 0)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    conn_str, env_name = get_database_config()
    logging.info("Connecting to database target: %s", env_name)
    engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")

    with engine.begin() as connection:
        results = [
            repair_table_day_of_week(connection, "events"),
            repair_table_day_of_week(connection, "events_history"),
        ]

    total = 0
    for item in results:
        total += item.affected_rows
        logging.info(
            "Updated %d rows in %s",
            item.affected_rows,
            item.table_name,
        )
    logging.info("Done. Total updated rows: %d", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
