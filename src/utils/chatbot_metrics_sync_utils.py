"""Helpers for chatbot metrics Render->local sync hygiene and diagnostics."""

from __future__ import annotations

import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd


def sanitize_records_for_sql(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows into SQL-safe records (None for NaN/NaT)."""
    if df.empty:
        return []

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        cleaned_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                cleaned_row[key] = None if pd.isna(value) else value.to_pydatetime()
                continue
            try:
                if pd.isna(value):
                    cleaned_row[key] = None
                    continue
            except Exception:
                pass
            cleaned_row[key] = value
        records.append(cleaned_row)
    return records


def count_nullish_datetime_values(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Count NaN/NaT-style values for selected datetime columns."""
    if df.empty:
        return {col: 0 for col in columns}

    result: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            result[col] = 0
            continue
        result[col] = int(df[col].isna().sum())
    return result


def safe_db_target_label(db_url: Optional[str]) -> str:
    """Return a redacted host/db label for logging."""
    if not db_url:
        return "missing"
    try:
        parsed = urlparse(db_url)
        host = (parsed.hostname or "unknown-host").strip()
        db_name = (parsed.path or "/").lstrip("/") or "unknown-db"
        return f"{host}/{db_name}"
    except Exception:
        return "unparseable"


def utc_now_iso_seconds() -> str:
    """UTC now formatted for log lines."""
    return datetime.datetime.utcnow().isoformat(timespec="seconds")
