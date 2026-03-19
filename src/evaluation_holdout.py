"""Helpers for loading the gold holdout URL set used by evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DEFAULT_GOLD_HOLDOUT_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "gold_holdout_urls.csv"


def load_gold_holdout_urls(path: str | Path | None = None) -> set[str]:
    """Load normalized holdout URLs from CSV, returning an empty set when unavailable."""
    csv_path = Path(path) if path else DEFAULT_GOLD_HOLDOUT_PATH
    if not csv_path.exists():
        return set()

    urls: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            raw_url = str(row.get("url") or "").strip()
            active_raw = str(row.get("active", "True") or "True").strip().lower()
            if not raw_url or active_raw in {"false", "0", "no", "n"}:
                continue
            urls.add(raw_url)
    return urls


def is_holdout_url(url: Any, path: str | Path | None = None) -> bool:
    """Return True when the provided URL is part of the gold holdout set."""
    safe_url = str(url or "").strip()
    if not safe_url:
        return False
    return safe_url in load_gold_holdout_urls(path)
