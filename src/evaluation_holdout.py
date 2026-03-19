"""Helpers for loading evaluation URL sets used by scorecards and training gates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse


DEFAULT_GOLD_HOLDOUT_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "gold_holdout_urls.csv"
DEFAULT_DEV_URLS_PATH = Path(__file__).resolve().parent.parent / "data" / "evaluation" / "dev_urls.csv"


def normalize_evaluation_url(value: Any) -> str:
    """Normalize URLs for stable train/dev/holdout comparisons."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text and "@" in text and "/" not in text and "?" not in text:
        return text.lower()
    try:
        parsed = urlparse(text)
        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").rstrip("/")
        return urlunparse((scheme, netloc, path, "", parsed.query, ""))
    except Exception:
        return text.lower().rstrip("/")


def load_evaluation_urls(path: str | Path | None) -> set[str]:
    """Load normalized URLs from an evaluation CSV, returning an empty set when unavailable."""
    csv_path = Path(path) if path else None
    if csv_path is None or not csv_path.exists():
        return set()

    urls: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            raw_url = normalize_evaluation_url(row.get("url"))
            active_raw = str(row.get("active", "True") or "True").strip().lower()
            if not raw_url or active_raw in {"false", "0", "no", "n"}:
                continue
            urls.add(raw_url)
    return urls


def load_gold_holdout_urls(path: str | Path | None = None) -> set[str]:
    """Load normalized holdout URLs from CSV, returning an empty set when unavailable."""
    return load_evaluation_urls(path or DEFAULT_GOLD_HOLDOUT_PATH)


def load_dev_urls(path: str | Path | None = None) -> set[str]:
    """Load normalized dev URLs from CSV, returning an empty set when unavailable."""
    return load_evaluation_urls(path or DEFAULT_DEV_URLS_PATH)


def is_holdout_url(url: Any, path: str | Path | None = None) -> bool:
    """Return True when the provided URL is part of the gold holdout set."""
    safe_url = normalize_evaluation_url(url)
    if not safe_url:
        return False
    return safe_url in load_gold_holdout_urls(path)


def is_dev_url(url: Any, path: str | Path | None = None) -> bool:
    """Return True when the provided URL is part of the dev evaluation set."""
    safe_url = normalize_evaluation_url(url)
    if not safe_url:
        return False
    return safe_url in load_dev_urls(path)
