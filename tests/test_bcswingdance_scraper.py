#!/usr/bin/env python3
"""
Integration test for scraping:
https://bcswingdance.ca/index.php/learn-to-dance/

This test:
1) Creates a temporary seed CSV containing only the target URL.
2) Temporarily rewrites config to force scraper.py to use that seed CSV.
3) Runs scraper.py.
4) Verifies that events were added for the configured source.

Run with:
  python tests/test_bcswingdance_scraper.py
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, "src")

from db import DatabaseHandler


TARGET_URL = "https://bcswingdance.ca/index.php/learn-to-dance/"
TARGET_SOURCE = "WCS Lessons, Social Dances, and Conventions"
TARGET_KEYWORDS = "west coast swing, east coast swing, wcs, swing, social dance"


def _count_source_events(db_handler: DatabaseHandler, source: str) -> int:
    query = "SELECT COUNT(*) AS c FROM events WHERE source = :source"
    rows = db_handler.execute_query(query, {"source": source})
    return int(rows[0][0]) if rows else 0


def main() -> int:
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("FAIL: config/config.yaml not found")
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        original_config = yaml.safe_load(fh)

    if not isinstance(original_config, dict):
        print("FAIL: config/config.yaml did not parse into a dictionary")
        return 1

    test_config = copy.deepcopy(original_config)
    try:
        db_handler = DatabaseHandler(test_config)
    except SQLAlchemyError as exc:
        print("FAIL: Could not connect to database for integration test.")
        print(f"Details: {exc}")
        return 1
    before_count = _count_source_events(db_handler, TARGET_SOURCE)

    temp_dir = Path(tempfile.mkdtemp(prefix="bcswingdance_seed_"))
    seed_csv = temp_dir / "seed_urls.csv"
    seed_csv.write_text(
        "source,keywords,link\n"
        f'"{TARGET_SOURCE}","{TARGET_KEYWORDS}","{TARGET_URL}"\n',
        encoding="utf-8",
    )

    # Force scraper.py to only crawl our single target URL.
    test_config.setdefault("startup", {})["use_db"] = False
    test_config.setdefault("input", {})["urls"] = str(temp_dir)
    test_config.setdefault("crawling", {})["urls_run_limit"] = 30
    test_config.setdefault("crawling", {})["max_website_urls"] = 10
    test_config.setdefault("crawling", {})["depth_limit"] = 2
    test_config.setdefault("crawling", {})["scraper_download_timeout_seconds"] = 35
    test_config.setdefault("crawling", {})["scraper_priority_download_timeout_seconds"] = 90
    test_config.setdefault("crawling", {})["scraper_retry_times"] = 1
    test_config.setdefault("crawling", {})["scraper_priority_retry_times"] = 3

    try:
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(test_config, fh, sort_keys=False)

        print("Running scraper.py against single target URL...")
        completed = subprocess.run(
            [sys.executable, "src/scraper.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print("FAIL: scraper.py exited non-zero")
            print("STDOUT (tail):")
            print("\n".join(completed.stdout.splitlines()[-40:]))
            print("STDERR (tail):")
            print("\n".join(completed.stderr.splitlines()[-40:]))
            return 1

        after_count = _count_source_events(db_handler, TARGET_SOURCE)
        delta = after_count - before_count

        print(f"Before count ({TARGET_SOURCE}): {before_count}")
        print(f"After count  ({TARGET_SOURCE}): {after_count}")
        print(f"Delta: {delta}")

        if delta <= 0:
            print(
                "FAIL: No new events were added for the target source. "
                "Check logs/scraper_log.txt for timeout/parser details."
            )
            return 1

        print("PASS: New events were added for target source.")
        return 0
    finally:
        # Restore original config regardless of success/failure.
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(original_config, fh, sort_keys=False)


if __name__ == "__main__":
    raise SystemExit(main())
