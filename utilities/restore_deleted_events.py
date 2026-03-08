#!/usr/bin/env python3
"""Restore deleted events from JSONL audit records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Allow importing src modules when executed from repository root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from db_config import get_database_config  # noqa: E402


EVENT_COLUMNS: Tuple[str, ...] = (
    "event_id",
    "event_name",
    "dance_style",
    "description",
    "day_of_week",
    "start_date",
    "end_date",
    "start_time",
    "end_time",
    "source",
    "location",
    "price",
    "url",
    "event_type",
    "address_id",
    "time_stamp",
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_iso8601(value: str) -> Optional[datetime]:
    text_value = (value or "").strip()
    if not text_value:
        return None
    if text_value.endswith("Z"):
        text_value = text_value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text_value)
    except ValueError:
        return None


def default_audit_path(config: Dict[str, Any]) -> str:
    return (
        config.get("output", {}).get("deleted_events_audit")
        or os.path.join("logs", "deleted_events_audit.jsonl")
    )


def read_audit_rows(audit_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(audit_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {line_number}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def filter_rows(
    rows: Iterable[Dict[str, Any]],
    event_ids: List[int],
    url_contains: str,
    deletion_source: str,
    since: Optional[datetime],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    event_id_set = set(event_ids)
    url_needle = (url_contains or "").strip().lower()
    source_needle = (deletion_source or "").strip().lower()

    for row in rows:
        event = row.get("event") or {}
        if not isinstance(event, dict):
            continue

        if event_id_set:
            row_event_id = event.get("event_id")
            try:
                row_event_id = int(row_event_id)
            except (TypeError, ValueError):
                continue
            if row_event_id not in event_id_set:
                continue

        if url_needle:
            row_url = str(event.get("url") or "").lower()
            if url_needle not in row_url:
                continue

        if source_needle:
            row_source = str(row.get("deletion_source") or "").lower()
            if source_needle != row_source:
                continue

        if since is not None:
            deleted_at = parse_iso8601(str(row.get("deleted_at_utc") or ""))
            if deleted_at is None or deleted_at < since:
                continue

        results.append(row)

    return results


def keep_latest_per_event_id(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_event_id: Dict[int, Dict[str, Any]] = {}
    no_event_id_rows: List[Dict[str, Any]] = []

    for row in rows:
        event = row.get("event") or {}
        if not isinstance(event, dict):
            continue
        try:
            event_id = int(event.get("event_id"))
        except (TypeError, ValueError):
            no_event_id_rows.append(row)
            continue

        existing = by_event_id.get(event_id)
        if existing is None:
            by_event_id[event_id] = row
            continue

        existing_ts = parse_iso8601(str(existing.get("deleted_at_utc") or ""))
        current_ts = parse_iso8601(str(row.get("deleted_at_utc") or ""))
        if existing_ts is None and current_ts is not None:
            by_event_id[event_id] = row
        elif existing_ts is not None and current_ts is not None and current_ts > existing_ts:
            by_event_id[event_id] = row

    return list(by_event_id.values()) + no_event_id_rows


def make_engine() -> Engine:
    connection_string, env_name = get_database_config()
    print(f"Target database: {env_name}")
    return create_engine(connection_string, isolation_level="AUTOCOMMIT")


def restore_rows(rows: Iterable[Dict[str, Any]], execute: bool) -> Tuple[int, int, int]:
    insert_sql = text(
        """
        INSERT INTO events (
            event_id, event_name, dance_style, description, day_of_week,
            start_date, end_date, start_time, end_time, source, location,
            price, url, event_type, address_id, time_stamp
        ) VALUES (
            :event_id, :event_name, :dance_style, :description, :day_of_week,
            :start_date, :end_date, :start_time, :end_time, :source, :location,
            :price, :url, :event_type, :address_id, :time_stamp
        )
        ON CONFLICT (event_id) DO NOTHING
        """
    )

    eligible = 0
    restored = 0
    skipped_missing_key = 0
    engine = make_engine() if execute else None

    try:
        connection = engine.connect() if engine is not None else None
        try:
            for row in rows:
                event = row.get("event") or {}
                if not isinstance(event, dict):
                    skipped_missing_key += 1
                    continue
                if event.get("event_id") in (None, ""):
                    skipped_missing_key += 1
                    continue

                params: Dict[str, Any] = {col: event.get(col) for col in EVENT_COLUMNS}
                eligible += 1

                if execute and connection is not None:
                    result = connection.execute(insert_sql, params)
                    if (result.rowcount or 0) > 0:
                        restored += 1
        finally:
            if connection is not None:
                connection.close()
    finally:
        if engine is not None:
            engine.dispose()

    return eligible, restored, skipped_missing_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore deleted events from logs/deleted_events_audit.jsonl."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("config", "config.yaml"),
        help="Path to config file (default: config/config.yaml).",
    )
    parser.add_argument(
        "--audit",
        default="",
        help="Optional explicit audit JSONL path; overrides config output.deleted_events_audit.",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        action="append",
        default=[],
        help="Event ID to restore. Repeat flag to provide multiple IDs.",
    )
    parser.add_argument(
        "--url-contains",
        default="",
        help="Only include rows where event.url contains this text (case-insensitive).",
    )
    parser.add_argument(
        "--deletion-source",
        default="",
        help="Only include rows from this exact deletion_source value.",
    )
    parser.add_argument(
        "--since",
        default="",
        help="Only include rows with deleted_at_utc >= this ISO timestamp/date.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max rows to process after filtering and dedupe (0 means no limit).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform restores. Without this flag, runs as dry-run only.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    audit_path = args.audit or default_audit_path(cfg)
    since = parse_iso8601(args.since) if args.since else None

    if args.since and since is None:
        print(f"Invalid --since value: {args.since}")
        return

    if not os.path.exists(audit_path):
        print(f"Audit file not found: {audit_path}")
        return

    rows = read_audit_rows(audit_path)
    if not rows:
        print(f"Audit file is empty: {audit_path}")
        return

    filtered = filter_rows(
        rows=rows,
        event_ids=args.event_id,
        url_contains=args.url_contains,
        deletion_source=args.deletion_source,
        since=since,
    )
    deduped = keep_latest_per_event_id(filtered)
    if args.limit and args.limit > 0:
        deduped = deduped[: args.limit]

    eligible, restored, skipped_missing_key = restore_rows(deduped, execute=args.execute)

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print("\nDeleted Event Restore Summary")
    print("=" * 30)
    print(f"Mode: {mode}")
    print(f"Audit path: {audit_path}")
    print(f"Total audit rows: {len(rows)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Rows after event_id de-dup: {len(deduped)}")
    print(f"Eligible rows: {eligible}")
    print(f"Restored rows: {restored}")
    print(f"Skipped (missing event_id/event payload): {skipped_missing_key}")
    if not args.execute:
        print("No database writes performed. Re-run with --execute to restore.")


if __name__ == "__main__":
    main()
