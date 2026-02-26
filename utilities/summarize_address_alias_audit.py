#!/usr/bin/env python3
"""Summarize address alias audit activity from CSV output."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import yaml


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_timestamp_day(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value[:10] if len(value) >= 10 else "unknown"


def read_alias_audit_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def print_top_counter(title: str, counter: Counter, limit: int = 10) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  (none)")
        return
    for key, count in counter.most_common(limit):
        display = key if key else "(blank)"
        print(f"  {count:5d}  {display}")


def summarize_rows(rows: List[Dict[str, str]], days: Optional[int]) -> None:
    day_counts: Counter = Counter()
    rule_counts: Counter = Counter()
    decision_counts: Counter = Counter()
    rule_decision_counts: Counter = Counter()
    match_type_counts: Counter = Counter()
    candidate_counts: Counter = Counter()
    canonical_counts: Counter = Counter()
    rows_by_day: Dict[str, int] = defaultdict(int)

    sorted_rows = sorted(rows, key=lambda r: r.get("timestamp", ""))
    if days and days > 0:
        valid_days = sorted({parse_timestamp_day(r.get("timestamp", "")) for r in sorted_rows if parse_timestamp_day(r.get("timestamp", "")) != "unknown"})
        keep_days = set(valid_days[-days:]) if valid_days else set()
    else:
        keep_days = set()

    filtered_rows: List[Dict[str, str]] = []
    for row in sorted_rows:
        day = parse_timestamp_day(row.get("timestamp", ""))
        if keep_days and day not in keep_days:
            continue
        filtered_rows.append(row)

    for row in filtered_rows:
        day = parse_timestamp_day(row.get("timestamp", ""))
        decision = row.get("decision", "")
        rule_name = row.get("rule_name", "")
        match_type = row.get("match_type", "")
        candidate = row.get("candidate", "")
        canonical_id = row.get("canonical_address_id", "")

        rows_by_day[day] += 1
        day_counts.update([day])
        decision_counts.update([decision or "(blank)"])
        rule_counts.update([rule_name or "(blank)"])
        rule_decision_counts.update([f"{rule_name or '(blank)'} | {decision or '(blank)'}"])
        match_type_counts.update([match_type or "(blank)"])
        candidate_counts.update([candidate or "(blank)"])
        canonical_counts.update([canonical_id or "(blank)"])

    print("\nAddress Alias Audit Summary")
    print("=" * 34)
    print(f"Rows analyzed: {len(filtered_rows)}")
    if days and days > 0:
        print(f"Window: last {days} day(s) present in audit")
    else:
        print("Window: all available rows")

    print_top_counter("Daily Volume", day_counts, limit=30)
    print_top_counter("Decision Counts", decision_counts, limit=10)
    print_top_counter("Rule Counts", rule_counts, limit=20)
    print_top_counter("Rule + Decision", rule_decision_counts, limit=20)
    print_top_counter("Match Type", match_type_counts, limit=10)
    print_top_counter("Top Candidate Text", candidate_counts, limit=15)
    print_top_counter("Canonical Address IDs", canonical_counts, limit=15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize address alias audit CSV output.")
    parser.add_argument(
        "--config",
        default=os.path.join("config", "config.yaml"),
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional explicit path to alias audit CSV; overrides config value.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help="If set (>0), summarize only the last N days present in the audit file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = args.csv or cfg.get("output", {}).get("address_alias_audit", "output/address_alias_hits.csv")

    if not os.path.exists(csv_path):
        print(f"Alias audit file not found: {csv_path}")
        print("Run ingestion first, then rerun this utility.")
        return

    rows = read_alias_audit_rows(csv_path)
    if not rows:
        print(f"Alias audit file is empty: {csv_path}")
        return

    summarize_rows(rows, days=args.days)


if __name__ == "__main__":
    main()
