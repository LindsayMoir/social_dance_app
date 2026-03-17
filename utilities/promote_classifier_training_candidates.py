#!/usr/bin/env python3
"""
Promote auto-positive classifier queue candidates into the training CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classifier_training_promoter import load_queue_summary, promote_training_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote URL-level classifier queue candidates into training CSV.")
    parser.add_argument(
        "--queue",
        default=os.path.join("output", "classifier_training_queue.json"),
        help="Path to classifier_training_queue.json",
    )
    parser.add_argument(
        "--training-csv",
        default=os.path.join("ml", "training_data", "original_td.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--max-promotions",
        type=int,
        default=10,
        help="Maximum number of candidates to append in one run",
    )
    parser.add_argument(
        "--max-per-domain-per-archetype",
        type=int,
        default=3,
        help="Cap promotions per domain/archetype combination",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_summary = load_queue_summary(args.queue)
    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=args.training_csv,
        max_promotions=args.max_promotions,
        max_per_domain_per_archetype=args.max_per_domain_per_archetype,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
