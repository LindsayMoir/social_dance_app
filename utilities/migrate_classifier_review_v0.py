#!/usr/bin/env python3
"""Migrate a legacy URL archetype classifier review CSV into the richer schema."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classifier_training_promoter import migrate_legacy_classifier_review_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate a legacy url_archetype_ml_classifier_review_v0.csv file into the richer schema.",
    )
    parser.add_argument(
        "--source",
        default=str(REPO_ROOT / "output" / "codex_review" / "url_archetype_ml_classifier_review_v0.csv"),
        help="Path to the legacy v0 classifier review CSV.",
    )
    parser.add_argument(
        "--target",
        default=str(REPO_ROOT / "output" / "codex_review" / "url_archetype_ml_classifier_review_v0_migrated.csv"),
        help="Path for the migrated richer-schema CSV.",
    )
    args = parser.parse_args()

    result = migrate_legacy_classifier_review_csv(
        source_path=args.source,
        target_path=args.target,
    )
    print(result["target_path"])
    print(result["rows_written"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
