#!/usr/bin/env python3
"""
Train and persist page-classifier ML models from the labeled training CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from page_classifier_ml import train_page_classifier_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the page-classifier ML models.")
    parser.add_argument(
        "--training-csv",
        default=os.path.join("ml", "training_data", "original_td.csv"),
        help="Path to labeled training CSV",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("ml", "models", "page_classifier_models.joblib"),
        help="Path to model artifact output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = train_page_classifier_models(
        training_csv_path=args.training_csv,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
