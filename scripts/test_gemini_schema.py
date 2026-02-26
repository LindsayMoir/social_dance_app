#!/usr/bin/env python3
"""
Smoke test Gemini schema compatibility in llm.py.

Default mode validates the generated Gemini schemas locally (no network).
Optional --live mode performs a single Gemini API request using sanitized schema.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from llm import LLMHandler  # noqa: E402


SCHEMA_TYPES: tuple[str, ...] = (
    "event_extraction",
    "address_extraction",
    "deduplication_response",
    "relevance_classification",
    "address_deduplication",
)


def _find_disallowed_key(value: Any) -> str | None:
    disallowed = {"name", "strict", "$schema", "additionalProperties"}
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in disallowed:
                return key
            bad = _find_disallowed_key(nested)
            if bad:
                return bad
        return None
    if isinstance(value, list):
        for nested in value:
            bad = _find_disallowed_key(nested)
            if bad:
                return bad
    return None


def run_local_schema_checks() -> int:
    """
    Validate local Gemini schema generation without DB/network.
    """
    handler = LLMHandler.__new__(LLMHandler)
    failed = False

    for schema_type in SCHEMA_TYPES:
        schema = handler._get_json_schema_by_type(schema_type, "gemini")
        if not isinstance(schema, dict):
            print(f"FAIL {schema_type}: expected dict schema, got {type(schema)}")
            failed = True
            continue

        top_type = schema.get("type")
        if top_type not in {"object", "array"}:
            print(f"FAIL {schema_type}: unsupported top-level type {top_type!r}")
            failed = True
            continue

        bad_key = _find_disallowed_key(schema)
        if bad_key:
            print(f"FAIL {schema_type}: found disallowed key {bad_key!r}")
            failed = True
            continue

        print(f"OK   {schema_type}")

    print("RESULT:", "PASS" if not failed else "FAIL")
    return 0 if not failed else 1


def run_live_probe(model: str, prompt: str) -> int:
    """
    Run one live Gemini schema-constrained call.
    """
    gemini_token = os.getenv("GEMINI_API_KEY")
    if not gemini_token:
        print("SKIP live probe: GEMINI_API_KEY is not set.")
        return 0

    handler = LLMHandler.__new__(LLMHandler)
    handler.gemini_token = gemini_token
    handler.gemini_timeout_seconds = 60

    try:
        response = handler.query_gemini(prompt=prompt, model=model, schema_type="event_extraction")
    except Exception as exc:
        print(f"LIVE_FAIL: {exc}")
        return 1

    if not response:
        print("LIVE_EMPTY: request succeeded but returned no content.")
        return 1

    print("LIVE_OK")
    print(response[:600])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Gemini schema compatibility.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also run one live Gemini API call using event_extraction schema.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model for --live probe.",
    )
    parser.add_argument(
        "--prompt",
        default="Return valid JSON with one dance event only.",
        help="Prompt used by --live probe.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    local_status = run_local_schema_checks()
    if local_status != 0:
        return local_status

    if not args.live:
        return 0

    return run_live_probe(model=args.model, prompt=args.prompt)


if __name__ == "__main__":
    raise SystemExit(main())
