#!/usr/bin/env python3
"""Tests for config-driven venue alias normalization."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from db import DatabaseHandler


def _build_handler_for_alias_tests() -> DatabaseHandler:
    """Create a lightweight DatabaseHandler instance without DB initialization."""
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.config = {
        "normalization": {
            "address_aliases": [
                {
                    "name": "wicket_hall_strathcona_aliases",
                    "aliases": [
                        "The Wicket Hall",
                        "Studio 919",
                        "The Strath",
                        "Strathcona Hotel",
                        "919 Douglas St, Victoria, BC V8W 2C2, CA",
                    ],
                    "canonical": {
                        "address_id": 109,
                        "full_address": "The Wicket Hall (Strathcona Hotel), 919 Douglas St, Victoria, BC V8W 2C2, CA",
                    },
                },
                {
                    "name": "scoped_alias",
                    "aliases": ["test scoped hall"],
                    "match": {"url_contains": "example.com/scoped", "source_equals": "scoped source"},
                    "canonical": {"address_id": 999},
                },
            ]
        },
        "output": {"address_alias_audit": "output/test/address_alias_hits_test.csv"},
    }
    handler.address_aliases = DatabaseHandler._load_address_aliases(handler)
    return handler


def test_address_alias_matches_strath_variants() -> None:
    """Known Strath/Wicket aliases should resolve to canonical mapping."""
    handler = _build_handler_for_alias_tests()
    alias_match = handler._find_address_alias_match(["Live at The Strath this Friday"])
    assert alias_match is not None
    assert alias_match["canonical"]["address_id"] == 109


def test_address_alias_matches_raw_address_variant() -> None:
    """Raw street-address text should resolve to canonical mapping."""
    handler = _build_handler_for_alias_tests()
    alias_match = handler._find_address_alias_match(["919 Douglas St, Victoria, BC V8W 2C2, CA"])
    assert alias_match is not None
    assert alias_match["canonical"]["full_address"].startswith("The Wicket Hall (Strathcona Hotel)")


def test_scoped_alias_requires_matching_context() -> None:
    """Scoped alias should only apply with matching source and URL context."""
    handler = _build_handler_for_alias_tests()

    no_context_match = handler._find_address_alias_match(["Test Scoped Hall"])
    assert no_context_match is None

    context_match = handler._find_address_alias_match(
        ["Test Scoped Hall"],
        context={"url": "https://example.com/scoped/events", "source": "Scoped Source"},
    )
    assert context_match is not None
    assert context_match["canonical"]["address_id"] == 999


def test_alias_conflict_detection_blocks_mismatched_postal_codes() -> None:
    """Alias safety check should detect conflicting postal codes."""
    handler = _build_handler_for_alias_tests()
    conflict = handler._alias_conflicts_with_parsed_address(
        parsed_address={"postal_code": "V8W 2C2", "street_number": "919"},
        canonical={"postal_code": "V9A 1A1", "street_number": "919"},
    )
    assert conflict is True
