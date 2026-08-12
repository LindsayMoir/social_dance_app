"""Unit tests for optional Playwright ARIA-snapshot extraction helpers."""

from __future__ import annotations

import asyncio
import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from browser_extraction import (
    AriaSnapshotResult,
    capture_aria_snapshot_async,
    capture_aria_snapshot_sync,
    choose_extraction_text,
    reduce_instagram_aria_snapshot,
    snapshot_settings,
    write_snapshot_diagnostic,
)


class _AsyncLocator:
    async def aria_snapshot(self, *, timeout: int) -> str:
        assert timeout == 250
        return '- main:\n  - heading "Friday Salsa"'


class _AsyncPage:
    def locator(self, selector: str) -> _AsyncLocator:
        assert selector == "body"
        return _AsyncLocator()


class _SyncLocator:
    def aria_snapshot(self, *, timeout: int) -> str:
        assert timeout == 500
        return '- main:\n  - text: "Saturday dance"'


class _SyncPage:
    def locator(self, selector: str) -> _SyncLocator:
        assert selector == "body"
        return _SyncLocator()


class _UnsupportedPage:
    def locator(self, _selector: str) -> object:
        return object()


def test_capture_aria_snapshot_async_uses_body_locator() -> None:
    result = asyncio.run(capture_aria_snapshot_async(_AsyncPage(), timeout_ms=250))

    assert result.available
    assert "Friday Salsa" in str(result.text)
    assert result.reason is None


def test_capture_aria_snapshot_sync_uses_body_locator() -> None:
    result = capture_aria_snapshot_sync(_SyncPage(), timeout_ms=500)

    assert result.available
    assert "Saturday dance" in str(result.text)


def test_capture_aria_snapshot_returns_supported_fallback_reason() -> None:
    result = capture_aria_snapshot_sync(_UnsupportedPage())

    assert result.text is None
    assert result.reason == "aria_snapshot_unsupported"


def test_choose_extraction_text_keeps_dom_when_snapshot_is_disabled_or_too_short() -> None:
    snapshot = AriaSnapshotResult(text="short")

    assert choose_extraction_text("DOM event details", snapshot, enabled=False) == (
        "DOM event details",
        "dom_text",
    )
    assert choose_extraction_text("DOM event details", snapshot, enabled=True, min_chars=10) == (
        "DOM event details",
        "dom_text",
    )


def test_choose_extraction_text_uses_sufficient_enabled_snapshot() -> None:
    snapshot = AriaSnapshotResult(text="ARIA snapshot event details")

    assert choose_extraction_text("DOM event details", snapshot, enabled=True, min_chars=10) == (
        "ARIA snapshot event details",
        "aria_snapshot",
    )


def test_reduce_instagram_aria_snapshot_keeps_post_and_excludes_comments() -> None:
    snapshot = """- link \"Instagram\":
- main:
  - link \"danceclub\":
  - time: April 22, 2025
  - text: \"Country swing workshop at 8 PM\"
  - link \"commenter\":
  - text: Great event!
- contentinfo:
  - link \"Meta\":"""

    reduced = reduce_instagram_aria_snapshot(snapshot)

    assert reduced is not None
    assert "Country swing workshop" in reduced
    assert "commenter" not in reduced
    assert "contentinfo" not in reduced


def test_reduce_instagram_aria_snapshot_returns_none_without_caption_shape() -> None:
    assert reduce_instagram_aria_snapshot("- main:\n  - heading \"No post\"") is None


def test_snapshot_settings_coerces_invalid_values() -> None:
    settings = snapshot_settings(
        {
            "crawling": {
                "aria_snapshot_enabled": "yes",
                "aria_snapshot_instagram_enabled": "true",
                "aria_snapshot_timeout_ms": "invalid",
                "aria_snapshot_min_chars": 0,
            }
        }
    )

    assert settings["enabled"] is True
    assert settings["instagram_enabled"] is True
    assert settings["timeout_ms"] == 5_000
    assert settings["min_chars"] == 1


def test_write_snapshot_diagnostic_preserves_capture_context(tmp_path) -> None:
    write_snapshot_diagnostic(
        debug_dir=str(tmp_path),
        source="facebook",
        url="https://www.facebook.com/events/123",
        snapshot=AriaSnapshotResult(text="- main:\n  - heading \"Dance\""),
        selected_representation="aria_snapshot",
        fallback_text_length=123,
    )

    records = list(tmp_path.glob("*.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text(encoding="utf-8"))
    assert payload["source"] == "facebook"
    assert payload["selected_representation"] == "aria_snapshot"
    assert payload["snapshot"]["text"].startswith("- main")
