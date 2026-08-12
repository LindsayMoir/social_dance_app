"""Shared, optional AI accessibility-snapshot extraction helpers.

These helpers operate only on a page that acquisition code has already rendered.
They do not navigate, authenticate, scroll, or otherwise change browser behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


DEFAULT_SNAPSHOT_TIMEOUT_MS = 5_000
DEFAULT_SNAPSHOT_MIN_CHARS = 80
DEFAULT_SNAPSHOT_MAX_CHARS = 40_000


@dataclass(frozen=True)
class AriaSnapshotResult:
    """Result of requesting an ARIA snapshot from an already-rendered page."""

    text: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Return whether the browser supplied non-empty snapshot text."""
        return bool(self.text)


def snapshot_settings(config: dict[str, Any]) -> dict[str, int | bool | str]:
    """Return validated ARIA-snapshot settings from the crawling configuration."""
    crawling = config.get("crawling", {}) if isinstance(config, dict) else {}
    if not isinstance(crawling, dict):
        crawling = {}

    return {
        "enabled": _as_bool(crawling.get("aria_snapshot_enabled", False)),
        "instagram_enabled": _as_bool(crawling.get("aria_snapshot_instagram_enabled", False)),
        "debug_enabled": _as_bool(crawling.get("aria_snapshot_debug_enabled", False)),
        "timeout_ms": _positive_int(
            crawling.get("aria_snapshot_timeout_ms"), DEFAULT_SNAPSHOT_TIMEOUT_MS
        ),
        "min_chars": _positive_int(
            crawling.get("aria_snapshot_min_chars"), DEFAULT_SNAPSHOT_MIN_CHARS
        ),
        "max_chars": _positive_int(
            crawling.get("aria_snapshot_max_chars"), DEFAULT_SNAPSHOT_MAX_CHARS
        ),
        "debug_dir": str(
            crawling.get("aria_snapshot_debug_dir", "debug/aria_snapshots")
        ).strip()
        or "debug/aria_snapshots",
    }


async def capture_aria_snapshot_async(
    page: Any,
    *,
    timeout_ms: int = DEFAULT_SNAPSHOT_TIMEOUT_MS,
    max_chars: int = DEFAULT_SNAPSHOT_MAX_CHARS,
) -> AriaSnapshotResult:
    """Capture the body ARIA snapshot through Playwright's async Locator API."""
    try:
        locator = page.locator("body")
        aria_snapshot = getattr(locator, "aria_snapshot", None)
        if not callable(aria_snapshot):
            return AriaSnapshotResult(text=None, reason="aria_snapshot_unsupported")
        snapshot = await aria_snapshot(timeout=timeout_ms)
    except Exception as exc:
        return AriaSnapshotResult(text=None, reason=f"aria_snapshot_error:{type(exc).__name__}")
    return _normalize_snapshot(snapshot, max_chars=max_chars)


def capture_aria_snapshot_sync(
    page: Any,
    *,
    timeout_ms: int = DEFAULT_SNAPSHOT_TIMEOUT_MS,
    max_chars: int = DEFAULT_SNAPSHOT_MAX_CHARS,
) -> AriaSnapshotResult:
    """Capture the body ARIA snapshot through Playwright's sync Locator API."""
    try:
        locator = page.locator("body")
        aria_snapshot = getattr(locator, "aria_snapshot", None)
        if not callable(aria_snapshot):
            return AriaSnapshotResult(text=None, reason="aria_snapshot_unsupported")
        snapshot = aria_snapshot(timeout=timeout_ms)
    except Exception as exc:
        return AriaSnapshotResult(text=None, reason=f"aria_snapshot_error:{type(exc).__name__}")
    return _normalize_snapshot(snapshot, max_chars=max_chars)


def choose_extraction_text(
    fallback_text: str | None,
    snapshot: AriaSnapshotResult,
    *,
    enabled: bool,
    min_chars: int = DEFAULT_SNAPSHOT_MIN_CHARS,
) -> tuple[str | None, str]:
    """Choose a usable snapshot only when the experimental feature is enabled."""
    fallback = str(fallback_text or "").strip() or None
    snapshot_text = str(snapshot.text or "").strip() or None
    if enabled and snapshot_text and len(snapshot_text) >= max(1, min_chars):
        return snapshot_text, "aria_snapshot"
    return fallback, "dom_text"


def reduce_instagram_aria_snapshot(snapshot_text: str | None) -> str | None:
    """Return the canonical Instagram post block from a Playwright ARIA snapshot.

    The accessibility tree places the post inside ``main`` and typically follows
    it with comment, recommendation, footer, and messaging subtrees. This
    reducer keeps only the main subtree through the first caption text node,
    including useful post-image alt text.
    It returns ``None`` when that stable shape is not present so callers can use
    the original snapshot unchanged.
    """
    lines = str(snapshot_text or "").splitlines()
    main_index = next((index for index, line in enumerate(lines) if line == "- main:"), None)
    if main_index is None:
        return None

    reduced = [lines[main_index]]
    saw_timestamp = False
    for line in lines[main_index + 1 :]:
        indentation = len(line) - len(line.lstrip(" "))
        if indentation == 0:
            break
        stripped_line = line.lstrip()
        if stripped_line.startswith("- img"):
            image_description = _aria_image_description(stripped_line)
            if image_description:
                reduced.append(f"{' ' * indentation}- text: Image description: {image_description}")
            continue
        reduced.append(line)
        if indentation == 2 and stripped_line.startswith("- time:"):
            saw_timestamp = True
            continue
        if saw_timestamp and indentation == 2 and stripped_line.startswith("- text:"):
            return "\n".join(reduced).strip()
    return None


def _aria_image_description(line: str) -> str | None:
    """Return useful Instagram post image alt text, excluding decorative images."""
    if not line.startswith('- img "') or not line.endswith('"'):
        return None
    description = line[len('- img "') : -1].strip()
    if not description:
        return None
    normalized_description = description.lower()
    ignored_descriptions = {
        "instagram",
        "tags",
        "more options",
        "like",
        "comment",
        "share",
        "save",
    }
    if (
        normalized_description in ignored_descriptions
        or "profile picture" in normalized_description
    ):
        return None
    return description


def write_snapshot_diagnostic(
    *,
    debug_dir: str,
    source: str,
    url: str,
    snapshot: AriaSnapshotResult,
    selected_representation: str,
    fallback_text_length: int,
) -> None:
    """Persist a bounded, opt-in snapshot capture record for reproducible debugging."""
    try:
        path = Path(debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        fingerprint = sha256(str(url).encode("utf-8")).hexdigest()[:16]
        target = path / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{fingerprint}.json"
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source or "unknown"),
            "url": str(url or ""),
            "selected_representation": selected_representation,
            "fallback_text_length": max(0, int(fallback_text_length)),
            "snapshot": asdict(snapshot),
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logging.warning("write_snapshot_diagnostic(): failed for %s: %s", url, exc)


def _normalize_snapshot(value: Any, *, max_chars: int) -> AriaSnapshotResult:
    """Normalize browser output and bound diagnostic/LLM input size."""
    text = str(value or "").strip()
    if not text:
        return AriaSnapshotResult(text=None, reason="aria_snapshot_empty")
    return AriaSnapshotResult(text=text[:max(1, max_chars)])


def _positive_int(value: Any, default: int) -> int:
    """Return a positive integer configuration value or its default."""
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    """Coerce common configuration boolean forms without truthiness surprises."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
