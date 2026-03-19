"""Centralized output artifact paths.

This module keeps the repo's output layout consistent after moving
top-level artifacts into logical subdirectories under ``output/``.
"""

from __future__ import annotations

import os
from typing import Final

OUTPUT_DIR: Final[str] = "output"
CHATBOT_DIR: Final[str] = os.path.join(OUTPUT_DIR, "chatbot")
CLASSIFIER_DIR: Final[str] = os.path.join(OUTPUT_DIR, "classifier")
CODEX_REVIEW_DIR: Final[str] = os.path.join(OUTPUT_DIR, "codex_review")
DUPLICATES_DIR: Final[str] = os.path.join(OUTPUT_DIR, "duplicates")
EVENTS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "events")
FB_DIR: Final[str] = os.path.join(OUTPUT_DIR, "fb")
REPLAY_DIR: Final[str] = os.path.join(OUTPUT_DIR, "replay")
REPORTS_DIR: Final[str] = os.path.join(OUTPUT_DIR, "reports")
TEST_DIR: Final[str] = os.path.join(OUTPUT_DIR, "test")


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory for a file path and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def chatbot_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(CHATBOT_DIR, filename))


def classifier_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(CLASSIFIER_DIR, filename))


def codex_review_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(CODEX_REVIEW_DIR, filename))


def duplicates_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(DUPLICATES_DIR, filename))


def events_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(EVENTS_DIR, filename))


def fb_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(FB_DIR, filename))


def replay_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(REPLAY_DIR, filename))


def reports_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(REPORTS_DIR, filename))


def test_output_path(filename: str) -> str:
    return ensure_parent_dir(os.path.join(TEST_DIR, filename))
