from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_notifier import EmailNotifier


def test_determine_status_handles_missing_average_score() -> None:
    notifier = EmailNotifier.__new__(EmailNotifier)

    status = notifier._determine_status(
        {
            "execution_success_rate": 1.0,
            "average_score": None,
        }
    )

    assert status == "⚠ WARNING"


def test_determine_status_handles_missing_execution_rate() -> None:
    notifier = EmailNotifier.__new__(EmailNotifier)

    status = notifier._determine_status(
        {
            "execution_success_rate": None,
            "average_score": 85.0,
        }
    )

    assert status == "INFO"
