from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from read_pdfs import ReadPDFs


def test_pdf_source_active_when_no_window_fields() -> None:
    row = pd.Series({"source": "Always On PDF"})
    assert ReadPDFs.is_pdf_source_active(row, today=date(2026, 3, 23)) == (True, "active")


def test_pdf_source_inactive_before_start_date() -> None:
    row = pd.Series(
        {
            "source": "Victoria Summer Music",
            "enabled": "true",
            "active_start_date": "2026-06-01",
            "active_end_date": "2026-09-30",
        }
    )
    assert ReadPDFs.is_pdf_source_active(row, today=date(2026, 3, 23)) == (
        False,
        "before_active_start_date",
    )


def test_pdf_source_active_inside_window() -> None:
    row = pd.Series(
        {
            "source": "Victoria Summer Music",
            "enabled": "true",
            "active_start_date": "2026-06-01",
            "active_end_date": "2026-09-30",
        }
    )
    assert ReadPDFs.is_pdf_source_active(row, today=date(2026, 7, 15)) == (True, "active")


def test_pdf_source_inactive_after_end_date() -> None:
    row = pd.Series(
        {
            "source": "Victoria Summer Music",
            "enabled": "true",
            "active_start_date": "2026-06-01",
            "active_end_date": "2026-09-30",
        }
    )
    assert ReadPDFs.is_pdf_source_active(row, today=date(2026, 10, 1)) == (
        False,
        "after_active_end_date",
    )


def test_pdf_source_respects_enabled_flag() -> None:
    row = pd.Series(
        {
            "source": "Disabled PDF",
            "enabled": "false",
            "active_start_date": "2026-01-01",
            "active_end_date": "2026-12-31",
        }
    )
    assert ReadPDFs.is_pdf_source_active(row, today=date(2026, 7, 15)) == (False, "disabled")
