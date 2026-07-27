from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "validation"))

from scraping_validator import ScrapingValidator


class _FakeDbHandler:
    def execute_query(self, query, params=None, statement_timeout_ms=None):
        normalized = " ".join(str(query).split())
        if "SELECT source, COUNT(*) AS counted FROM events" in normalized:
            assert statement_timeout_ms == 15000
            return [
                ("Salsa Caliente", 20),
                ("Victoria Summer Music", 9),
                ("Victoria Latin Dance Association", 15),
                ("WCS Lessons, Social Dances, and Conventions – BC Swing Dance", 12),
                ("Red Hot Swing", 8),
                ("Eventbrite", 50),
                ("The Loft Pub Victoria", 10),
            ]
        return []


def test_check_source_distribution_accepts_two_column_pdf_sources_csv(tmp_path) -> None:
    pdf_sources = tmp_path / "pdfs.csv"
    pdf_sources.write_text(
        "source,parent_url\n"
        "Victoria Summer Music,https://example.com/vsm\n",
        encoding="utf-8",
    )

    validator = ScrapingValidator(
        _FakeDbHandler(),
        {
            "testing": {
                "validation": {
                    "scraping": {
                        "pdf_sources_file": str(pdf_sources),
                    }
                }
            }
        },
    )

    result = validator.check_source_distribution()

    assert result["status"] != "FAIL"
    assert "Victoria Summer Music" not in result["missing_sources"]
