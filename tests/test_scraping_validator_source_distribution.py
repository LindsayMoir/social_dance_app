from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "validation"))

from scraping_validator import ScrapingValidator


class _FakeDbHandler:
    def execute_query(self, query, params=None):
        normalized = " ".join(str(query).split())
        if "SELECT source, COUNT(*) AS counted FROM events" in normalized:
            return [
                ("Salsa Caliente", 20),
                ("Victoria Latin Dance Association", 15),
                ("WCS Lessons, Social Dances, and Conventions – BC Swing Dance", 12),
                ("Red Hot Swing", 8),
                ("Eventbrite", 50),
                ("The Loft Pub Victoria", 10),
            ]
        return []


def test_check_source_distribution_excludes_inactive_seasonal_pdf_sources(tmp_path) -> None:
    pdf_sources = tmp_path / "pdfs.csv"
    pdf_sources.write_text(
        "pdf_url,parent_url,source,keywords,enabled,active_start_date,active_end_date\n"
        "https://example.com/vsm.pdf,https://example.com/vsm,Victoria Summer Music,\"dance, live music\",true,2026-06-01,2026-09-30\n",
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
