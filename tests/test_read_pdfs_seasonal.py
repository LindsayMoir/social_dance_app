from __future__ import annotations

import os
import sys
from datetime import date
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import read_pdfs
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


def test_resolve_parent_page_assets_uses_butchart_parent_page_links(monkeypatch) -> None:
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://butchartgardens.com/wp-content/uploads/2026/05/2026-Entertainment-calendar-SF-1-663x1024.png" />
      </head>
      <body>
        <a href="https://butchartgardens.com/wp-content/uploads/2026/05/2026-Entertainment-calendar-double-sided-2.pdf">
          Download Calendar
        </a>
      </body>
    </html>
    """

    def _fake_get(url: str, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        assert url == "https://butchartgardens.com/summer-entertainment-calendar/"
        assert headers == read_pdfs.DEFAULT_REQUEST_HEADERS
        assert timeout == 30
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr(read_pdfs.requests, "get", _fake_get)

    resolved = ReadPDFs.resolve_parent_page_assets(
        source="The Butchart Gardens Outdoor Summer Concerts",
        parent_url="https://butchartgardens.com/summer-entertainment-calendar/",
        fallback_pdf_url="https://old.example.com/calendar.pdf",
        fallback_image_url="https://old.example.com/calendar.png",
    )

    assert resolved["pdf_url"] == (
        "https://butchartgardens.com/wp-content/uploads/2026/05/"
        "2026-Entertainment-calendar-double-sided-2.pdf"
    )
    assert resolved["image_url"] == (
        "https://butchartgardens.com/wp-content/uploads/2026/05/"
        "2026-Entertainment-calendar-SF-1-663x1024.png"
    )


def test_resolve_parent_page_assets_falls_back_when_parent_page_fetch_fails(monkeypatch) -> None:
    def _fake_get(url: str, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        raise read_pdfs.requests.RequestException("boom")

    monkeypatch.setattr(read_pdfs.requests, "get", _fake_get)

    resolved = ReadPDFs.resolve_parent_page_assets(
        source="The Butchart Gardens Outdoor Summer Concerts",
        parent_url="https://butchartgardens.com/summer-entertainment-calendar/",
        fallback_pdf_url="https://old.example.com/calendar.pdf",
        fallback_image_url="https://old.example.com/calendar.png",
    )

    assert resolved["pdf_url"] == "https://old.example.com/calendar.pdf"
    assert resolved["image_url"] == "https://old.example.com/calendar.png"


def test_resolve_source_urls_handles_two_column_butchart_csv(monkeypatch) -> None:
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://butchartgardens.com/wp-content/uploads/2026/05/2026-Entertainment-calendar-SF-1-663x1024.png" />
      </head>
      <body>
        <a href="https://butchartgardens.com/wp-content/uploads/2026/05/2026-Entertainment-calendar-double-sided-2.pdf">
          Download Calendar
        </a>
      </body>
    </html>
    """

    def _fake_get(url: str, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr(read_pdfs.requests, "get", _fake_get)

    reader = ReadPDFs.__new__(ReadPDFs)
    reader.config = {"input": {}}

    resolved = reader.resolve_source_urls(
        pd.Series(
            {
                "source": "The Butchart Gardens Outdoor Summer Concerts",
                "parent_url": "https://butchartgardens.com/summer-entertainment-calendar/",
            }
        )
    )

    assert resolved["parent_url"] == "https://butchartgardens.com/summer-entertainment-calendar/"
    assert resolved["pdf_url"] == (
        "https://butchartgardens.com/wp-content/uploads/2026/05/"
        "2026-Entertainment-calendar-double-sided-2.pdf"
    )
    assert resolved["image_url"] == (
        "https://butchartgardens.com/wp-content/uploads/2026/05/"
        "2026-Entertainment-calendar-SF-1-663x1024.png"
    )


def test_resolve_source_urls_handles_two_column_victoria_summer_music_csv(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <a href="https://files.example.com/current-vsm-calendar.pdf">Summer Calendar</a>
      </body>
    </html>
    """

    def _fake_get(url: str, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        assert url == "https://ellaquinceier.wixsite.com/victoriasummermusic"
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr(read_pdfs.requests, "get", _fake_get)

    reader = ReadPDFs.__new__(ReadPDFs)
    reader.config = {"input": {}}

    resolved = reader.resolve_source_urls(
        pd.Series(
            {
                "source": "Victoria Summer Music",
                "parent_url": "https://ellaquinceier.wixsite.com/victoriasummermusic",
            }
        )
    )

    assert resolved["parent_url"] == "https://ellaquinceier.wixsite.com/victoriasummermusic"
    assert resolved["pdf_url"] == "https://files.example.com/current-vsm-calendar.pdf"
    assert resolved["image_url"] is None


def test_parse_butchart_gardens_concerts_returns_none_when_llm_query_fails(monkeypatch) -> None:
    fake_handler = SimpleNamespace(
        generate_prompt=lambda *_args, **_kwargs: ("prompt", "event_extraction"),
        query_openai=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("insufficient_quota")),
    )

    monkeypatch.setattr(read_pdfs, "dump_pdf_text", lambda _pdf_file: "calendar text")
    monkeypatch.setattr(read_pdfs, "get_llm_handler", lambda: fake_handler)

    result = read_pdfs.parse_butchart_gardens_concerts(
        pdf_file=object(),
        parser_context={
            "pdf_url": "https://butchartgardens.com/wp-content/uploads/2026/05/calendar.pdf",
            "image_url": "https://butchartgardens.com/wp-content/uploads/2026/05/calendar.png",
        },
    )

    assert result is None
