import asyncio
import logging
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from images import ImageScraper


class _FakeLoop:
    def __init__(self, return_value: str):
        self.return_value = return_value
        self.calls = []

    def run_until_complete(self, awaitable):
        self.calls.append(awaitable)
        return self.return_value


def test_dynamic_extractor_uses_local_instagram_path() -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.loop = _FakeLoop("ig text")

    called = {"rd_ext": 0}

    def _rd_ext_extract(_url):
        called["rd_ext"] += 1
        return "rd text token"

    def _local_extract(_url):
        return "local ig text token"

    scraper.read_extract = SimpleNamespace(extract_event_text=_rd_ext_extract)
    scraper._extract_page_text_playwright = _local_extract

    text = scraper._extract_dynamic_page_text("https://www.instagram.com/p/abc123/")
    assert text == "ig text"
    assert called["rd_ext"] == 0


def test_dynamic_extractor_uses_rd_ext_for_non_instagram() -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.loop = _FakeLoop("non-ig text")

    called = {"rd_ext": 0}

    def _rd_ext_extract(_url):
        called["rd_ext"] += 1
        return "rd text token"

    scraper.read_extract = SimpleNamespace(extract_event_text=_rd_ext_extract)

    text = scraper._extract_dynamic_page_text("https://example.com/events")
    assert text == "non-ig text"
    assert called["rd_ext"] == 1


def test_extract_page_text_playwright_sanitizes_html() -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")

    class _FakePage:
        async def goto(self, *_args, **_kwargs):
            return None

        async def wait_for_timeout(self, *_args, **_kwargs):
            return None

        async def content(self):
            return (
                "<html><head><style>.x{color:red}</style><script>ignore_me()</script></head>"
                "<body><div>Latin social dance tonight</div></body></html>"
            )

    scraper.read_extract = SimpleNamespace(page=_FakePage())
    text = asyncio.run(scraper._extract_page_text_playwright("https://www.instagram.com/p/xyz/"))
    assert text is not None
    assert "Latin social dance tonight" in text
    assert "ignore_me" not in text
