import asyncio
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import images
from images import ImageScraper, _extract_instagram_post_links


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


def test_extract_instagram_post_links_from_profile_html() -> None:
    html = """
    <html><body>
      <a href="/p/ABC123/">Post 1</a>
      <a href="/reel/XYZ789/?hl=en">Reel</a>
      <a href="/bachatavictoria/">Profile</a>
      <a href="/p/ABC123/">Duplicate</a>
    </body></html>
    """
    links = _extract_instagram_post_links(html, "https://www.instagram.com/bachatavictoria/")
    assert links == [
        "https://www.instagram.com/p/ABC123/",
        "https://www.instagram.com/reel/XYZ789/?hl=en",
    ]


def test_process_webpage_url_expands_instagram_posts_before_page_images(monkeypatch) -> None:
    original_process_webpage_url = ImageScraper.process_webpage_url
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata"]
    scraper.images_per_page_limit = 2
    scraper.config = {"crawling": {"max_website_urls": 10, "prompt_max_length": 10000}}

    class _FakeDb:
        def write_url_to_db(self, _row):
            return None

    class _FakeLLM:
        def generate_prompt(self, *_args, **_kwargs):
            return ("prompt", "event_extraction")

        def process_llm_response(self, *_args, **_kwargs):
            return False

    class _FakePage:
        async def content(self):
            return """
            <html><body>
              <a href="/p/POST001/">Post 1</a>
              <a href="/p/POST002/">Post 2</a>
              <img src="https://static.cdninstagram.com/ui.webp" />
            </body></html>
            """

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.read_extract = SimpleNamespace(page=_FakePage())
    def _run_until_complete(awaitable):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return """
        <html><body>
          <a href="/p/POST001/">Post 1</a>
          <a href="/p/POST002/">Post 2</a>
          <img src="https://static.cdninstagram.com/ui.webp" />
        </body></html>
        """

    scraper.loop = SimpleNamespace(run_until_complete=_run_until_complete)
    scraper._extract_dynamic_page_text = lambda _url: "bachata victoria profile text"
    fake_response = Mock()
    fake_response.text = "<html><body>short</body></html>"
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(images.requests, "get", lambda *_args, **_kwargs: fake_response)

    called_posts: list[tuple[str, str, str, str]] = []
    called_images: list[tuple[str, str, str, str]] = []

    def _process_post(url, parent, source, keywords):
        called_posts.append((url, parent, source, keywords))

    def _process_image(url, parent, source, keywords):
        called_images.append((url, parent, source, keywords))

    scraper.process_image_url = _process_image
    scraper.process_webpage_url = _process_post

    original_process_webpage_url(
        scraper,
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Sebastian y Hannah",
        "bachata",
    )

    assert called_posts == [
        ("https://www.instagram.com/p/POST001/", "https://www.instagram.com/bachatavictoria/", "Sebastian y Hannah", "bachata"),
        ("https://www.instagram.com/p/POST002/", "https://www.instagram.com/bachatavictoria/", "Sebastian y Hannah", "bachata"),
    ]
    assert called_images == []
