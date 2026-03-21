import asyncio
import time
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import images
from images import (
    ImageScraper,
    _build_image_context_text,
    _extract_instagram_post_links,
    _is_degraded_instagram_profile_text,
    _is_instagram_login_redirect_url,
    _is_ignored_instagram_ui_asset,
    _looks_like_authenticated_instagram_profile,
    _safe_screenshot_stem,
    _score_image_candidate,
)


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


def test_build_image_context_text_includes_parent_context() -> None:
    combined = _build_image_context_text(
        ocr_text="715 Yates St 8PM",
        parent_url="https://www.instagram.com/bachatavictoria/",
        source="Bachata Victoria BC",
        page_context_text="Next Social: March 20th",
    )
    assert "Source: Bachata Victoria BC" in combined
    assert "Parent_URL: https://www.instagram.com/bachatavictoria/" in combined
    assert "Parent_Page_Text: Next Social: March 20th" in combined
    assert "715 Yates St 8PM" in combined


def test_instagram_ui_asset_filter_and_scoring() -> None:
    ui_url = "https://static.cdninstagram.com/rsrc.php/yJ/r/53X3pk-t2Gn.webp"
    poster_url = "https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster.jpg"
    assert _is_ignored_instagram_ui_asset(ui_url) is True
    assert _is_ignored_instagram_ui_asset(poster_url) is False
    assert _score_image_candidate(ui_url, 1200, 1200) < 0
    assert _score_image_candidate(poster_url, 1080, 1350) > _score_image_candidate(poster_url, 1080, 400)


def test_instagram_degraded_profile_text_detection() -> None:
    assert _is_degraded_instagram_profile_text("This content is no longer available. Sign up for Instagram.") is True
    assert _is_degraded_instagram_profile_text("Bachata Victoria BC 116 posts 1100 followers Next Social: March 20th") is False


def test_instagram_login_redirect_detection() -> None:
    assert _is_instagram_login_redirect_url(
        "https://www.instagram.com/accounts/login/?next=%2Fbachatavictoria%2F&source=omni_redirect"
    ) is True
    assert _is_instagram_login_redirect_url("https://www.instagram.com/bachatavictoria/") is False


def test_authenticated_instagram_profile_detection() -> None:
    assert _looks_like_authenticated_instagram_profile(
        "https://www.instagram.com/bachatavictoria/",
        "Bachata Victoria BC 116 posts 1,098 followers 756 following Next Social: March 20th",
    ) is True
    assert _looks_like_authenticated_instagram_profile(
        "https://www.instagram.com/accounts/login/?next=%2Fbachatavictoria%2F&source=omni_redirect",
        "See everyday moments from your close friends. Continue Use another profile Create new account.",
    ) is False


def test_manual_instagram_recovery_skips_in_headless_mode() -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.config = {"crawling": {"headless": True}}

    result = asyncio.run(scraper._attempt_manual_instagram_recovery("https://www.instagram.com/bachatavictoria/"))
    assert result is False


def test_safe_screenshot_stem_is_filesystem_safe() -> None:
    stem = _safe_screenshot_stem("https://www.instagram.com/bachatavictoria/p/ABC123/?hl=en")
    assert "/" not in stem
    assert "instagram.com" in stem


def test_process_webpage_url_expands_instagram_posts_before_page_images(monkeypatch) -> None:
    original_process_webpage_url = ImageScraper.process_webpage_url
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata"]
    scraper.images_per_page_limit = 2
    scraper.max_images_per_page = 5
    scraper.instagram_vision_image_limit = 3
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


def test_process_webpage_url_uses_playwright_post_fallback_when_html_has_no_posts(monkeypatch) -> None:
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
            return '<html><body><img src="https://static.cdninstagram.com/ui.webp" /></body></html>'

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.read_extract = SimpleNamespace(page=_FakePage())

    def _run_until_complete(awaitable):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        name = getattr(getattr(awaitable, "cr_code", None), "co_name", "")
        if name == "_extract_instagram_post_links_playwright":
            return [
                "https://www.instagram.com/p/FALLBACK001/",
                "https://www.instagram.com/reel/FALLBACK002/",
            ]
        return '<html><body><img src="https://static.cdninstagram.com/ui.webp" /></body></html>'

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
        ("https://www.instagram.com/p/FALLBACK001/", "https://www.instagram.com/bachatavictoria/", "Sebastian y Hannah", "bachata"),
        ("https://www.instagram.com/reel/FALLBACK002/", "https://www.instagram.com/bachatavictoria/", "Sebastian y Hannah", "bachata"),
    ]
    assert called_images == []


def test_process_webpage_url_skips_degraded_instagram_shell_without_posts_or_media(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata"]
    scraper.images_per_page_limit = 2
    scraper.config = {"crawling": {"max_website_urls": 10, "prompt_max_length": 10000}}

    written_rows: list[tuple] = []
    screenshot_calls: list[tuple[str, str, str, str, str | None]] = []

    class _FakeDb:
        def write_url_to_db(self, row):
            written_rows.append(row)

    class _FakeLLM:
        def generate_prompt(self, *_args, **_kwargs):
            raise AssertionError("profile LLM should not run for degraded shell pages without posts/media")

        def process_llm_response(self, *_args, **_kwargs):
            raise AssertionError("profile LLM should not run for degraded shell pages without posts/media")

    class _FakePage:
        async def content(self):
            return '<html><body><img src="https://static.cdninstagram.com/rsrc.php/yJ/r/ui.webp" /></body></html>'

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.read_extract = SimpleNamespace(page=_FakePage())

    def _run_until_complete(awaitable):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        name = getattr(getattr(awaitable, "cr_code", None), "co_name", "")
        if name == "_extract_instagram_post_links_playwright":
            return []
        return '<html><body><img src="https://static.cdninstagram.com/rsrc.php/yJ/r/ui.webp" /></body></html>'

    scraper.loop = SimpleNamespace(run_until_complete=_run_until_complete)
    scraper._extract_dynamic_page_text = lambda _url: (
        "See everyday moments from your close friends. Continue Use another profile Create new account."
    )
    scraper._process_local_image_path = lambda path, canonical_url, parent_url, source, keywords, page_context_text=None: screenshot_calls.append(
        (str(path), canonical_url, source, keywords, page_context_text)
    ) or True
    fake_response = Mock()
    fake_response.text = "<html><body>short</body></html>"
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(images.requests, "get", lambda *_args, **_kwargs: fake_response)

    ImageScraper.process_webpage_url(
        scraper,
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Sebastian y Hannah",
        "bachata",
    )

    assert written_rows == []
    assert len(screenshot_calls) == 1
    assert screenshot_calls[0][1] == "https://www.instagram.com/bachatavictoria/"


def test_process_webpage_url_skips_profile_llm_for_degraded_shell_with_posts(monkeypatch) -> None:
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
            raise AssertionError("profile LLM should be skipped when post links are available")

        def process_llm_response(self, *_args, **_kwargs):
            raise AssertionError("profile LLM should be skipped when post links are available")

    class _FakePage:
        async def content(self):
            return """
            <html><body>
              <a href="/p/POST001/">Post 1</a>
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
          <img src="https://static.cdninstagram.com/ui.webp" />
        </body></html>
        """

    scraper.loop = SimpleNamespace(run_until_complete=_run_until_complete)
    scraper._extract_dynamic_page_text = lambda _url: (
        "See everyday moments from your close friends. Continue Use another profile Create new account. Bachata Victoria."
    )
    fake_response = Mock()
    fake_response.text = "<html><body>short</body></html>"
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(images.requests, "get", lambda *_args, **_kwargs: fake_response)

    called_posts: list[tuple[str, str, str, str]] = []
    scraper.process_webpage_url = lambda url, parent, source, keywords: called_posts.append((url, parent, source, keywords))

    original_process_webpage_url(
        scraper,
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Sebastian y Hannah",
        "bachata",
    )

    assert called_posts == [
        ("https://www.instagram.com/p/POST001/", "https://www.instagram.com/bachatavictoria/", "Sebastian y Hannah", "bachata"),
    ]


def test_process_image_url_uses_parent_url_for_prompt_context(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata", "dance", "salsa"]
    scraper._process_local_image_path_with_vision = lambda *_args, **_kwargs: False

    class _FakeDb:
        def check_image_events_exist(self, _url):
            return False

        def should_process_url(self, _url):
            return True

        def write_url_to_db(self, _row):
            return None

    class _FakeLLM:
        def __init__(self):
            self.generate_args = None
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            self.generate_args = (url, extracted_text, prompt_type)
            return ("prompt", "event_extraction")

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return False

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.download_image = lambda _url: "images/fake.jpg"
    scraper.ocr_image_to_text = lambda _path: "BACHATA 715 YATES ST 8PM SOCIAL DANCE"

    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: (None, None))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda url, fallback_prompt_type='default': "fb" if "instagram.com" in url else fallback_prompt_type)

    scraper.process_image_url(
        "https://instagram.fcxh2-1.fna.fbcdn.net/poster.jpg",
        "https://www.instagram.com/bachatavictoria/",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    generate_url, generate_text, generate_prompt_type = scraper.llm_handler.generate_args
    assert generate_url == "https://www.instagram.com/bachatavictoria/"
    assert generate_prompt_type == "fb"
    assert "Parent_Page_Text: Next Social: March 20th" in generate_text
    assert "715 Yates ST".lower() in generate_text.lower()


def test_process_image_url_uses_ocr_before_vision() -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata", "dance", "salsa"]

    class _FakeDb:
        def check_image_events_exist(self, _url):
            return False

        def should_process_url(self, _url):
            return True

        def write_url_to_db(self, _row):
            return None

    scraper.db_handler = _FakeDb()
    scraper.download_image = lambda _url: "images/fake.jpg"

    called = {"vision": 0, "ocr": 0}

    def _vision(path, canonical_url, parent_url, source, keywords, page_context_text=None):
        called["vision"] += 1
        assert str(path) == "images/fake.jpg"
        assert canonical_url == "https://www.instagram.com/bachatavictoria/"
        assert keywords == "bachata"
        return True

    scraper._process_local_image_path_with_vision = _vision
    scraper.ocr_image_to_text = lambda _path: called.__setitem__("ocr", called["ocr"] + 1) or ""

    scraper.process_image_url(
        "https://instagram.fcxh2-1.fna.fbcdn.net/poster.jpg",
        "https://www.instagram.com/bachatavictoria/",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    assert called["ocr"] == 1
    assert called["vision"] == 1


def test_process_image_url_can_skip_vision_after_rank_limit(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata", "dance", "salsa"]

    class _FakeDb:
        def check_image_events_exist(self, _url):
            return False

        def should_process_url(self, _url):
            return True

        def write_url_to_db(self, _row):
            return None

    class _FakeLLM:
        def __init__(self):
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            return ("prompt", "event_extraction")

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return True

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.download_image = lambda _url: "images/fake.jpg"

    called = {"vision": 0, "ocr": 0}

    def _vision(*_args, **_kwargs):
        called["vision"] += 1
        return True

    scraper._process_local_image_path_with_vision = _vision
    scraper.ocr_image_to_text = lambda _path: called.__setitem__("ocr", called["ocr"] + 1) or "BACHATA 715 YATES ST 8PM"

    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: (None, None))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda *_args, **_kwargs: "fb")

    scraper.process_image_url(
        "https://instagram.fcxh2-1.fna.fbcdn.net/poster.jpg",
        "https://www.instagram.com/bachatavictoria/",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
        use_vision_first=False,
    )

    assert called["vision"] == 0
    assert called["ocr"] == 1
    assert scraper.llm_handler.process_args is not None


def test_process_local_image_path_reuses_existing_ocr_llm_flow(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.keywords_list = ["bachata", "dance"]
    scraper._process_local_image_path_with_vision = lambda *_args, **_kwargs: False

    class _FakeLLM:
        def __init__(self):
            self.generate_args = None
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            self.generate_args = (url, extracted_text, prompt_type)
            return ("prompt", "event_extraction")

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return True

    scraper.llm_handler = _FakeLLM()
    scraper.ocr_image_to_text = lambda _path: "BACHATA 715 YATES ST 8PM SOCIAL DANCE"
    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: ("2026-03-20", "Friday"))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda *_args, **_kwargs: "fb")

    result = scraper._process_local_image_path(
        Path("/tmp/fake_screenshot.png"),
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    assert result is True
    generate_url, generate_text, generate_prompt_type = scraper.llm_handler.generate_args
    assert generate_url == "https://www.instagram.com/bachatavictoria/"
    assert generate_prompt_type == "fb"
    assert "Detected_Date: 2026-03-20" in generate_text


def test_process_local_image_path_uses_ocr_first(tmp_path, monkeypatch) -> None:
    screenshot_path = tmp_path / "poster.png"
    screenshot_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.keywords_list = ["bachata", "dance"]

    written = {}

    class _FakeDb:
        def write_events_to_db(self, events_df, url, parent_url, source, keywords):
            written["events_df"] = events_df.copy()
            written["url"] = url
            written["parent_url"] = parent_url
            written["source"] = source
            written["keywords"] = keywords

    class _FakeLLM:
        def __init__(self):
            self.generate_args = None
            self.query_args = None
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            self.generate_args = (url, extracted_text, prompt_type)
            return ("prompt", "event_extraction")

        def query_openai(self, prompt, model, image_url=None, schema_type=None):
            self.query_args = (prompt, model, image_url, schema_type)
            return '[{"event_name":"Bachata Victoria Social Night","start_date":"2026-03-20"}]'

        def extract_and_parse_json(self, result, url, schema_type=None):
            return [{"event_name": "Bachata Victoria Social Night", "start_date": "2026-03-20"}]

        def _apply_url_context_to_events_df(self, events_df, url, parent_url):
            events_df = events_df.copy()
            events_df["url"] = url
            return events_df

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return True

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.ocr_image_to_text = lambda _path: "BACHATA 715 YATES ST 8PM SOCIAL DANCE"
    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: (None, None))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda *_args, **_kwargs: "fb")

    result = scraper._process_local_image_path(
        screenshot_path,
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    assert result is True
    assert "events_df" not in written
    assert scraper.llm_handler.query_args is None
    assert scraper.llm_handler.process_args is not None


def test_process_local_image_path_falls_back_when_vision_times_out(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.keywords_list = ["bachata", "dance"]

    class _FakeLLM:
        def __init__(self):
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            return ("prompt", "event_extraction")

        def query_openai(self, *args, **kwargs):
            time.sleep(images._VISION_REQUEST_TIMEOUT_SECONDS + 1)
            return None

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return True

    scraper.llm_handler = _FakeLLM()
    scraper.ocr_image_to_text = lambda _path: "BACHATA 715 YATES ST 8PM SOCIAL DANCE"
    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: (None, None))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda *_args, **_kwargs: "fb")

    result = scraper._process_local_image_path(
        Path("/tmp/fake_screenshot.png"),
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    assert result is True
    assert scraper.llm_handler.process_args is not None


def test_process_local_image_path_falls_back_to_ocr_when_vision_fails(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.keywords_list = ["bachata", "dance"]
    scraper._process_local_image_path_with_vision = lambda *_args, **_kwargs: False

    class _FakeLLM:
        def __init__(self):
            self.process_args = None

        def generate_prompt(self, url, extracted_text, prompt_type):
            return ("prompt", "event_extraction")

        def process_llm_response(self, image_url, parent_url, extracted_text, source, found, prompt_type):
            self.process_args = (image_url, parent_url, extracted_text, source, found, prompt_type)
            return True

    scraper.llm_handler = _FakeLLM()
    scraper.ocr_image_to_text = lambda _path: "BACHATA 715 YATES ST 8PM SOCIAL DANCE"
    monkeypatch.setattr(images, "detect_date_from_image", lambda _path: (None, None))
    monkeypatch.setattr(images, "resolve_prompt_type", lambda *_args, **_kwargs: "fb")

    result = scraper._process_local_image_path(
        Path("/tmp/fake_screenshot.png"),
        "https://www.instagram.com/bachatavictoria/",
        "",
        "Bachata Victoria BC",
        "bachata",
        page_context_text="Next Social: March 20th",
    )

    assert result is True
    assert scraper.llm_handler.process_args is not None
    assert "Parent_Page_Text: Next Social: March 20th" in scraper.llm_handler.process_args[2]
    process_url, process_parent_url, process_text, process_source, process_found, process_prompt_type = scraper.llm_handler.process_args
    assert process_url == "https://www.instagram.com/bachatavictoria/"
    assert process_source == "Bachata Victoria BC"
    assert "bachata" in process_found
    assert process_prompt_type == "fb"


def test_ocr_image_to_text_prefers_paddleocr(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")

    class _FakePaddle:
        def ocr(self, *_args, **_kwargs):
            return [[
                [None, ("BACHATA VICTORIA", 0.99)],
                [None, ("715 YATES ST", 0.98)],
            ]]

    monkeypatch.setattr(ImageScraper, "_paddle_ocr_engine", _FakePaddle())
    monkeypatch.setattr(ImageScraper, "_paddle_ocr_init_attempted", True)
    monkeypatch.setattr(images.pytesseract, "image_to_string", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tesseract should not be used")))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        images.Image.new("RGB", (40, 40), color="white").save(tmp_path)
        text = scraper.ocr_image_to_text(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    assert "BACHATA VICTORIA" in text
    assert "715 YATES ST" in text


def test_ocr_image_to_text_falls_back_to_tesseract(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")

    monkeypatch.setattr(ImageScraper, "_paddle_ocr_engine", None)
    monkeypatch.setattr(ImageScraper, "_paddle_ocr_init_attempted", True)
    monkeypatch.setattr(images.pytesseract, "image_to_string", lambda *_args, **_kwargs: "fallback text")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        images.Image.new("RGB", (40, 40), color="white").save(tmp_path)
        text = scraper.ocr_image_to_text(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    assert text == "fallback text"


def test_process_webpage_url_ranks_instagram_images_and_skips_ui_assets(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata"]
    scraper.images_per_page_limit = 2
    scraper.max_images_per_page = 5
    scraper.instagram_vision_image_limit = 3
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
              <img src="https://static.cdninstagram.com/rsrc.php/yJ/r/ui.webp" />
              <img src="https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster1.jpg" />
              <img src="https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster2.jpg" />
            </body></html>
            """

        request = None

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.read_extract = SimpleNamespace(page=_FakePage())
    scraper._extract_dynamic_page_text = lambda _url: "bachata victoria profile text"
    fake_response = Mock()
    fake_response.text = "<html><body>short</body></html>"
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(images.requests, "get", lambda *_args, **_kwargs: fake_response)

    results = [
        """
        <html><body>
          <img src="https://static.cdninstagram.com/rsrc.php/yJ/r/ui.webp" />
          <img src="https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster1.jpg" />
          <img src="https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster2.jpg" />
        </body></html>
        """,
        b"ui",
        b"poster1",
        b"poster2",
    ]

    def _run_until_complete(awaitable):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return results.pop(0)

    scraper.loop = SimpleNamespace(run_until_complete=_run_until_complete)

    class _FakeImage:
        def __init__(self, size):
            self.size = size

    def _fake_open(obj):
        value = getattr(obj, "getvalue", lambda: b"")().decode("utf-8", errors="ignore")
        if "poster1" in value:
            return _FakeImage((1080, 1350))
        if "poster2" in value:
            return _FakeImage((1080, 1080))
        return _FakeImage((1200, 1200))

    def _request_get(url):
        if "poster2" in url:
            async def _poster2():
                return b"poster2"
            return SimpleNamespace(status=200, body=_poster2)
        if "poster1" in url:
            async def _poster1():
                return b"poster1"
            return SimpleNamespace(status=200, body=_poster1)
        async def _ui():
            return b"ui"
        return SimpleNamespace(status=200, body=_ui)

    scraper.read_extract.page.request = SimpleNamespace(get=_request_get)
    monkeypatch.setattr(images, "Image", SimpleNamespace(open=_fake_open))

    called_images: list[tuple[str, str, str, str, str | None, bool]] = []

    def _process_image(url, parent, source, keywords, page_context_text=None, use_vision_first=True):
        called_images.append((url, parent, source, keywords, page_context_text, use_vision_first))

    scraper.process_image_url = _process_image

    ImageScraper.process_webpage_url(
        scraper,
        "https://www.instagram.com/p/POST001/",
        "https://www.instagram.com/bachatavictoria/",
        "Sebastian y Hannah",
        "bachata",
    )

    assert {row[0] for row in called_images} == {
        "https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster1.jpg",
        "https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster2.jpg",
    }
    assert all("static.cdninstagram.com" not in row[0] for row in called_images)


def test_process_webpage_url_hard_caps_processed_images_at_five(monkeypatch) -> None:
    scraper = ImageScraper.__new__(ImageScraper)
    scraper.logger = logging.getLogger("test.images")
    scraper.urls_visited = set()
    scraper.keywords_list = ["bachata"]
    scraper.images_per_page_limit = 10
    scraper.max_images_per_page = 5
    scraper.instagram_vision_image_limit = 3
    scraper.config = {"crawling": {"max_website_urls": 10, "prompt_max_length": 10000}}

    class _FakeDb:
        def write_url_to_db(self, _row):
            return None

    class _FakeLLM:
        def generate_prompt(self, *_args, **_kwargs):
            return ("prompt", "event_extraction")

        def process_llm_response(self, *_args, **_kwargs):
            return False

    img_tags = "\n".join(
        f'<img src="https://instagram.fcxh2-1.fna.fbcdn.net/v/t51.82787-15/poster{i}.jpg" />'
        for i in range(7)
    )

    class _FakePage:
        async def content(self):
            return f"<html><body>{img_tags}</body></html>"

        request = None

    scraper.db_handler = _FakeDb()
    scraper.llm_handler = _FakeLLM()
    scraper.read_extract = SimpleNamespace(page=_FakePage())
    scraper._extract_dynamic_page_text = lambda _url: "bachata victoria profile text"

    fake_response = Mock()
    fake_response.text = "<html><body>short</body></html>"
    fake_response.raise_for_status.return_value = None
    monkeypatch.setattr(images.requests, "get", lambda *_args, **_kwargs: fake_response)

    rendered_html = f"<html><body>{img_tags}</body></html>"

    def _run_until_complete(awaitable):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        name = getattr(getattr(awaitable, "cr_code", None), "co_name", "")
        if name == "content":
            return rendered_html
        return b"poster"

    scraper.loop = SimpleNamespace(run_until_complete=_run_until_complete)

    class _FakeImage:
        def __init__(self, size):
            self.size = size

    monkeypatch.setattr(images, "Image", SimpleNamespace(open=lambda _obj: _FakeImage((1080, 1350))))

    async def _image_bytes():
        return b"poster"

    scraper.read_extract.page.request = SimpleNamespace(get=lambda _url: SimpleNamespace(status=200, body=_image_bytes))

    called_images: list[str] = []
    scraper.process_image_url = lambda url, *_args, **_kwargs: called_images.append(url)

    ImageScraper.process_webpage_url(
        scraper,
        "https://www.instagram.com/p/POST001/",
        "https://www.instagram.com/bachatavictoria/",
        "Sebastian y Hannah",
        "bachata",
    )

    assert len(called_images) == 5
