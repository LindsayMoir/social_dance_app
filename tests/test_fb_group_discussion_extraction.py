"""Regression coverage for Facebook group discussion extraction."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fb as fb_module
from fb import FacebookEventScraper, FacebookGroupPost, _facebook_group_post_identity


GROUP_URL = "https://www.facebook.com/groups/895697871649474/"
DISCUSSION_TEXT = (
    "Victoria Zouk Collective Group. "
    "Hello! Are you all still dancing on Wednesdays at White Eagle Hall? "
    "Yes, we are! Meet Wednesdays 6:30 to 9ish in the lower hall."
)


class _GroupLocator:
    def aria_snapshot(self, *, timeout: int) -> str:
        assert timeout == 500
        return DISCUSSION_TEXT


class _GroupPage:
    url = GROUP_URL

    def wait_for_timeout(self, _timeout: int) -> None:
        return None

    def query_selector_all(self, _selector: str) -> list[object]:
        return []

    def locator(self, selector: str) -> _GroupLocator:
        assert selector == "body"
        return _GroupLocator()

    def content(self) -> str:
        return "<html><body>Victoria Zouk Collective Group</body></html>"


class _Node:
    def __init__(self, *, href: str = "", src: str = "", text: str = "") -> None:
        self.href = href
        self.src = src
        self.text = text

    def click(self) -> None:
        return None

    def get_attribute(self, name: str) -> str:
        return self.href if name == "href" else self.src


class _Article:
    def __init__(self, text: str, href: str, image_urls: list[str]) -> None:
        self._text = text
        self._href = href
        self._images = image_urls

    def query_selector_all(self, selector: str) -> list[_Node]:
        if selector == "text=/See more/i":
            return [_Node()]
        if selector == "img":
            return [_Node(src=image_url) for image_url in self._images]
        return []

    def query_selector(self, selector: str) -> _Node | None:
        if "/posts/" in selector:
            return _Node(href=self._href)
        return None

    def inner_text(self) -> str:
        return self._text


class _DiscussionPage(_GroupPage):
    def evaluate(self, _script: str) -> None:
        return None

    def query_selector_all(self, selector: str) -> list[_Article]:
        assert selector == '[role="article"]'
        return [
            _Article(
                "Alive Tango Victoria's intermediate and beginner-friendly classes. "
                "Wednesdays 6:30pm class and 8pm practica.",
                "/groups/alivetango/posts/123456789/",
                ["https://scontent.xx.fbcdn.net/v/t39.30808-6/poster.jpg"],
            )
        ]


def _group_scraper() -> FacebookEventScraper:
    scraper = FacebookEventScraper.__new__(FacebookEventScraper)
    scraper.logged_in_page = _GroupPage()
    scraper.config = {
        "crawling": {
            "aria_snapshot_enabled": False,
            "aria_snapshot_debug_enabled": False,
            "aria_snapshot_timeout_ms": 500,
            "aria_snapshot_min_chars": 80,
            "aria_snapshot_max_chars": 4_000,
        }
    }
    scraper.fb_post_nav_wait_ms = 0
    scraper.fb_post_expand_wait_ms = 0
    scraper.fb_final_wait_ms = 0
    scraper.total_url_attempts = 0
    return scraper


def test_group_discussion_uses_rendered_accessibility_text_when_dom_omits_post() -> None:
    scraper = _group_scraper()

    extracted = scraper.extract_event_text(GROUP_URL, assume_navigated=True)

    assert extracted == DISCUSSION_TEXT


def test_group_discussion_bypasses_event_page_heading_slicer() -> None:
    scraper = _group_scraper()

    relevant = scraper.extract_relevant_text(DISCUSSION_TEXT, GROUP_URL)

    assert relevant == DISCUSSION_TEXT


def test_group_post_collection_keeps_permalink_text_and_poster_url() -> None:
    scraper = _group_scraper()
    scraper.logged_in_page = _DiscussionPage()
    scraper.config["crawling"].update(
        {
            "facebook_group_feed_scroll_depth": 1,
            "facebook_group_posts_per_run": 3,
            "facebook_group_images_per_post": 2,
        }
    )

    posts = scraper.extract_facebook_group_posts(GROUP_URL)

    assert len(posts) == 1
    assert posts[0].url == "https://www.facebook.com/groups/alivetango/posts/123456789/"
    assert "Wednesdays 6:30pm" in posts[0].text
    assert posts[0].image_urls == ("https://scontent.xx.fbcdn.net/v/t39.30808-6/poster.jpg",)


def test_group_post_identity_uses_content_hash_when_no_permalink_exists() -> None:
    identity = _facebook_group_post_identity(GROUP_URL, None, "Wednesday tango at 8 pm")

    assert identity.startswith(f"{GROUP_URL.rstrip('/')}/#discussion-post=")


def test_group_post_queues_poster_when_caption_has_no_dance_keyword(monkeypatch) -> None:
    class _Db:
        def __init__(self) -> None:
            self.rows: list[list[object]] = []

        def should_process_url(self, _url: str) -> bool:
            return True

        def write_url_to_db(self, row: list[object]) -> None:
            self.rows.append(row)

    scraper = FacebookEventScraper.__new__(FacebookEventScraper)
    scraper.keywords_list = ["tango"]
    scraper.events_written_to_db = 0
    scraper.extract_facebook_group_posts = lambda _url: [
        FacebookGroupPost(
            url="https://www.facebook.com/groups/alivetango/posts/987654321/",
            text="New schedule posted. See the attached image for details.",
            image_urls=("https://scontent.xx.fbcdn.net/v/t39.30808-6/poster.jpg",),
        )
    ]
    db = _Db()
    monkeypatch.setattr(fb_module, "db_handler", db)

    written = scraper.process_facebook_group_posts(GROUP_URL, "Alive Tango", "tango")

    assert written == 0
    assert any(row[0] == "https://scontent.xx.fbcdn.net/v/t39.30808-6/poster.jpg" for row in db.rows)
    assert any(row[-1] == "group_post_no_keywords" for row in db.rows)
