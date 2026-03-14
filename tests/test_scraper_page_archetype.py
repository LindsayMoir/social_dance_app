import sys

sys.path.insert(0, "src")

from scraper import (
    classify_page_archetype,
    extract_visible_text_from_html,
    should_extract_on_parent_page,
)


def test_extract_visible_text_from_html_removes_script_noise() -> None:
    html = """
    <html>
      <body>
        <main>
          <h1>Wednesday Jam</h1>
          <p>Wednesday March 11 at 8:00 PM</p>
        </main>
        <script>
          var related = "Thursday March 12 at The Lab Victoria";
        </script>
      </body>
    </html>
    """
    text = extract_visible_text_from_html(html)
    assert "Wednesday Jam" in text
    assert "March 11" in text
    assert "The Lab Victoria" not in text


def test_classify_page_archetype_detects_listing_page() -> None:
    links = [
        "https://lamppostsocial.com/event/a",
        "https://lamppostsocial.com/event/b",
        "https://lamppostsocial.com/events/c",
        "https://lamppostsocial.com/event/d",
        "https://lamppostsocial.com/about",
        "https://lamppostsocial.com/contact",
    ]
    archetype = classify_page_archetype(
        url="https://lamppostsocial.com/events",
        visible_text="Upcoming events. View all. Read more. Tickets available.",
        page_links=links,
        calendar_sources=[],
        calendar_ids_count=0,
    )
    assert archetype == "incomplete_event"
    assert should_extract_on_parent_page(archetype, "https://lamppostsocial.com/events") is False


def test_should_extract_on_parent_page_allows_event_detail_in_incomplete_event() -> None:
    assert (
        should_extract_on_parent_page(
            "incomplete_event",
            "https://lamppostsocial.com/event/wednesday-jam",
        )
        is True
    )
