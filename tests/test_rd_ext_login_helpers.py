import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rd_ext import (
    _extract_candidate_calendar_detail_cta_urls,
    _extract_revealed_calendar_event_blocks,
    _build_synthetic_calendar_detail_url,
    _derive_rd_ext_effective_keywords,
    _get_login_probe_url,
    _is_embedded_event_iframe_url,
    _looks_like_calendar_detail_cta,
    _looks_like_event_iframe_text,
    _looks_like_clickable_calendar_box_label,
    _looks_like_child_event_url,
    _rd_ext_keywords_found,
    _should_follow_layered_calendar_detail_url,
    _should_attempt_content_reveal_click,
    _should_follow_same_domain_child_link,
    ReadExtract,
    _wait_for_login_completion,
)


class _FakePage:
    def __init__(self, urls, selector_states):
        self._urls = list(urls)
        self._selector_states = list(selector_states)
        self.url = self._urls[0]
        self._index = 0

    async def query_selector(self, selector):
        state = self._selector_states[min(self._index, len(self._selector_states) - 1)]
        return object() if state.get(selector, False) else None

    async def wait_for_timeout(self, _ms):
        if self._index < len(self._urls) - 1:
            self._index += 1
            self.url = self._urls[self._index]


def test_wait_for_login_completion_succeeds_after_redirect() -> None:
    page = _FakePage(
        urls=[
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/organizer/home/",
        ],
        selector_states=[
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": False},
            {"input#email": False, "input#password": False},
        ],
    )

    result = asyncio.run(
        _wait_for_login_completion(
            page,
            login_url="https://www.eventbrite.com/signin/",
            email_selector="input#email",
            pass_selector="input#password",
            timeout_ms=5000,
        )
    )

    assert result is True


def test_wait_for_login_completion_fails_when_login_controls_persist() -> None:
    page = _FakePage(
        urls=[
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
        ],
        selector_states=[
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": True},
        ],
    )

    result = asyncio.run(
        _wait_for_login_completion(
            page,
            login_url="https://www.eventbrite.com/signin/",
            email_selector="input#email",
            pass_selector="input#password",
            timeout_ms=1500,
        )
    )

    assert result is False


def test_get_login_probe_url_uses_authenticated_eventbrite_target() -> None:
    assert (
        _get_login_probe_url("Eventbrite", "https://www.eventbrite.com/signin/")
        == "https://www.eventbrite.com/organizer/home/"
    )
    assert (
        _get_login_probe_url("Facebook", "https://www.facebook.com/login")
        == "https://www.facebook.com/login"
    )


def test_derive_rd_ext_effective_keywords_prefers_text_evidence() -> None:
    keywords = "live music, swing, west coast swing, wcs"
    text = "Join us for West Coast Swing lessons and social dancing before the band."

    effective = _derive_rd_ext_effective_keywords(text, keywords)

    assert effective == ["swing", "west coast swing"]


def test_derive_rd_ext_effective_keywords_falls_back_to_live_music() -> None:
    keywords = "live music, swing, west coast swing, wcs"
    text = "Doors at 7pm. Featuring a local indie rock band and dinner specials."

    effective = _derive_rd_ext_effective_keywords(text, keywords)

    assert effective == ["live music"]


def test_rd_ext_keywords_found_counts_live_music_fallback() -> None:
    assert _rd_ext_keywords_found(["live music"]) is True


def test_should_follow_same_domain_child_link_skips_navigation_pages() -> None:
    base_url = "https://www.theemporia.ca/events"

    assert not _should_follow_same_domain_child_link(
        base_url, "https://www.theemporia.ca/menu"
    )
    assert not _should_follow_same_domain_child_link(
        base_url, "https://www.theemporia.ca/reservations"
    )
    assert not _should_follow_same_domain_child_link(
        base_url, "https://www.theemporia.ca/catering"
    )


def test_should_follow_same_domain_child_link_keeps_event_candidates() -> None:
    base_url = "https://www.theemporia.ca/events"

    assert _should_follow_same_domain_child_link(
        base_url, "https://www.theemporia.ca/events/friday-night-dance/"
    )
    assert _should_follow_same_domain_child_link(
        base_url, "https://www.theemporia.ca/live-band-april-12/"
    )


def test_should_attempt_content_reveal_click_only_for_known_interactive_pages() -> None:
    assert _should_attempt_content_reveal_click("https://www.theemporia.ca/events")
    assert _should_attempt_content_reveal_click("https://www.theemporia.ca/events/")
    assert not _should_attempt_content_reveal_click("https://www.theemporia.ca/menu")
    assert not _should_attempt_content_reveal_click("https://loftpubvictoria.com/events/month/")


def test_is_embedded_event_iframe_url_identifies_boomte_calendar() -> None:
    assert _is_embedded_event_iframe_url("https://calendar.boomte.ch/widget?pageId=za8dj")
    assert _is_embedded_event_iframe_url("https://example.com/widget?currentRoute=.%2Fevents")
    assert not _is_embedded_event_iframe_url("https://www.theemporia.ca/events")


def test_looks_like_child_event_url_accepts_single_event_patterns() -> None:
    assert _looks_like_child_event_url("https://calendar.boomte.ch/single/abc123")
    assert _looks_like_child_event_url("https://thedukesaloon.com/event/backcountry-fridays/")
    assert not _looks_like_child_event_url("https://www.theemporia.ca/menu")


def test_looks_like_event_iframe_text_requires_event_and_date_signals() -> None:
    rich_event_text = (
        "Agenda\nHeather Ferguson\nThursday, 02 April\nDescription\n"
        "Add to Calendar\nShare this Event"
    )
    assert _looks_like_event_iframe_text(rich_event_text)
    assert not _looks_like_event_iframe_text("Embedded newsletter signup widget")


def test_looks_like_clickable_calendar_box_label_filters_calendar_ui_noise() -> None:
    assert _looks_like_clickable_calendar_box_label("Miguelito Valdes + band")
    assert _looks_like_clickable_calendar_box_label("Friday Night Live with DJ Rye")
    assert not _looks_like_clickable_calendar_box_label("Month")
    assert not _looks_like_clickable_calendar_box_label("Apr 21")
    assert not _looks_like_clickable_calendar_box_label("12")


def test_looks_like_calendar_detail_cta_identifies_detail_navigation() -> None:
    assert _looks_like_calendar_detail_cta("Learn more", "")
    assert _looks_like_calendar_detail_cta("Event Details", "")
    assert _looks_like_calendar_detail_cta("", "https://www.bardandbanker.com/live-music/miguelito-valdes")
    assert not _looks_like_calendar_detail_cta("Buy tickets", "https://example.com/tickets")
    assert not _looks_like_calendar_detail_cta("Add to Calendar", "https://calendar.google.com")


def test_extract_candidate_calendar_detail_cta_urls_prefers_detail_links() -> None:
    html = """
    <div class="tribe-events-calendar-month__calendar-event-details">
      <a href="/live-music/miguelito-valdes-band">Learn more</a>
      <a href="/tickets">Buy tickets</a>
      <a href="https://calendar.google.com/calendar/event?eid=abc">Add to Calendar</a>
    </div>
    """
    urls = _extract_candidate_calendar_detail_cta_urls(
        html,
        "https://www.bardandbanker.com/live-music",
    )

    assert urls == ["https://www.bardandbanker.com/live-music/miguelito-valdes-band"]


def test_extract_candidate_calendar_detail_cta_urls_requires_explicit_detail_text_when_requested() -> None:
    html = """
    <div class="tribe-events-calendar-month__calendar-event-details">
      <a href="/live-music/miguelito-valdes-band">Miguelito Valdes + band</a>
      <a href="https://www.opentable.com/r/bard-and-banker">Reserve</a>
    </div>
    """
    urls = _extract_candidate_calendar_detail_cta_urls(
        html,
        "https://www.bardandbanker.com/live-music",
        require_explicit_text=True,
    )

    assert urls == []


def test_extract_revealed_calendar_event_blocks_pairs_local_detail_ctas() -> None:
    html = """
    <div class="fc-popover">
      <div class="fc-event">
        <div>Miguelito Valdes + band</div>
        <a href="/live-music/miguelito-valdes-band">Learn more</a>
      </div>
      <div class="fc-event">
        <div>St. Cecilia</div>
        <a href="/live-music/st-cecilia">Learn more</a>
      </div>
    </div>
    """
    blocks = _extract_revealed_calendar_event_blocks(
        html,
        "https://www.bardandbanker.com/live-music",
    )

    assert len(blocks) == 2
    assert blocks[0]["detail_url"] == "https://www.bardandbanker.com/live-music/miguelito-valdes-band"
    assert "Miguelito" in blocks[0]["label"]
    assert blocks[1]["detail_url"] == "https://www.bardandbanker.com/live-music/st-cecilia"


def test_extract_revealed_calendar_event_blocks_ignores_booking_links_without_local_learn_more() -> None:
    html = """
    <div class="fc-popover">
      <div class="fc-event">
        <div>Miguelito Valdes + band</div>
        <a href="https://www.opentable.com/r/bard-and-banker">Reserve</a>
      </div>
    </div>
    """
    blocks = _extract_revealed_calendar_event_blocks(
        html,
        "https://www.bardandbanker.com/live-music",
    )

    assert blocks == []


def test_extract_revealed_calendar_event_blocks_keeps_sufficient_clicked_panel_without_learn_more() -> None:
    html = """
    <div class="fc-popover">
      <div class="fc-event">
        <div>Miguelito Valdes + band</div>
        <div>Friday, April 25 8:30 PM</div>
        <div>Bard and Banker, 1022 Government St</div>
        <div>Live music in the lounge.</div>
      </div>
    </div>
    """
    blocks = _extract_revealed_calendar_event_blocks(
        html,
        "https://www.bardandbanker.com/live-music",
    )

    assert len(blocks) == 1
    assert blocks[0]["detail_url"] == ""
    assert "Miguelito" in blocks[0]["label"]


def test_should_follow_layered_calendar_detail_url_uses_shared_url_filters(monkeypatch) -> None:
    class _FakeDb:
        @staticmethod
        def avoid_domains(url: str) -> bool:
            return "opentable.com" in url

        @staticmethod
        def should_process_url(url: str) -> bool:
            return "skip-me.com" not in url

    monkeypatch.setattr("rd_ext.get_db_handler", lambda: _FakeDb())

    assert _should_follow_layered_calendar_detail_url("https://www.bardandbanker.com/live-music/miguelito") is True
    assert _should_follow_layered_calendar_detail_url("https://www.opentable.com/r/bard-and-banker") is False
    assert _should_follow_layered_calendar_detail_url("https://skip-me.com/event") is False


def test_build_synthetic_calendar_detail_url_is_stable_and_readable() -> None:
    url = _build_synthetic_calendar_detail_url(
        "https://www.bardandbanker.com/live-music",
        "Miguelito Valdes + band",
        3,
    )
    assert url.startswith("https://www.bardandbanker.com/live-music?calendar_click=")
    assert "#miguelito-valdes-band" in url


def test_extract_from_url_routes_bard_listing_to_calendar_event_extraction(monkeypatch) -> None:
    extractor = ReadExtract.__new__(ReadExtract)
    extractor.config = {}

    class _FakeDb:
        @staticmethod
        def should_process_url(_url: str) -> bool:
            return True

    async def _fake_extract_calendar_events(url: str, venue_name: str = "Calendar"):
        assert url == "https://www.bardandbanker.com/live-music"
        assert venue_name == "Bard and Banker"
        return [
            ("https://example.com/event-1", "Event One", 3),
            ("https://example.com/event-2", "Event Two", 3),
        ]

    monkeypatch.setattr("rd_ext.get_db_handler", lambda: _FakeDb())
    monkeypatch.setattr(extractor, "extract_calendar_events", _fake_extract_calendar_events)

    result = asyncio.run(
        extractor.extract_from_url("https://www.bardandbanker.com/live-music", multiple=False)
    )

    assert result == {
        "https://example.com/event-1": "Event One",
        "https://example.com/event-2": "Event Two",
    }
