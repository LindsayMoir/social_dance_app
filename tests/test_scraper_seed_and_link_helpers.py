import pandas as pd
import sys

sys.path.insert(0, 'src')

import yaml

from scraper import (
    EventSpider,
    is_calendar_candidate,
    merge_seed_urls,
    normalize_http_links,
    normalize_url_for_compare,
    prioritize_links_for_crawl,
    should_force_follow_link,
)


def test_normalize_http_links_keeps_relative_and_protocol_relative_urls():
    links = normalize_http_links(
        base_url="https://vlda.ca/resources/",
        raw_links=[
            "/calendar",
            "//calendar.google.com/calendar/embed?src=test%40group.calendar.google.com",
            "mailto:info@example.com",
            "javascript:void(0)",
            "https://example.com/events",
            "/calendar",  # duplicate
        ],
    )

    assert "https://vlda.ca/calendar" in links
    assert "https://calendar.google.com/calendar/embed?src=test%40group.calendar.google.com" in links
    assert "https://example.com/events" in links
    assert len([u for u in links if u == "https://vlda.ca/calendar"]) == 1


def test_is_calendar_candidate_matches_google_and_seed_roots():
    calendar_roots = {normalize_url_for_compare("https://vlda.ca/resources/")}
    assert is_calendar_candidate("https://calendar.google.com/calendar/embed?src=abc", calendar_roots)
    assert is_calendar_candidate("https://vlda.ca/resources/calendar-feed", calendar_roots)
    assert not is_calendar_candidate("https://example.com/events", calendar_roots)


def test_merge_seed_urls_always_includes_whitelist_and_dedups():
    seed_df = pd.DataFrame(
        [
            {"source": "DB Source", "keywords": "salsa", "link": "https://vlda.ca/resources/"},
            {"source": "DB Source 2", "keywords": "swing", "link": "https://example.com/events"},
        ]
    )
    whitelist_df = pd.DataFrame(
        [
            {"source": "Whitelist", "keywords": "salsa", "link": "https://vlda.ca/resources"},
            {"source": "Whitelist", "keywords": "latin", "link": "https://latindancecanada.com/"},
        ]
    )

    merged = merge_seed_urls(seed_df, whitelist_df)
    links = set(merged["link"].tolist())

    assert "https://example.com/events" in links
    assert "https://latindancecanada.com/" in links
    assert len(merged[merged["link"].str.contains("vlda.ca/resources", na=False)]) == 1


def test_should_force_follow_link_for_same_domain_from_whitelisted_parent(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, url: str) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    assert should_force_follow_link(
        parent_url="https://vlda.ca/resources/",
        parent_is_whitelisted=True,
        link="https://vlda.ca/events/fall-event-calendar-2025/",
        calendar_roots=set(),
    ) is True


def test_should_force_follow_link_false_for_non_calendar_other_domain(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, url: str) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    assert should_force_follow_link(
        parent_url="https://vlda.ca/resources/",
        parent_is_whitelisted=True,
        link="https://example.com/unrelated",
        calendar_roots=set(),
    ) is False


def test_prioritize_links_for_crawl_prefers_event_calendar_links():
    links = [
        "https://vlda.ca/about/",
        "https://vlda.ca/contact/",
        "https://vlda.ca/events/fall-event-calendar-2025/",
        "https://vlda.ca/resources/",
        "https://vlda.ca/events/",
    ]
    top = prioritize_links_for_crawl(links, 2)
    assert "https://vlda.ca/events/fall-event-calendar-2025/" in top
    assert "https://vlda.ca/events/" in top


def test_extract_calendar_ids_supports_ics_hycal_format():
    with open('config/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    spider = EventSpider.__new__(EventSpider)

    sample = (
        'document.addEventListener("DOMContentLoaded",function(){'
        'hycal_render_calendar({"ics":"https:\\/\\/calendar.google.com\\/calendar\\/ical\\/'
        '17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com'
        '\\/public\\/basic.ics"})})'
    )
    ids = spider.extract_calendar_ids(sample)
    assert "17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com" in ids
