"""Regression coverage for Eventbrite search-result URL extraction."""

from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ebs import canonicalize_eventbrite_event_url, ordered_eventbrite_result_urls


def test_canonicalize_eventbrite_event_url_preserves_history_compatible_query_parameters() -> None:
    href = "https://www.eventbrite.ca/e/salsa-night-dance-city-tickets-1997389619119?aff=ebdssbdestsearch"

    assert canonicalize_eventbrite_event_url(href) == (
        "https://www.eventbrite.ca/e/salsa-night-dance-city-tickets-1997389619119?aff=ebdssbdestsearch"
    )


def test_ordered_eventbrite_result_urls_keeps_first_unique_result_cards() -> None:
    hrefs = [
        "/e/salsa-night-dance-city-tickets-1997389619119?aff=search",
        "https://www.eventbrite.com/e/salsa-night-dance-city-tickets-1997389619119?aff=duplicate",
        "https://www.eventbrite.ca/e/caliente-salsa-saturdays-tickets-1997391464639?aff=search",
        "https://www.eventbrite.ca/d/canada--victoria/events/",
        "https://www.eventbrite.com/e/bachata-social-tickets-1997000000000?aff=search",
    ]

    assert ordered_eventbrite_result_urls(hrefs, limit=2) == [
        "https://www.eventbrite.com/e/salsa-night-dance-city-tickets-1997389619119?aff=search",
        "https://www.eventbrite.ca/e/caliente-salsa-saturdays-tickets-1997391464639?aff=search",
    ]


def test_ordered_eventbrite_result_urls_rejects_non_event_links() -> None:
    assert ordered_eventbrite_result_urls(
        [
            "https://www.eventbrite.com/d/canada--victoria/events/",
            "https://example.com/e/salsa-tickets-1997389619119",
            "https://www.eventbrite.com/e/salsa-night/",
        ],
        limit=10,
    ) == []
