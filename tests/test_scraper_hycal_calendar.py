import sys

sys.path.insert(0, "src")

from scraper import EventSpider, extract_hycal_proxy_links, max_links_to_follow_for_page


def test_extract_hycal_proxy_links_from_html() -> None:
    html = """
    <html><body>
      <script>
        const u = "/wp-json/hycal/v1/ics-proxy?url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2Fabc123%2540group.calendar.google.com%2Fpublic%2Fbasic.ics";
      </script>
    </body></html>
    """
    links = extract_hycal_proxy_links("https://vlda.ca/resources/", html)
    assert len(links) == 1
    assert links[0].startswith("https://vlda.ca/wp-json/hycal/v1/ics-proxy?url=")


def test_extract_hycal_proxy_links_supports_alternate_versions_and_rest_route() -> None:
    html = """
    <html><body>
      <a href="/wp-json/hycal/v2/ics-proxy?url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2Fabc%2540group.calendar.google.com%2Fpublic%2Fbasic.ics">v2</a>
      <a href="/index.php?rest_route=%2Fhycal%2Fv3%2Fics-proxy&url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2Fdef%2540group.calendar.google.com%2Fpublic%2Fbasic.ics">rest</a>
    </body></html>
    """
    links = extract_hycal_proxy_links("https://example.org/resources/", html)
    assert any("/wp-json/hycal/v2/ics-proxy" in link for link in links)
    assert any("rest_route=%2Fhycal%2Fv3%2Fics-proxy" in link for link in links)


def test_expand_calendar_url_candidates_unwraps_hycal_url() -> None:
    spider = EventSpider.__new__(EventSpider)
    hycal = (
        "https://vlda.ca/wp-json/hycal/v1/ics-proxy"
        "?url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2F"
        "17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae%40group.calendar.google.com"
        "%2Fpublic%2Fbasic.ics"
    )
    candidates = spider._expand_calendar_url_candidates(hycal)
    assert any("wp-json/hycal/v1/ics-proxy" in c for c in candidates)
    assert any("calendar.google.com/calendar/ical/" in c for c in candidates)


def test_expand_calendar_url_candidates_unwraps_rest_route_hycal_url() -> None:
    spider = EventSpider.__new__(EventSpider)
    hycal = (
        "https://example.org/index.php?rest_route=%2Fhycal%2Fv3%2Fics-proxy"
        "&url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2F"
        "xyz123%40group.calendar.google.com%2Fpublic%2Fbasic.ics"
    )
    candidates = spider._expand_calendar_url_candidates(hycal)
    assert any("rest_route=" in c for c in candidates)
    assert any("calendar.google.com/calendar/ical/xyz123@group.calendar.google.com/public/basic.ics" in c for c in candidates)


def test_extract_calendar_ids_from_hycal_proxy_url() -> None:
    spider = EventSpider.__new__(EventSpider)
    hycal = (
        "https://vlda.ca/wp-json/hycal/v1/ics-proxy"
        "?url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2F"
        "17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae%40group.calendar.google.com"
        "%2Fpublic%2Fbasic.ics"
    )
    ids = spider.extract_calendar_ids(hycal, allow_gmail=True)
    assert ids == ["17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com"]


def test_extract_calendar_ids_from_hycal_rest_route_proxy_url() -> None:
    spider = EventSpider.__new__(EventSpider)
    hycal = (
        "https://example.org/index.php?rest_route=%2Fhycal%2Fv3%2Fics-proxy"
        "&url=https%3A%2F%2Fcalendar.google.com%2Fcalendar%2Fical%2F"
        "xyz123%40group.calendar.google.com%2Fpublic%2Fbasic.ics"
    )
    ids = spider.extract_calendar_ids(hycal, allow_gmail=True)
    assert ids == ["xyz123@group.calendar.google.com"]


def test_max_links_to_follow_for_page_reduces_low_confidence_listing_pages() -> None:
    assert max_links_to_follow_for_page(
        "simple_page",
        confidence=0.55,
        base_limit=10,
        url="https://example.com/about/",
        is_whitelisted_origin=False,
        is_calendar_root=False,
    ) == 3


def test_max_links_to_follow_for_page_preserves_priority_roots() -> None:
    assert max_links_to_follow_for_page(
        "simple_page",
        confidence=0.40,
        base_limit=10,
        url="https://vlda.ca/resources/",
        is_whitelisted_origin=True,
        is_calendar_root=False,
    ) == 10
