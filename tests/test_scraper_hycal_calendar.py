import sys

sys.path.insert(0, "src")

from scraper import EventSpider, extract_hycal_proxy_links


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
