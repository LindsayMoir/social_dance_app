import pandas as pd
import sys
import asyncio
import logging
from collections import defaultdict, deque

sys.path.insert(0, 'src')

import yaml
from scrapy.http import HtmlResponse, Request, Response

from scraper import (
    EventSpider,
    is_calendar_candidate,
    is_facebook_url,
    is_whitelist_candidate,
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


def test_is_facebook_url_matches_facebook_hosts_only():
    assert is_facebook_url("https://www.facebook.com/groups/1634269246863069/") is True
    assert is_facebook_url("https://m.facebook.com/events/123456/") is True
    assert is_facebook_url("https://facebook.com/events/123456/") is True
    assert is_facebook_url("https://notfacebook.com/events/123456/") is False


def test_is_whitelist_candidate_matches_root_and_subpath():
    whitelist_roots = {normalize_url_for_compare("https://latindancecanada.com/")}
    assert is_whitelist_candidate("https://latindancecanada.com/", whitelist_roots)
    assert is_whitelist_candidate("https://latindancecanada.com/group-classes/", whitelist_roots)
    assert not is_whitelist_candidate("https://example.com/events", whitelist_roots)


def test_remaining_scraper_owned_whitelist_roots_excludes_fb_owned():
    spider = EventSpider.__new__(EventSpider)
    spider.whitelist_roots = {
        normalize_url_for_compare("https://latindancecanada.com/"),
        normalize_url_for_compare("https://www.facebook.com/groups/123456/"),
    }
    spider.attempted_whitelist_roots = {normalize_url_for_compare("https://latindancecanada.com/")}
    spider.whitelist_transferred_to_fb_roots = {normalize_url_for_compare("https://www.facebook.com/groups/123456/")}

    assert spider._remaining_scraper_owned_whitelist_roots() == set()


def test_parse_marks_whitelist_non_text_response(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, _url: str) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.whitelist_roots = {normalize_url_for_compare("https://www.instagram.com/bachatavictoria/")}
    spider.attempted_whitelist_roots = set()
    spider.whitelist_transferred_to_fb_roots = set()
    spider.whitelist_non_text_response_roots = set()

    response = Response(
        url="https://www.instagram.com/bachatavictoria/",
        request=Request(url="https://www.instagram.com/bachatavictoria/"),
    )

    list(
        spider.parse(
            response,
            keywords="bachata",
            source="Sebastian y Hannah",
            url="https://www.instagram.com/bachatavictoria/",
        )
    )

    root = normalize_url_for_compare("https://www.instagram.com/bachatavictoria/")
    assert root in spider.attempted_whitelist_roots
    assert root in spider.whitelist_non_text_response_roots


def test_parse_records_text_extracted_metric_for_html_response(monkeypatch):
    import scraper as scraper_module

    metrics_written = []
    url_rows = []

    class DummyDB:
        def is_whitelisted_url(self, _url: str) -> bool:
            return False

        def maybe_reuse_static_event_detail_from_history(self, url: str):
            return {"reused": False}

        def write_url_to_db(self, row):
            url_rows.append(row)

        def write_url_scrape_metric(self, row):
            metrics_written.append(row)

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return True

    class DummyLLM:
        def process_llm_response(self, *args, **kwargs):
            return None

    class _Classification:
        archetype = "simple_page"
        owner_step = "scraper.py"
        subtype = ""

    class _Decision:
        classification = _Classification()
        confidence = 0.95
        stage = "rule"
        features = {}

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)
    monkeypatch.setattr(scraper_module, "llm_handler", DummyLLM(), raising=False)
    monkeypatch.setattr(scraper_module, "classify_page_with_confidence", lambda **kwargs: _Decision())
    monkeypatch.setattr(scraper_module, "extract_html_features", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(scraper_module, "should_extract_on_parent_page", lambda *args, **kwargs: False)

    spider = EventSpider.__new__(EventSpider)
    spider.whitelist_roots = set()
    spider.attempted_whitelist_roots = set()
    spider.whitelist_transferred_to_fb_roots = set()
    spider.whitelist_non_text_response_roots = set()
    spider.visited_link = set()
    spider.calendar_urls_set = set()
    spider.keywords_list = ["salsa"]
    spider.page_archetype_stats = {}
    spider.config = {"crawling": {"max_website_urls": 5, "urls_run_limit": 100}}
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.scraper_domain_failure_window_seconds = 600
    spider.scraper_domain_cooldown_seconds = 600
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    response = HtmlResponse(
        url="https://example.com/events",
        body=b"<html><body><h1>Live music tonight</h1><p>Doors at 8pm.</p></body></html>",
        encoding="utf-8",
        request=Request(url="https://example.com/events"),
    )

    list(
        spider.parse(
            response,
            keywords="live music",
            source="Example Source",
            url="https://example.com/events",
        )
    )

    assert url_rows
    assert metrics_written
    assert metrics_written[0]["access_attempted"] is True
    assert metrics_written[0]["access_succeeded"] is True
    assert metrics_written[0]["text_extracted"] is True
    assert metrics_written[0]["keywords_found"] is False


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


def test_parse_extracts_calendar_id_from_rendered_page_text(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, _url: str) -> bool:
            return False

        def get_historical_classifier_memory(self, _url: str):
            return None

        def maybe_reuse_static_event_detail_from_history(self, *, url: str):
            return {"reused": False}

        def write_url_to_db(self, _row):
            return None

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

    class DummyLLM:
        def process_llm_response(self, *args, **kwargs) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)
    monkeypatch.setattr(scraper_module, "llm_handler", DummyLLM(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.keywords_list = ["dance"]
    spider.config = {"crawling": {"max_website_urls": 10, "urls_run_limit": 100}}
    spider.calendar_urls_set = set()
    spider.whitelist_roots = set()
    spider.attempted_whitelist_roots = set()
    spider.whitelist_transferred_to_fb_roots = set()
    spider.whitelist_non_text_response_roots = set()
    spider.visited_link = set()
    spider.page_archetype_stats = {}
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    processed_ids = []

    def capture_calendar_id(calendar_id, _calendar_url, _url, _source, _keywords):
        processed_ids.append(calendar_id)

    monkeypatch.setattr(spider, "process_calendar_id", capture_calendar_id)
    monkeypatch.setattr(spider, "fetch_google_calendar_events", lambda *args, **kwargs: None)

    html = """
    <html>
      <body>
        <div>Dance community resources</div>
        <script>
          hycal_render_calendar({"ics":"https:\\/\\/calendar.google.com\\/calendar\\/ical\\/
          17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com
          \\/public\\/basic.ics"})
        </script>
      </body>
    </html>
    """
    response = HtmlResponse(
        url="https://vlda.ca/resources/",
        body=html,
        encoding="utf-8",
        request=Request(url="https://vlda.ca/resources/"),
    )

    list(spider.parse(response, keywords="salsa", source="Victoria Latin Dance Association", url="https://vlda.ca/resources/"))

    assert "17de2ca43f525c87058ffafe1f0206da82a19d00a85a6ae979ad6bb618dcd8ae@group.calendar.google.com" in processed_ids


def test_parse_extracts_calendar_id_from_raw_html_attributes(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, _url: str) -> bool:
            return False

        def get_historical_classifier_memory(self, _url: str):
            return None

        def maybe_reuse_static_event_detail_from_history(self, *, url: str):
            return {"reused": False}

        def write_url_to_db(self, _row):
            return None

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

    class DummyLLM:
        def process_llm_response(self, *args, **kwargs) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)
    monkeypatch.setattr(scraper_module, "llm_handler", DummyLLM(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.keywords_list = ["dance"]
    spider.config = {"crawling": {"max_website_urls": 10, "urls_run_limit": 100}}
    spider.calendar_urls_set = set()
    spider.whitelist_roots = set()
    spider.attempted_whitelist_roots = set()
    spider.whitelist_transferred_to_fb_roots = set()
    spider.whitelist_non_text_response_roots = set()
    spider.visited_link = set()
    spider.page_archetype_stats = {}
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    processed_ids = []

    def capture_calendar_id(calendar_id, _calendar_url, _url, _source, _keywords):
        processed_ids.append(calendar_id)

    monkeypatch.setattr(spider, "process_calendar_id", capture_calendar_id)
    monkeypatch.setattr(spider, "fetch_google_calendar_events", lambda *args, **kwargs: None)

    html = """
    <html>
      <body>
        <div>Dance community resources</div>
        <div data-calendar="https://calendar.google.com/calendar/ical/abc123%40group.calendar.google.com/public/basic.ics"></div>
      </body>
    </html>
    """
    response = HtmlResponse(
        url="https://vlda.ca/resources/",
        body=html,
        encoding="utf-8",
        request=Request(url="https://vlda.ca/resources/"),
    )

    list(spider.parse(response, keywords="salsa", source="Victoria Latin Dance Association", url="https://vlda.ca/resources/"))

    assert "abc123@group.calendar.google.com" in processed_ids


def test_parse_does_not_fetch_plain_calendar_root_links(monkeypatch):
    import scraper as scraper_module

    class DummyDB:
        def is_whitelisted_url(self, _url: str) -> bool:
            return False

        def get_historical_classifier_memory(self, _url: str):
            return None

        def maybe_reuse_static_event_detail_from_history(self, *, url: str):
            return {"reused": False}

        def write_url_to_db(self, _row):
            return None

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

    class DummyLLM:
        def process_llm_response(self, *args, **kwargs) -> bool:
            return False

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)
    monkeypatch.setattr(scraper_module, "llm_handler", DummyLLM(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.keywords_list = ["dance"]
    spider.config = {"crawling": {"max_website_urls": 10, "urls_run_limit": 100}}
    spider.calendar_urls_set = {normalize_url_for_compare("https://vlda.ca/resources/")}
    spider.whitelist_roots = set()
    spider.attempted_whitelist_roots = set()
    spider.whitelist_transferred_to_fb_roots = set()
    spider.whitelist_non_text_response_roots = set()
    spider.visited_link = set()
    spider.page_archetype_stats = {}
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    fetched_calendar_urls = []

    monkeypatch.setattr(spider, "process_calendar_id", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        spider,
        "fetch_google_calendar_events",
        lambda calendar_url, *_args, **_kwargs: fetched_calendar_urls.append(calendar_url),
    )

    html = """
    <html>
      <body>
        <a href="/resources/">Resources</a>
      </body>
    </html>
    """
    response = HtmlResponse(
        url="https://vlda.ca/about/",
        body=html,
        encoding="utf-8",
        request=Request(url="https://vlda.ca/about/"),
    )

    list(spider.parse(response, keywords="salsa", source="Victoria Latin Dance Association", url="https://vlda.ca/about/"))

    assert fetched_calendar_urls == []


def test_start_whitelist_seed_bypasses_history_gate(monkeypatch, tmp_path):
    import scraper as scraper_module

    seed_csv = tmp_path / "seed.csv"
    seed_csv.write_text(
        "source,keywords,link\n"
        "Test Source,salsa,https://latindancecanada.com/\n"
        "Other Source,swing,https://example.com/events\n",
        encoding="utf-8",
    )

    class DummyDB:
        def get_db_connection(self):
            return object()

        def is_whitelisted_url(self, url: str) -> bool:
            return "latindancecanada.com" in url

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

        def write_url_to_db(self, _row):
            return None

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.config = {
        "startup": {"use_db": False},
        "input": {"urls": str(tmp_path)},
    }
    spider.whitelist_urls_df = pd.DataFrame(
        [{"source": "Whitelist", "keywords": "salsa", "link": "https://latindancecanada.com/"}]
    )
    spider.whitelist_roots = {normalize_url_for_compare("https://latindancecanada.com/")}
    spider.attempted_whitelist_roots = set()
    spider.calendar_urls_set = set()
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    async def _collect():
        reqs = []
        async for req in spider.start():
            reqs.append(req)
        return reqs

    requests = asyncio.run(_collect())
    start_urls = [r.url for r in requests]
    assert "https://latindancecanada.com/" in start_urls
    assert "https://example.com/events" not in start_urls


def test_start_sets_high_priority_for_whitelist_seed(monkeypatch, tmp_path):
    import scraper as scraper_module

    seed_csv = tmp_path / "seed.csv"
    seed_csv.write_text(
        "source,keywords,link\n"
        "Test Source,salsa,https://latindancecanada.com/\n",
        encoding="utf-8",
    )

    class DummyDB:
        def get_db_connection(self):
            return object()

        def is_whitelisted_url(self, _url: str) -> bool:
            return True

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

        def write_url_to_db(self, _row):
            return None

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.config = {
        "startup": {"use_db": False},
        "input": {"urls": str(tmp_path)},
    }
    spider.whitelist_urls_df = pd.DataFrame(
        [{"source": "Whitelist", "keywords": "salsa", "link": "https://latindancecanada.com/"}]
    )
    spider.whitelist_roots = {normalize_url_for_compare("https://latindancecanada.com/")}
    spider.attempted_whitelist_roots = set()
    spider.calendar_urls_set = set()
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    async def _collect():
        reqs = []
        async for req in spider.start():
            reqs.append(req)
        return reqs

    requests = asyncio.run(_collect())
    assert requests
    assert requests[0].url == "https://latindancecanada.com/"
    assert requests[0].priority == 1000


def test_start_logs_loaded_seed_csv_files(monkeypatch, tmp_path, caplog):
    import scraper as scraper_module

    (tmp_path / "gs_urls.csv").write_text(
        "source,keywords,link\n"
        "GS Source,salsa,https://example.com/events\n",
        encoding="utf-8",
    )
    (tmp_path / "calendar_urls.csv").write_text(
        "source,keywords,link\n"
        "Calendar Source,swing,https://calendar.example.com/\n",
        encoding="utf-8",
    )

    class DummyDB:
        def get_db_connection(self):
            return object()

        def is_whitelisted_url(self, _url: str) -> bool:
            return False

        def avoid_domains(self, _url: str) -> bool:
            return False

        def should_process_url(self, _url: str) -> bool:
            return False

        def write_url_to_db(self, _row):
            return None

    monkeypatch.setattr(scraper_module, "db_handler", DummyDB(), raising=False)

    spider = EventSpider.__new__(EventSpider)
    spider.config = {
        "startup": {"use_db": False},
        "input": {"urls": str(tmp_path)},
    }
    spider.whitelist_urls_df = pd.DataFrame(columns=["source", "keywords", "link"])
    spider.whitelist_roots = set()
    spider.attempted_whitelist_roots = set()
    spider.calendar_urls_set = set()
    spider.domain_failure_events = defaultdict(deque)
    spider.domain_cooldown_until = {}
    spider.domain_cooldown_skip_count = 0
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3

    async def _collect():
        reqs = []
        async for req in spider.start():
            reqs.append(req)
        return reqs

    with caplog.at_level(logging.INFO):
        asyncio.run(_collect())

    assert "Loaded 2 URL seed CSV files" in caplog.text
    assert "gs_urls.csv=1" in caplog.text
    assert "calendar_urls.csv=1" in caplog.text
