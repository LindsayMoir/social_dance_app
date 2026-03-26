from __future__ import annotations

from collections import defaultdict, deque

from scrapy import Request
from twisted.internet.error import DNSLookupError, TimeoutError
from twisted.python.failure import Failure

from scraper import EventSpider


def _build_spider() -> EventSpider:
    spider = EventSpider.__new__(EventSpider)
    spider.domain_transient_failure_counts = defaultdict(int)
    spider.domain_timeout_failure_counts = defaultdict(int)
    spider.domain_exception_failure_counts = defaultdict(int)
    spider.domain_failure_events = defaultdict(deque)
    spider.scraper_priority_download_timeout_seconds = 90
    spider.scraper_priority_retry_times = 3
    spider._record_domain_transient_failure = lambda *_args, **_kwargs: None
    return spider


def test_handle_request_error_schedules_one_extended_retry_for_timeout() -> None:
    spider = _build_spider()
    request = Request(
        url="https://example.com/events",
        meta={"download_timeout": 35, "retry_times": 1},
    )
    failure = Failure(TimeoutError("timed out"))
    failure.request = request

    retry_request = spider.handle_request_error(failure)

    assert isinstance(retry_request, Request)
    assert retry_request.meta["ds_extended_retry"] is True
    assert retry_request.meta["download_timeout"] == 90
    assert retry_request.meta["retry_times"] == 3
    assert retry_request.priority == 100


def test_handle_request_error_does_not_repeat_extended_retry() -> None:
    spider = _build_spider()
    request = Request(
        url="https://example.com/events",
        meta={"ds_extended_retry": True, "download_timeout": 90, "retry_times": 3},
    )
    failure = Failure(DNSLookupError("Temporary failure in name resolution"))
    failure.request = request

    assert spider.handle_request_error(failure) is None
