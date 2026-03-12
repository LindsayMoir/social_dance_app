import sys

sys.path.insert(0, "src")

from utils.crawl_telemetry_tuning import tune_crawl_config_from_first_pass


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_tuner_reduces_scraper_pressure_when_timeout_ratio_is_high(tmp_path):
    scraper_log = tmp_path / "scraper_log.txt"
    fb_log = tmp_path / "fb_log.txt"
    _write(
        str(scraper_log),
        """
noise
scraper.py starting...
Dumping Scrapy stats:
{'downloader/request_count': 300, 'retry/reason_count/twisted.internet.error.TimeoutError': 90, 'retry/count': 110, 'retry/max_reached': 25}
Spider closed (finished)
domain_circuit_breaker_summary: triggers=7 skips=2 active_blocks=1 reason=finished
""".strip(),
    )
    _write(str(fb_log), "fb access check: state=ok")

    base_updates = {
        "crawling": {
            "scraper_download_timeout_seconds": 50,
            "scraper_retry_times": 2,
            "scraper_concurrent_requests": 8,
            "scraper_concurrent_requests_per_domain": 2,
        }
    }
    result = tune_crawl_config_from_first_pass(
        current_updates=base_updates,
        scraper_log_path=str(scraper_log),
        fb_log_path=str(fb_log),
    )

    assert result.updates["crawling"]["scraper_download_timeout_seconds"] == 40
    assert result.updates["crawling"]["scraper_retry_times"] == 1
    assert result.updates["crawling"]["scraper_concurrent_requests"] == 6
    assert result.updates["crawling"]["scraper_concurrent_requests_per_domain"] == 1


def test_tuner_speeds_up_fb_pacing_when_block_pressure_is_low(tmp_path):
    scraper_log = tmp_path / "scraper_log.txt"
    fb_log = tmp_path / "fb_log.txt"
    _write(
        str(scraper_log),
        """
scraper.py starting...
Dumping Scrapy stats:
{'downloader/request_count': 210, 'retry/reason_count/twisted.internet.error.TimeoutError': 8, 'retry/count': 11, 'retry/max_reached': 1}
Spider closed (finished)
domain_circuit_breaker_summary: triggers=0 skips=0 active_blocks=0 reason=finished
""".strip(),
    )
    fb_checks = "\n".join(["fb access check: state=ok"] * 80)
    _write(
        str(fb_log),
        f"""
{fb_checks}
write_run_statistics(): Writing run statistics for Test Run
""".strip(),
    )

    base_updates = {
        "crawling": {
            "fb_inter_request_wait_min_ms": 5000,
            "fb_inter_request_wait_max_ms": 15000,
            "fb_post_nav_wait_ms": 1800,
            "fb_post_expand_wait_ms": 900,
            "fb_final_wait_ms": 700,
            "scraper_concurrent_requests": 8,
            "scraper_concurrent_requests_per_domain": 2,
        }
    }
    result = tune_crawl_config_from_first_pass(
        current_updates=base_updates,
        scraper_log_path=str(scraper_log),
        fb_log_path=str(fb_log),
    )

    assert result.updates["crawling"]["fb_inter_request_wait_min_ms"] == 4000
    assert result.updates["crawling"]["fb_inter_request_wait_max_ms"] == 13500
    assert result.updates["crawling"]["fb_post_nav_wait_ms"] == 1500
    assert result.updates["crawling"]["fb_post_expand_wait_ms"] == 700
    assert result.updates["crawling"]["fb_final_wait_ms"] == 550
    assert result.updates["crawling"]["scraper_concurrent_requests"] == 10
    assert result.updates["crawling"]["scraper_concurrent_requests_per_domain"] == 3


def test_tuner_slows_fb_pacing_when_temp_blocks_seen(tmp_path):
    scraper_log = tmp_path / "scraper_log.txt"
    fb_log = tmp_path / "fb_log.txt"
    _write(
        str(scraper_log),
        """
scraper.py starting...
Dumping Scrapy stats:
{'downloader/request_count': 100, 'retry/reason_count/twisted.internet.error.TimeoutError': 5, 'retry/count': 8, 'retry/max_reached': 2}
Spider closed (finished)
domain_circuit_breaker_summary: triggers=1 skips=0 active_blocks=0 reason=finished
""".strip(),
    )
    _write(
        str(fb_log),
        """
fb access check: state=blocked
fb temp block policy: strike=1 url=https://www.facebook.com/events/123 action=cooldown wait_seconds=300
""".strip(),
    )

    base_updates = {
        "crawling": {
            "fb_inter_request_wait_min_ms": 5000,
            "fb_inter_request_wait_max_ms": 15000,
            "fb_post_nav_wait_ms": 1800,
            "fb_post_expand_wait_ms": 900,
            "fb_final_wait_ms": 700,
        }
    }
    result = tune_crawl_config_from_first_pass(
        current_updates=base_updates,
        scraper_log_path=str(scraper_log),
        fb_log_path=str(fb_log),
    )

    assert result.updates["crawling"]["fb_inter_request_wait_min_ms"] == 6000
    assert result.updates["crawling"]["fb_inter_request_wait_max_ms"] == 16500
    assert result.updates["crawling"]["fb_post_nav_wait_ms"] == 2100
    assert result.updates["crawling"]["fb_post_expand_wait_ms"] == 1050
    assert result.updates["crawling"]["fb_final_wait_ms"] == 850


def test_tuner_parses_scraper_stats_with_datetime_literals(tmp_path):
    scraper_log = tmp_path / "scraper_log.txt"
    fb_log = tmp_path / "fb_log.txt"
    _write(
        str(scraper_log),
        """
scraper.py starting...
Dumping Scrapy stats:
{'downloader/request_count': 477, 'retry/count': 161, 'retry/max_reached': 29,
 'retry/reason_count/twisted.internet.error.TimeoutError': 140,
 'finish_time': datetime.datetime(2026, 3, 11, 23, 34, 14, tzinfo=datetime.timezone.utc)}
Spider closed (finished)
domain_circuit_breaker_summary: triggers=5 skips=0 active_blocks=3 reason=finished
""".strip(),
    )
    _write(str(fb_log), "")

    base_updates = {
        "crawling": {
            "scraper_download_timeout_seconds": 50,
            "scraper_retry_times": 2,
            "scraper_concurrent_requests": 8,
            "scraper_concurrent_requests_per_domain": 2,
        }
    }
    result = tune_crawl_config_from_first_pass(
        current_updates=base_updates,
        scraper_log_path=str(scraper_log),
        fb_log_path=str(fb_log),
    )

    assert result.telemetry["scraper"]["request_count"] == 477
    assert result.telemetry["scraper"]["timeout_retry_count"] == 140
