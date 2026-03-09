#!/usr/bin/env python3
"""
Utility: scrape events from exactly one domain using scraper.py components.

This utility reuses EventSpider from src/scraper.py and constrains crawl scope
to a single domain via Scrapy's allowed_domains filter.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import yaml
from scrapy.crawler import CrawlerProcess

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from logging_config import setup_logging
from scraper import EventSpider


def _normalize_seed_url(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Domain URL is required.")
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    parsed = urlparse(value)
    if not parsed.netloc:
        raise ValueError(f"Invalid domain URL: {raw}")
    path = parsed.path or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _domain_from_url(url: str) -> str:
    host = (urlparse(url).netloc or "").strip().lower()
    if not host:
        raise ValueError(f"Could not extract domain from URL: {url}")
    return host


def _write_single_seed_files(temp_dir: Path, seed_url: str, source_name: str, keywords: str) -> None:
    header = ["source", "keywords", "link"]
    row = [source_name, keywords, seed_url]

    seeds_path = temp_dir / "single_domain_urls.csv"
    with seeds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    whitelist_path = temp_dir / "aaa_urls.csv"
    with whitelist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    calendar_path = temp_dir / "calendar_urls.csv"
    with calendar_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["link"])
        writer.writerow([seed_url])


class DomainRestrictedEventSpider(EventSpider):
    """EventSpider variant restricted to one domain."""

    name = "single_domain_event_spider"

    def __init__(self, target_domain: str, *args, **kwargs):
        self.allowed_domains = [target_domain]
        self.target_domain = target_domain
        super().__init__(*args, **kwargs)
        self.logger.info(
            "DomainRestrictedEventSpider initialized: allowed_domains=%s",
            self.allowed_domains,
        )


def run_single_domain_scrape(
    domain_url: str,
    config_path: str = "config/config.yaml",
    source_name: str | None = None,
    keywords: str = "",
    urls_run_limit: int | None = None,
) -> None:
    seed_url = _normalize_seed_url(domain_url)
    target_domain = _domain_from_url(seed_url)
    source_label = source_name or target_domain

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    with tempfile.TemporaryDirectory(prefix="single_domain_scrape_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        _write_single_seed_files(temp_dir, seed_url, source_label, keywords)

        cfg.setdefault("input", {})
        cfg.setdefault("startup", {})
        cfg.setdefault("crawling", {})
        cfg["input"]["urls"] = str(temp_dir)
        cfg["input"]["calendar_urls"] = str(temp_dir / "calendar_urls.csv")
        cfg["startup"]["use_db"] = False
        if urls_run_limit is not None:
            cfg["crawling"]["urls_run_limit"] = int(urls_run_limit)

        logging_file = "logs/single_domain_scrape_log.txt"
        process = CrawlerProcess(
            settings={
                "LOG_FILE": logging_file,
                "LOG_LEVEL": "INFO",
                "LOG_FORMAT": (
                    "%(asctime)s [%(name)s] %(levelname)s: "
                    f"[run_id={os.getenv('DS_RUN_ID', 'na')}] "
                    f"[step={os.getenv('DS_STEP_NAME', 'single_domain_scrape')}] %(message)s"
                ),
                "DEPTH_LIMIT": cfg["crawling"]["depth_limit"],
                "FEEDS": {"output/output.json": {"format": "json"}},
                "DEFAULT_REQUEST_HEADERS": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/114.0.0.0 Safari/537.36"
                    ),
                },
                "HTTPERROR_ALLOWED_CODES": [406],
                "DOWNLOAD_TIMEOUT": int(
                    cfg.get("crawling", {}).get("scraper_download_timeout_seconds", 35) or 35
                ),
                "PLAYWRIGHT_TIMEOUT": int(
                    cfg.get("crawling", {}).get("scraper_playwright_timeout_ms", 35000) or 35000
                ),
                "RETRY_ENABLED": True,
                "RETRY_TIMES": int(cfg.get("crawling", {}).get("scraper_retry_times", 1) or 1),
                "CONCURRENT_REQUESTS": int(
                    cfg.get("crawling", {}).get("scraper_concurrent_requests", 16) or 16
                ),
                "CONCURRENT_REQUESTS_PER_DOMAIN": int(
                    cfg.get("crawling", {}).get("scraper_concurrent_requests_per_domain", 8) or 8
                ),
            }
        )

        process.crawl(
            DomainRestrictedEventSpider,
            config=cfg,
            target_domain=target_domain,
        )
        process.start()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape only one domain using scraper.py methods."
    )
    parser.add_argument("domain_url", help="Domain URL to scrape (e.g., https://bcswingdance.ca)")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override source label written to seed CSV (default: domain host)",
    )
    parser.add_argument(
        "--keywords",
        default="",
        help="Optional comma-separated keywords to attach to the seed row",
    )
    parser.add_argument(
        "--urls-run-limit",
        type=int,
        default=None,
        help="Override crawling.urls_run_limit for this run",
    )
    args = parser.parse_args()

    os.environ["DS_STEP_NAME"] = "single_domain_scrape"
    setup_logging("single_domain_scrape")
    start_time = datetime.now()
    run_single_domain_scrape(
        domain_url=args.domain_url,
        config_path=args.config,
        source_name=args.source,
        keywords=args.keywords,
        urls_run_limit=args.urls_run_limit,
    )
    end_time = datetime.now()
    print(f"Single-domain scrape complete. Duration: {end_time - start_time}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
