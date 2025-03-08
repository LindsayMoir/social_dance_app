"""
scraper_crawl.py

This script runs a single crawl using Scrapy's CrawlerProcess.
It loads the configuration from config/config.yaml, instantiates an EventSpider,
and starts the crawl. This file is meant to be launched as a separate subprocess.
"""

import yaml
from scrapy.crawler import CrawlerProcess
from scraper import EventSpider  # Ensure the import path is correct

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    process = CrawlerProcess(settings={
        "LOG_FILE": config['logging']['log_file'],
        "LOG_LEVEL": "INFO",
        "DEPTH_LIMIT": config['crawling']['depth_limit'],
        "FEEDS": {
            "output/output.json": {"format": "json"}
        }
    })
    process.crawl(EventSpider, config=config)
    process.start()  # This starts and stops the reactor in this subprocess.
