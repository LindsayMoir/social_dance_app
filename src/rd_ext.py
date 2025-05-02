import asyncio
import json  # New import for JSON parsing
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import pandas as pd
import os
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
import random
import yaml

from db import DatabaseHandler
from llm import LLMHandler
from credentials import get_credentials  # Import the utility function


class ReadExtract:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.logged_in = False

    async def init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.config['crawling']['headless'])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    def extract_text_with_playwright(self, url):
        """
        Synchronously extracts the text content from a web page using Playwright.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config['crawling']['headless'])
                page = browser.new_page()
                page.goto(url, timeout=10000)
                page.wait_for_timeout(3000)

                # Check for Google sign-in page
                if "accounts.google.com" in page.url or page.is_visible("input#identifierId"):
                    logging.info("Google sign-in detected. Aborting extraction.")
                    browser.close()
                    return None

                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                text = ' '.join(soup.stripped_strings)
                browser.close()
                return text
        except Exception as e:
            logging.error(f"extract_text_with_playwright(): Failed for {url}: {e}")
            return None

    async def login_if_required(self, url: str) -> bool:
        """
        Checks URL and performs login if needed for known domains.
        """
        url_lower = url.lower()
        if "facebook.com" in url_lower:
            return await self.login_to_facebook("Facebook")
        if "google" in url_lower:
            return await self.login_to_website(
                organization="Google",
                login_url="https://accounts.google.com/signin",
                email_selector="input[type='email']",
                pass_selector="input[type='password']",
                submit_selector="button[type='submit']"
            )
        if "allevents" in url_lower:
            return await self.login_to_website(
                organization="Allevents",
                login_url="https://www.allevents.in/login",
                email_selector="input[name='email']",
                pass_selector="input[name='password']",
                submit_selector="button[type='submit']"
            )
        if "eventbrite" in url_lower:
            return await self.login_to_website(
                organization="Eventbrite",
                login_url="https://www.eventbrite.com/signin/",
                email_selector="input#email",
                pass_selector="input#password",
                submit_selector="button[type='submit']"
            )
        return True

    async def extract_event_text(self, link: str, max_retries: int = 3) -> str:
        """
        Extracts text from a single event page with retries.
        """
        if not await self.login_if_required(link):
            logging.error(f"Login failed for {link}")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                await self.page.goto(link, timeout=15000)
                await self.page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(random.uniform(3, 6))

                content = await self.page.content()
                soup = BeautifulSoup(content, 'html.parser')
                text = ' '.join(soup.stripped_strings)

                if text.strip():
                    logging.info(f"extract_event_text(): Success for {link} on attempt {attempt}")
                    return text
                logging.warning(f"extract_event_text(): Empty on attempt {attempt} for {link}")
            except Exception as e:
                logging.error(f"extract_event_text(): Error on attempt {attempt} for {link}: {e}")
            await asyncio.sleep(attempt * 2)

        logging.error(f"extract_event_text(): Failed after {max_retries} attempts for {link}")
        return None

    async def extract_links(self, url: str) -> list:
        """
        Finds all <a href> links on a page, resolves absolute URLs,
        filters to same-domain HTTP(S), skips ICS/calendar feeds, removes duplicates.
        """
        if not await self.login_if_required(url):
            logging.error(f"Login failed for link extraction at {url}")
            return []

        await self.page.goto(url, timeout=15000)
        await self.page.wait_for_load_state("domcontentloaded")
        soup = BeautifulSoup(await self.page.content(), 'html.parser')
        base = self.page.url
        found = set()

        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            # Skip anchors and JavaScript
            if href.startswith('#') or href.lower().startswith('javascript:'):
                continue
            abs_url = urljoin(base, href)
            parsed = urlparse(abs_url)
            # Only HTTP/S
            if parsed.scheme not in ('http', 'https'):
                continue
            # Same domain only
            if parsed.netloc != urlparse(base).netloc:
                continue
            # Skip calendar/ICS feeds
            # e.g., query params outlook-ical, webcal links, .ics endpoints
            if 'ical' in parsed.query.lower() or parsed.path.lower().endswith('.ics'):
                logging.info(f"Skipping calendar feed link: {abs_url}")
                continue
            found.add(abs_url)

        return list(found)

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def main(self, url: str, multiple: bool = False):
        """
        Initialize browser, then perform either single or multi-page scraping based on `multiple` flag.
        """
        await self.init_browser()
        logging.info(f"main(): Initialized browser for {url} (multiple={multiple})")

        # Single-URL mode
        if not multiple:
            text = await self.extract_event_text(url)
            await self.close()
            return text

        # Multi-URL mode: discover and scrape all same-domain links
        links = await self.extract_links(url)
        results = {}

        # Always include main page
        main_text = await self.extract_event_text(url)
        results[url] = main_text

        for link in links:
            if link == url:
                continue
            text = await self.extract_event_text(link)
            results[link] = text

        await self.close()
        if len(results) == 1:
            return main_text
        return results


if __name__ == "__main__":
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Setup logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("\n\nrd_ext.py starting...")

    start_time = datetime.now()
    logging.info(f"__main__: Start at {start_time}")

    db_handler = DatabaseHandler(config)
    start_df = db_handler.count_events_urls_start(os.path.basename(__file__))

    read_extract = ReadExtract("config/config.yaml")
    llm_handler = LLMHandler("config/config.yaml")

    df = pd.read_csv(config['input']['edge_cases'])
    # Expecting columns in order: source, keywords, url, multiple
    for source, keywords, url, multiple in df.itertuples(index=False, name=None):
        multiple_flag = str(multiple).strip().lower() == 'yes'
        logging.info(f"__main__: url={url}, source={source}, keywords={keywords}, multiple={multiple_flag}")
        extracted = asyncio.run(read_extract.main(url, multiple_flag))

        if isinstance(extracted, dict):
            for event_url, text in extracted.items():
                llm_handler.process_llm_response(event_url, text, source, keywords, prompt=event_url)
        else:
            llm_handler.process_llm_response(url, extracted, source, keywords, prompt=url)

    db_handler.count_events_urls_end(start_df, os.path.basename(__file__))
    end_time = datetime.now()
    logging.info(f"__main__: End at {end_time}, duration {end_time - start_time}\n\n")
