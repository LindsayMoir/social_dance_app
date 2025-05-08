"""
rd_ext.py

input (str): The event URL.

This module defines the ReadExtract class, which uses the asyncio version of Playwright 
to manage a single browser page globally for tasks such as logging into Facebook 
and extracting event text. The class ensures that only one login occurs and the same 
page is reused across multiple operations. It is designed to be easily callable 
from other class libraries and supports asynchronous operations.

Objectives:
1. Maintain one globally shared page across the class instance.
2. Ensure a single login session is reused for all operations, avoiding multiple logins.
3. Provide an easy-to-use interface for other class libraries.
4. Utilize the asyncio version of Playwright for asynchronous operations.
5. Provides a place to deal with odd edge cases that may arise if you are using the logic in __main__
    a. This will result in database updates to the events and urls tables.

output (str or dict): The extracted text content. For pages such as Bard & Banker live music, a dictionary mapping event URLs to text is returned.
"""

import asyncio
import json  # New import for JSON parsing
from bs4 import BeautifulSoup
from datetime import datetime
import logging
import pandas as pd
import os
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
import random
from urllib.parse import urljoin, urlparse
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
        

    def extract_text_with_playwright(self, url):
        """
        Synchronously extracts the text content from a web page using Playwright.
        If a Google sign-in page is encountered, the method aborts and returns None.

        Args:
            url (str): The URL of the web page to extract text from.

        Returns:
            str or None: The extracted text content, or None if Google sign-in is detected or an error occurs.
        """
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config['crawling']['headless'])
                page = browser.new_page()
                page.goto(url, timeout=10000)
                page.wait_for_timeout(3000)

                # Check for Google sign-in page
                if "accounts.google.com" in page.url or page.is_visible("input#identifierId"):
                    logging.info("def extract_text_with_playwright(): Google sign-in detected. Aborting extraction.")
                    browser.close()
                    return None

                content = page.content()
                soup = BeautifulSoup(content, 'html.parser')
                extracted_text = ' '.join(soup.stripped_strings)

                browser.close()
                return extracted_text
        except Exception as e:
            logging.error(f"def extract_text_with_playwright(): Failed to extract text from {url}: {e}")
            return None


    async def init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.config['crawling']['headless'])
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()


    async def login_to_facebook(self, organization):
        # Close any existing page to avoid multiple windows
        if self.page:
            await self.page.close()
        try:
            context = await self.browser.new_context(storage_state="auth.json")
            self.page = await context.new_page()
            await self.page.goto("https://www.facebook.com/", timeout=60000)
            if "login" not in self.page.url.lower():
                logging.info("def login_to_facebook(): Loaded existing session. Already logged into Facebook.")
                self.logged_in = True
                return True
        except Exception:
            logging.info("def login_to_facebook(): No valid saved session found. Proceeding with manual login.")

        email, password, _ = get_credentials(self.config, organization)  # Use utility function

        await self.page.goto("https://www.facebook.com/login", timeout=60000)
        await self.page.fill("input[name='email']", email)
        await self.page.fill("input[name='pass']", password)
        await self.page.click("button[name='login']")
        await self.page.wait_for_timeout(random.randint(4000, 6000))

        logging.warning("Please solve any captcha or challenge in the browser, then press Enter here to continue...")
        input("After solving captcha/challenge (if any), press Enter to continue...")

        await self.page.wait_for_timeout(random.randint(4000, 6000))

        if "login" in self.page.url.lower():
            logging.error("def login_to_facebook(): Login failed. Please check your credentials or solve captcha challenges.")
            return False

        try:
            await self.page.context.storage_state(path="auth.json")
            logging.info("def login_to_facebook(): Session state saved for future use.")
        except Exception as e:
            logging.warning(f"def login_to_facebook(): Could not save session state: {e}")

        self.logged_in = True
        logging.info("def login_to_facebook(): Login to Facebook successful.")
        return True


    async def login_to_website(self, organization, login_url, email_selector, pass_selector, submit_selector):
        # Close any existing page
        if self.page:
            await self.page.close()

        # Try to load existing session
        try:
            context = await self.browser.new_context(storage_state=f"{organization.lower()}_auth.json")
            self.page = await context.new_page()
            await self.page.goto(login_url, timeout=20000)

            if organization.lower() not in self.page.url.lower():
                logging.info(f"Loaded existing session for {organization}, no login required.")
                self.logged_in = True
                return True
        except Exception:
            logging.info(f"No valid saved session for {organization}, checking if login is needed...")

        # Check if login form is needed
        try:
            await self.page.wait_for_selector(email_selector, timeout=5000)
            logging.info(f"Login required for {organization}.")
        except:
            logging.info(f"No login form detected for {organization}, skipping login.")
            self.logged_in = True
            return True

        # Handle Eventbrite-specific login
        if 'eventbrite' in login_url:
            email_selector = "input[name='email']"
            password_selector = "input[name='password']"

            if await self.page.query_selector(email_selector):
                email = os.getenv('EVENTBRITE_APPID_UID')
                password = os.getenv('EVENTBRITE_KEY_PW')
                logging.info("Eventbrite login popup detected.")

                await self.page.fill(email_selector, email)
                await self.page.click("button[type='submit']")
                await self.page.wait_for_selector(password_selector, timeout=5000)
                await self.page.fill(password_selector, password)
                await self.page.click("button[type='submit']")
                await self.page.wait_for_timeout(3000)
                self.logged_in = True
                return True
            else:
                logging.info("No Eventbrite login popup detected.")
                self.logged_in = True
                return True

        # Generic login handling
        email, password, _ = get_credentials(organization)
        await self.page.fill(email_selector, email)
        await self.page.fill(pass_selector, password)
        await self.page.click(submit_selector)
        await self.page.wait_for_timeout(random.randint(4000, 6000))

        logging.warning(f"Please solve any captcha on {organization}'s login page, then press Enter to continue...")
        input("Press Enter after solving captcha (if any)...")
        await self.page.wait_for_timeout(random.randint(4000, 6000))

        if "login" in self.page.url.lower():
            logging.error(f"Login to {organization} failed. Credentials may be incorrect.")
            return False

        try:
            await self.page.context.storage_state(path=f"{organization.lower()}_auth.json")
            logging.info(f"Session state saved for {organization}.")
        except Exception as e:
            logging.warning(f"Could not save session state for {organization}: {e}")

        self.logged_in = True
        logging.info(f"Login to {organization} successful.")
        return True


    async def login_if_required(self, url: str) -> bool:
        """
        Determines if the URL belongs to Facebook, Google, allevents, or Eventbrite.
        If so, calls the corresponding login method. Otherwise, returns True (no login needed).
        """
        url_lower = url.lower()

        # If it's Facebook
        if "facebook.com" in url_lower:
            return await self.login_to_facebook("Facebook")

        # If it's Google
        elif "google" in url_lower:
            return await self.login_to_website(
                organization="Google",
                login_url="https://accounts.google.com/signin",
                email_selector="input[type='email']",
                pass_selector="input[type='password']",
                submit_selector="button[type='submit']"
            )

        # If it's Allevents
        elif "allevents" in url_lower:
            return await self.login_to_website(
                organization="allevents",
                login_url="https://www.allevents.in/login",
                email_selector="input[name='email']",
                pass_selector="input[name='password']",
                submit_selector="button[type='submit']"
            )

        # If it's Eventbrite
        elif "eventbrite" in url_lower:
            return await self.login_to_website(
                organization="eventbrite",
                login_url="https://www.eventbrite.com/signin/",
                email_selector="input#email",
                pass_selector="input#password",
                submit_selector="button[type='submit']"
            )

        # Otherwise, no login needed
        return True


    async def extract_event_text(self, link, max_retries=3):
        """
        Extracts text from an event page with retries if extraction fails.

        Args:
            link (str): The event URL.
            max_retries (int): Maximum number of retries in case of failure.

        Returns:
            str: Extracted text content or None if extraction fails.
        """
        login_success = await self.login_if_required(link)
        if not login_success:
            logging.error(f"def extract_event_text(): Login failed. Aborting extraction for {link}.")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                await self.page.goto(link, timeout=15000)
                await self.page.wait_for_load_state("domcontentloaded")  # Ensure the DOM is fully loaded
                await asyncio.sleep(random.uniform(3, 6))  # Randomized delay for stability

                # Extract page content
                content = await self.page.content()
                soup = BeautifulSoup(content, 'html.parser')
                extracted_text = ' '.join(soup.stripped_strings)

                if extracted_text.strip():  # Ensure non-empty text
                    logging.info(f"def extract_event_text(): Successfully extracted text from {link} on attempt {attempt}.")
                    return extracted_text
                else:
                    logging.warning(f"def extract_event_text(): Attempt {attempt} - No text found for {link}. Retrying...")

            except Exception as e:
                logging.error(f"def extract_event_text(): Attempt {attempt} failed for {link}. Error: {e}")

            # Wait before retrying (exponential backoff)
            await asyncio.sleep(attempt * 2)

        logging.error(f"def extract_event_text(): Extraction failed after {max_retries} attempts for {link}.")
        return None

    async def extract_live_music_event_urls(self, url):
        """
        Special method to handle the Bard & Banker live-music page.
        It navigates to the page, parses any JSON–LD with @type 'ItemList', and extracts event URLs.

        Args:
            url (str): The Bard & Banker live-music page URL.

        Returns:
            list: A list of unique event URLs.
        """
        await self.page.goto(url, timeout=15000)
        await self.page.wait_for_load_state("domcontentloaded")
        content = await self.page.content()
        soup = BeautifulSoup(content, 'html.parser')
        event_urls = []
        # Look for JSON–LD script tags
        scripts = soup.find_all("script", type="application/ld+json")
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("@type") == "ItemList":
                    item_list = data.get("itemListElement", [])
                    for item in item_list:
                        event_item = item.get("item", {})
                        event_url = event_item.get("url")
                        if event_url:
                            event_urls.append(event_url)
            except Exception as e:
                logging.warning(f"extract_live_music_event_urls(): Failed to parse JSON–LD: {e}")
        return list(set(event_urls))


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


if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("\n\nrd_ext.py starting...")

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before rd_ext.py
    start_df = db_handler.count_events_urls_start(file_name)

    # Instantiate the classes
    read_extract = ReadExtract("config/config.yaml")
    llm_handler = LLMHandler("config/config.yaml")

    # Read .csv file to deal with oddities
    df = pd.read_csv(config['input']['edge_cases'])

    # Expecting columns in order: source, keywords, url, multiple
    for source, keywords, url, multiple in df.itertuples(index=False, name=None):
        multiple_flag = str(multiple).strip().lower() == 'yes'
        logging.info(f"__main__: url={url}, source={source}, keywords={keywords}, multiple={multiple_flag}")
        extracted = asyncio.run(read_extract.main(url, multiple_flag))

        # If multiple events were found (i.e. extracted is a dict), process each event separately
        if isinstance(extracted, dict):
            for event_url, text in extracted.items():
                llm_status = llm_handler.process_llm_response(event_url, text, source, keywords, prompt=event_url)
        else:
            llm_status = llm_handler.process_llm_response(url, extracted, source, keywords, prompt=url)

    # Count events and urls after rd_ext.py
    db_handler.count_events_urls_end(start_df, file_name)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
