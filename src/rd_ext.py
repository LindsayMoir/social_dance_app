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
from datetime import date, datetime, timedelta
import logging
import pandas as pd
import os
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from playwright.sync_api import sync_playwright
import random
from urllib.parse import urljoin, urlparse
import yaml

from llm import LLMHandler
from credentials import get_credentials  # Import the utility function
from secret_paths import get_auth_file

# Module-level handler that will be initialized when needed
db_handler = None

def get_db_handler():
    """Get or create the global db_handler instance."""
    global db_handler
    if db_handler is None:
        llm_handler = LLMHandler("config/config.yaml")
        db_handler = llm_handler.db_handler
    return db_handler


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
        # Use Render secret path if available, otherwise local path
        storage = get_auth_file(organization.lower())

        # Tear down old page
        if self.page:
            await self.page.close()

        # 1) Try existing session
        try:
            ctx = await self.browser.new_context(storage_state=storage)
            self.page = await ctx.new_page()
            await self.page.goto("https://www.facebook.com/", timeout=60000)
            if "login" not in self.page.url.lower():
                logging.info("login_to_facebook: reused session, already logged in.")
                return True
        except Exception:
            logging.info("login_to_facebook: no saved session, fresh login.")

        # 2) Fresh login
        email, password, _ = get_credentials(self.config, organization)
        await self.page.goto("https://www.facebook.com/login", timeout=60000)
        await self.page.fill("input[name='email']", email)
        await self.page.fill("input[name='pass']", password)
        await self.page.click("button[name='login']")
        await asyncio.sleep(random.uniform(3,6))

        # 3) Detect CAPTCHA using centralized handler
        from captcha_handler import CaptchaHandler
        await CaptchaHandler.detect_and_handle_async(self.page, "Facebook", timeout=10000)

        # 4) Finalize
        await asyncio.sleep(random.uniform(3,6))
        if "login" in self.page.url.lower():
            logging.error("login_to_facebook: still on login page, aborting.")
            return False

        try:
            await self.page.context.storage_state(path=storage)
            logging.info("login_to_facebook: session saved.")
            # Sync to database
            from secret_paths import sync_auth_to_db
            sync_auth_to_db(storage, 'facebook')
        except Exception as e:
            logging.warning(f"login_to_facebook: could not save session: {e}")

        return True


    async def login_to_website(
        self,
        organization: str,
        login_url: str,
        email_selector: str,
        pass_selector: str,
        submit_selector: str
    ) -> bool:
        """
        Logs in to an arbitrary site, reusing a saved storage_state if available,
        or falling back to filling the form (including Eventbrite flow) and prompting
        for any manual 2FA/CAPTCHA. Uses a single context/page throughout.
        Returns True on success (or if no login form is found), False on failure.
        """
        # Use Render secret path if available, otherwise local path
        storage_path = get_auth_file(organization.lower())

        # ──────────── Quick bail if already logged in ────────────
        if self.logged_in:
            logging.info(f"login_to_website(): Already logged in to {organization}, reusing session.")
            return True

        # ──────────── 1) Try to reuse existing session ────────────
        try:
            ctx = await self.browser.new_context(storage_state=storage_path)
            page = await ctx.new_page()
            await page.goto(login_url, wait_until="domcontentloaded", timeout=20000)
            # If we land off the login page, session is valid
            if login_url.split("?")[0] not in page.url:
                self.context = ctx
                self.page = page
                self.logged_in = True
                logging.info(f"login_to_website(): Reused session for {organization}.")
                return True
            # Otherwise, close and do fresh login
            await page.close()
            await ctx.close()
            logging.info(f"login_to_website(): Storage found but still on login page; will do fresh login for {organization}.")
        except Exception:
            logging.info(f"login_to_website(): No valid saved session for {organization}, fresh login.")

        # ──────────── 2) Fresh login: reset context/page ────────────
        if getattr(self, 'page', None):
            await self.page.close()
        if getattr(self, 'context', None):
            await self.context.close()
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        await self.page.goto(login_url, wait_until="domcontentloaded", timeout=20000)

        # ──────────── 3) Check for login form presence ────────────
        try:
            await self.page.wait_for_selector(email_selector, timeout=5000)
            logging.info(f"login_to_website(): Login form detected for {organization}.")
        except PlaywrightTimeoutError:
            logging.info(f"login_to_website(): No login form for {organization}; assuming no login needed.")
            self.logged_in = True
            return True

        # ──────────── 4) Eventbrite‐specific flow ────────────
        if "eventbrite" in login_url:
            # Submit email
            await self.page.fill(email_selector, os.getenv("EVENTBRITE_APPID_UID"))
            await self.page.click(submit_selector)
            logging.info("login_to_website(): Submitted Eventbrite email.")

            # Wait for password field
            pw_selectors = [pass_selector, "input[type='password']"]
            try:
                await self.page.wait_for_selector(
                    ",".join(pw_selectors), timeout=15000
                )
            except PlaywrightTimeoutError:
                await self.page.screenshot(path="debug/eventbrite_login.png", full_page=True)
                logging.error("login_to_website(): Password field never appeared for Eventbrite.")
                return False

            # Fill password
            handle = None
            for sel in pw_selectors:
                handle = await self.page.query_selector(sel)
                if handle:
                    break
            password = os.getenv("EVENTBRITE_KEY_PW")
            await handle.fill(password)
            await self.page.click(submit_selector)
            logging.info("login_to_website(): Filled Eventbrite password and clicked submit.")

            # Manual 2FA pause
            input(
                "🔒 2FA step: check your email for the code, paste it into the browser’s 2FA field, "
                "click Continue there, then press Enter here to resume…"
            )
            await self.page.wait_for_timeout(2000)

            # Save session state
            try:
                await self.context.storage_state(path=storage_path)
                logging.info("login_to_website(): Saved Eventbrite session state.")
                # Sync to database
                from secret_paths import sync_auth_to_db
                sync_auth_to_db(storage_path, organization)
            except Exception as e:
                logging.warning(f"login_to_website(): Could not save Eventbrite session: {e}")

            self.logged_in = True
            return True

        # ──────────── 5) Generic login flow ────────────
        # Fill credentials
        email, password, _ = get_credentials(self.config, organization)
        await self.page.fill(email_selector, email)
        await self.page.fill(pass_selector, password)
        await self.page.click(submit_selector)
        await self.page.wait_for_timeout(random.uniform(3, 6))

        # Detect CAPTCHA using centralized handler
        from captcha_handler import CaptchaHandler
        await CaptchaHandler.detect_and_handle_async(self.page, organization, timeout=5000)

        # Final verification
        if "login" in self.page.url.lower():
            logging.error(f"login_to_website(): Still on login page after submit for {organization}.")
            return False

        # Save storage state
        try:
            await self.context.storage_state(path=storage_path)
            logging.info(f"login_to_website(): Saved session state for {organization}.")
            # Sync to database
            from secret_paths import sync_auth_to_db
            sync_auth_to_db(storage_path, organization)
        except Exception as e:
            logging.warning(f"login_to_website(): Could not save session state: {e}")

        self.logged_in = True
        logging.info(f"login_to_website(): Login to {organization} succeeded.")
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
        Extracts text from an event page with retries and crash recovery.
        """
        # Ensure we're logged in
        if not await self.login_if_required(link):
            logging.error(f"extract_event_text: Login failed for {link}, skipping.")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                # Navigate; wait only for the DOM, not all resources
                await self.page.goto(
                    link,
                    wait_until="domcontentloaded",
                    timeout=30000
                )

                # Small random delay to let any JS render
                await asyncio.sleep(random.uniform(2, 5))

                # For Eventbrite pages, check for and click "View all event details" button
                if "eventbrite.com" in link.lower():
                    try:
                        # Look for the "View all event details" button
                        view_details_button = await self.page.query_selector('text="View all event details"')
                        if view_details_button:
                            logging.info(f"extract_event_text: Found 'View all event details' button for {link}, clicking...")
                            await view_details_button.click()
                            # Wait a moment for content to load after clicking
                            await asyncio.sleep(random.uniform(2, 4))
                    except Exception as e:
                        logging.warning(f"extract_event_text: Error clicking 'View all event details' button for {link}: {e}")

                # Pull the page HTML and parse
                content = await self.page.content()
                soup = BeautifulSoup(content, "html.parser")
                text = " ".join(soup.stripped_strings)

                if text.strip():
                    logging.info(f"extract_event_text: Success on attempt {attempt} for {link}")
                    return text
                else:
                    logging.warning(f"extract_event_text: Attempt {attempt} yielded empty text; retrying...")

            except PlaywrightTimeoutError as te:
                logging.error(f"extract_event_text: Attempt {attempt} timeout for {link}: {te}")

            except PlaywrightError as pe:
                msg = str(pe)
                # Detect a crash and recreate our page/context
                if "Page crashed" in msg or "Navigation" in msg and "interrupted" in msg:
                    logging.warning(f"extract_event_text: Page crashed or interrupted on attempt {attempt} for {link}. Resetting page.")
                    # Close old context & page, then re-init a fresh one
                    try:
                        await self.context.close()
                    except: pass
                    self.context = await self.browser.new_context()
                    self.page = await self.context.new_page()
                    continue
                else:
                    logging.error(f"extract_event_text: Attempt {attempt} failed for {link}: {pe}")

            except Exception as e:
                logging.error(f"extract_event_text: Unexpected error on attempt {attempt} for {link}: {e}")

            # Exponential backoff before retry
            backoff = attempt * 2
            logging.info(f"extract_event_text: Waiting {backoff}s before next attempt...")
            await asyncio.sleep(backoff)

        logging.error(f"extract_event_text: All {max_retries} attempts failed for {link}.")
        return None


    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


    async def extract_from_url(self, url: str, multiple: bool = False):
        """
        Extracts text from URL(s) without managing browser lifecycle.
        Browser must already be initialized before calling this method.

        Special handling for The Coda calendar - uses extract_coda_events() to process
        individual event links instead of extracting the entire page at once (which causes LLM timeouts).

        Args:
            url: The URL to extract from
            multiple: Whether to follow links on the page and extract from multiple URLs

        Returns:
            str or dict: Text content if single, or dict of {url: text} if multiple
        """
        logging.info(f"extract_from_url(): Processing {url} (multiple={multiple})")

        # Check urls to see if they should be scraped
        if not get_db_handler().should_process_url(url):
            logging.info(f"extract_from_url(): Skipping URL {url} based on historical relevancy.")
            return None

        # Special handling for The Coda calendar - extract individual event links
        # This avoids LLM timeout issues by processing events individually instead of all at once
        if 'gotothecoda.com/calendar' in url:
            logging.info(f"extract_from_url(): Detected The Coda calendar, using special event extraction")
            event_data = await self.extract_coda_events(url)
            if event_data:
                # Return dict mapping event URLs to their text
                # This triggers the multi-event processing logic in __main__ (line 698-701)
                results = {event_url: event_text for event_url, event_text in event_data}
                logging.info(f"extract_from_url(): Returning {len(results)} Coda events as dict for multi-event processing")
                return results
            else:
                logging.warning(f"extract_from_url(): No Coda events extracted from {url}, returning None")
                return None

        if not multiple:
            text = await self.extract_event_text(url)
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

            # Check urls to see if they should be scraped
            if not get_db_handler().should_process_url(link):
                logging.info(f"extract_from_url(): Skipping URL {link} based on historical relevancy.")
                continue

            text = await self.extract_event_text(link)
            results[link] = text

        if len(results) == 1:
            return main_text
        return results

    async def main(self, url: str, multiple: bool = False):
        """
        Initialize browser, then perform either single or multi-page scraping based on `multiple` flag.
        """
        await self.init_browser()
        logging.info(f"main(): Initialized browser for {url} (multiple={multiple})")

        result = await self.extract_from_url(url, multiple)

        await self.close()
        return result


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

    async def extract_coda_events(self, calendar_url: str) -> list:
        """
        Extracts individual event links from The Coda calendar page.

        The Coda displays all events as a grid of cards with "More Info" links.
        This method:
        1. Navigates to the calendar page
        2. Extracts all event links
        3. Visits each event individually
        4. Extracts text from each event page
        5. Returns list of (event_url, event_text) tuples

        Args:
            calendar_url (str): The Coda calendar URL (https://gotothecoda.com/calendar)

        Returns:
            list: List of tuples (event_url, event_text) for each event found

        Raises:
            Exception: If unable to navigate to calendar page
        """
        event_data = []

        try:
            logging.info(f"extract_coda_events(): Starting extraction from {calendar_url}")

            # Navigate to the calendar page
            if not await self.login_if_required(calendar_url):
                logging.error(f"extract_coda_events(): Login failed for {calendar_url}")
                return event_data

            await self.page.goto(calendar_url, timeout=15000)
            await self.page.wait_for_load_state("domcontentloaded")

            # Extract all event links from the calendar page
            # The Coda uses various link patterns - get all links and filter
            event_links = []

            # Strategy 1: Get all links and filter for event-like URLs
            all_links = await self.page.query_selector_all("a")
            logging.info(f"extract_coda_events(): Found {len(all_links)} total links on page")

            # Look for links to individual event pages
            unique_urls = set()
            for link in all_links:
                href = await link.get_attribute('href')
                if href:
                    # Make absolute URL
                    if href.startswith('/'):
                        href = urljoin(calendar_url, href)
                    elif not href.startswith('http'):
                        href = urljoin(calendar_url, href)

                    # Check if this looks like an individual event link
                    # The Coda events are at /show/{id} or similar patterns
                    if ('/show/' in href or '/event' in href.lower() or '/events/' in href) and href != calendar_url:
                        unique_urls.add(href)

            event_links = list(unique_urls)
            logging.info(f"extract_coda_events(): Filtered to {len(event_links)} unique event URLs")

            if not event_links:
                logging.warning(f"extract_coda_events(): No event links found on {calendar_url}")
                # Log some debug info about what's on the page
                content = await self.page.content()
                logging.warning(f"extract_coda_events(): Page content length: {len(content)} bytes")
                if 'more info' in content.lower():
                    logging.warning(f"extract_coda_events(): 'more info' text exists in page")
                if '/events/' in content or '/show/' in content:
                    logging.warning(f"extract_coda_events(): Event-like paths exist in page")
                return event_data

            # Process each event link
            for idx, event_url in enumerate(event_links, 1):
                try:
                    logging.info(f"extract_coda_events(): [{idx}/{len(event_links)}] Processing {event_url}")

                    # Navigate to the event page
                    await self.page.goto(event_url, timeout=10000)
                    await self.page.wait_for_load_state("domcontentloaded")

                    # Extract text from event page
                    content = await self.page.content()
                    soup = BeautifulSoup(content, 'html.parser')

                    # Remove scripts and styles
                    for tag in soup(['script', 'style']):
                        tag.decompose()

                    event_text = ' '.join(soup.stripped_strings)

                    if event_text:
                        event_data.append((event_url, event_text))
                        logging.info(f"extract_coda_events(): Extracted {len(event_text)} chars from event {idx}")
                    else:
                        logging.warning(f"extract_coda_events(): No text extracted from {event_url}")

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logging.error(f"extract_coda_events(): Failed to process event link {idx}: {e}")
                    continue

            logging.info(f"extract_coda_events(): Successfully extracted {len(event_data)} events")
            return event_data

        except Exception as e:
            logging.error(f"extract_coda_events(): Failed to extract Coda events: {e}")
            return event_data


    def uvic_rueda(self):
        """
        Reads the UVic Rueda event definition from config, updates the 'date'
        to the next Wednesday (inclusive), and writes it to the DB.
        """
        # 1. grab the dict
        event_dict = config['constants']['uvic_rueda_dict']

        # ─── compute next Wednesday (0=Mon, 1=Tue, 2=Wed … 6=Sun)
        today = date.today()
        days_ahead = (2 - today.weekday() + 7) % 7
        next_wed = today + timedelta(days=days_ahead)

        # 2. overwrite the date field in your dict
        event_dict['start_date'] = next_wed.isoformat()   # e.g. "2025-05-14"
        event_dict['end_date'] = event_dict['start_date']

        # 2A Set the price
        event_dict['price'] = '0'

        # 3. build your one‑row DataFrame
        df = pd.DataFrame([event_dict])

        # 3. write to Postgres via db_handler
        get_db_handler().write_events_to_db(df, 
                                      url=event_dict['url'], 
                                      parent_url = '',
                                      source=event_dict['source'], 
                                      keywords=event_dict['dance_style'])
        # 4. log the action
        logging.info(f"uvic_rueda(): Added UVic Rueda event for {next_wed.isoformat()} to DB.")


    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


if __name__ == "__main__":

    # Setup centralized logging
    from logging_config import setup_logging
    setup_logging('rd_ext')

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    logging.info("\n\nrd_ext.py starting...")

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Instantiate the classes
    read_extract = ReadExtract("config/config.yaml")
    llm_handler = LLMHandler("config/config.yaml")
    
    # Set the module-level db_handler
    db_handler = llm_handler.db_handler
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before rd_ext.py
    start_df = db_handler.count_events_urls_start(file_name)

    # Read .csv file to deal with oddities
    df = pd.read_csv(config['input']['edge_cases'])

    # Define async function to process all URLs with single browser instance
    async def process_all_urls():
        # Initialize browser once
        await read_extract.init_browser()
        logging.info("__main__: Browser initialized once for all URLs")

        # Expecting columns in order: source, keywords, url, multiple
        for source, keywords, url, multiple in df.itertuples(index=False, name=None):
            multiple_flag = str(multiple).strip().lower() == 'yes'
            logging.info(f"__main__: url={url}, source={source}, keywords={keywords}, multiple={multiple_flag}")
            extracted = await read_extract.extract_from_url(url, multiple_flag)

            # If multiple events were found (i.e. extracted is a dict), process each event separately
            if isinstance(extracted, dict):
                for event_url, text in extracted.items():
                    parent_url = url # Use the original URL as parent
                    llm_status = llm_handler.process_llm_response(event_url, parent_url, text, source, keywords, prompt_type=url)
            else:
                parent_url = ''  # No parent URL for single events
                llm_status = llm_handler.process_llm_response(url, parent_url, extracted, source, keywords, prompt_type=url)

        # Close browser once after all URLs processed
        await read_extract.close()
        logging.info("__main__: Browser closed after processing all URLs")

    # Run the async function
    asyncio.run(process_all_urls())

    # Add uvic wednesday rueda event. This event sometimes appears and then it dissapears. Lets just put it in.
    read_extract.uvic_rueda()

    # Count events and urls after rd_ext.py
    db_handler.count_events_urls_end(start_df, file_name)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
