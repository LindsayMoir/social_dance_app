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
        storage = f"{organization.lower()}_auth.json"

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

        # 3) Detect reCAPTCHA
        try:
            await self.page.wait_for_selector("iframe[src*='recaptcha']", timeout=10000)
            await self.page.screenshot(path="debug/facebook_recaptcha.png", full_page=True)
            logging.info("login_to_facebook: reCAPTCHA detected—please solve in browser.")
            input("After solving CAPTCHA, press Enter to continue…")
        except PlaywrightTimeoutError:
            logging.info("login_to_facebook: no reCAPTCHA detected.")

        # 4) Finalize
        await asyncio.sleep(random.uniform(3,6))
        if "login" in self.page.url.lower():
            logging.error("login_to_facebook: still on login page, aborting.")
            return False

        try:
            await self.page.context.storage_state(path=storage)
            logging.info("login_to_facebook: session saved.")
        except Exception as e:
            logging.warning(f"login_to_facebook: could not save session: {e}")

        return True

    async def login_to_website(self, organization, login_url, email_selector, pass_selector, submit_selector):
        storage = f"{organization.lower()}_auth.json"

        # Tear down old page
        if self.page:
            await self.page.close()

        # 1) Try existing session
        try:
            ctx = await self.browser.new_context(storage_state=storage)
            self.page = await ctx.new_page()
            await self.page.goto(login_url, wait_until="domcontentloaded", timeout=20000)
            if "login" not in self.page.url.lower():
                logging.info(f"login_to_website(): reused session for {organization}.")
                return True
        except Exception:
            logging.info(f"login_to_website(): no saved session for {organization}, manual login.")

        # 2) Detect form presence
        try:
            await self.page.wait_for_selector(email_selector, timeout=5000)
            logging.info(f"login_to_website(): Login form detected for {organization}.")
        except PlaywrightTimeoutError:
            logging.info(f"login_to_website(): No login form for {organization}; skipping login.")
            return True

        # 3) Eventbrite‐specific flow
        if 'eventbrite' in login_url:
            await self.page.fill("input[name='email']", os.getenv('EVENTBRITE_APPID_UID'))
            await self.page.click("button[type='submit']")
            logging.info("login_to_website(): Email submitted for Eventbrite.")

            pw_selectors = ["input[name='password']", "input[type='password']", "[data-automation='password-input']"]
            try:
                await self.page.wait_for_selector(", ".join(pw_selectors), timeout=15000)
            except PlaywrightTimeoutError:
                await self.page.screenshot(path="debug/eventbrite_login.png", full_page=True)
                logging.error("login_to_website(): password field never appeared.")
                return False

            handle = None
            for sel in pw_selectors:
                handle = await self.page.query_selector(sel)
                if handle: break

            await handle.fill(os.getenv('EVENTBRITE_KEY_PW'))
            await self.page.click("button[type='submit']")
            await asyncio.sleep(3)

            # reCAPTCHA?
            try:
                await self.page.wait_for_selector("iframe[src*='recaptcha']", timeout=10000)
                await self.page.screenshot(path="debug/eventbrite_recaptcha.png", full_page=True)
                input("Solve any Eventbrite CAPTCHA, then press Enter…")
            except PlaywrightTimeoutError:
                pass

            # save
            try:
                await self.page.context.storage_state(path=storage)
                logging.info("login_to_website(): Eventbrite session saved.")
            except Exception as e:
                logging.warning(f"login_to_website(): failed to save: {e}")

            return True

        # 4) Generic flow
        email, password, _ = get_credentials(organization)
        await self.page.fill(email_selector, email)
        await self.page.fill(pass_selector, password)
        await self.page.click(submit_selector)
        await asyncio.sleep(random.uniform(3,6))

        logging.warning(f"login_to_website(): If a CAPTCHA appeared, solve it then press Enter.")
        input("Press Enter once done…")
        await asyncio.sleep(random.uniform(3,6))

        if "login" in self.page.url.lower():
            logging.error(f"login_to_website(): still on login page—failed.")
            return False

        try:
            await self.page.context.storage_state(path=storage)
            logging.info(f"login_to_website(): session saved for {organization}.")
        except Exception as e:
            logging.warning(f"login_to_website(): could not save: {e}")

        return True


    async def login_to_website(self, organization, login_url, email_selector, pass_selector, submit_selector):
        """
        Logs in to an arbitrary site, reusing a saved storage_state if available,
        or falling back to filling the form and prompting for any CAPTCHA.

        Args:
            organization (str): Name of the site (used for naming storage files).
            login_url (str): URL to navigate to for login.
            email_selector (str): CSS selector for the email/username input.
            pass_selector (str): CSS selector for the password input.
            submit_selector (str): CSS selector for the submit button.

        Returns:
            bool: True if logged in or no login needed, False on failure.
        """
        # 1) Tear down any existing page
        if self.page:
            await self.page.close()

        # 2) Try to load an existing session
        storage_path = f"{organization.lower()}_auth.json"
        try:
            context = await self.browser.new_context(storage_state=storage_path)
            self.page = await context.new_page()
            await self.page.goto(login_url, timeout=20000)
            # if redirected away from login, assume we're already authenticated
            if login_url.split('?')[0] not in self.page.url:
                logging.info(f"login_to_website(): Reused session for {organization}.")
                self.logged_in = True
                return True
        except Exception:
            logging.info(f"login_to_website(): No valid saved session for {organization}, proceeding to manual login.")

        # 3) Detect whether a login form is present
        try:
            await self.page.wait_for_selector(email_selector, timeout=5000)
            logging.info(f"login_to_website(): Login form detected for {organization}.")
        except PlaywrightTimeoutError:
            logging.info(f"login_to_website(): No login form at {organization}, skipping.")
            self.logged_in = True
            return True

        # 4) Eventbrite‐specific handling (popup flows, multiple password selectors, captcha)
        if 'eventbrite' in login_url:
            email_sel = "input[name='email']"
            password_selectors = [
                "input[name='password']",
                "input[type='password']",
                "[data-automation='password-input']"
            ]

            # fill email
            await self.page.fill(email_sel, os.getenv('EVENTBRITE_APPID_UID'))
            await self.page.click("button[type='submit']")
            logging.info("login_to_website(): Submitted Eventbrite email.")

            # wait for any one of the password selectors
            try:
                await self.page.wait_for_selector(", ".join(password_selectors), timeout=15000)
            except PlaywrightTimeoutError:
                await self.page.screenshot(path="debug/eventbrite_login.png", full_page=True)
                logging.error("login_to_website(): Password field never appeared; screenshot at debug/eventbrite_login.png")
                return False

            # choose the first one that matches
            handle = None
            for sel in password_selectors:
                handle = await self.page.query_selector(sel)
                if handle:
                    break

            # fill password + submit
            await handle.fill(os.getenv('EVENTBRITE_KEY_PW'))
            await self.page.click("button[type='submit']")
            await self.page.wait_for_timeout(3000)

            # save session
            try:
                await self.page.context.storage_state(path=storage_path)
                logging.info(f"login_to_website(): Saved Eventbrite session state.")
            except Exception as e:
                logging.warning(f"login_to_website(): Couldn't save session: {e}")

            self.logged_in = True
            return True

        # 5) Generic login flow
        email, password, _ = get_credentials(organization)
        await self.page.fill(email_selector, email)
        await self.page.fill(pass_selector, password)
        await self.page.click(submit_selector)
        # let any JS / redirects finish
        await self.page.wait_for_timeout(random.uniform(3, 6))

        # prompt user to solve CAPTCHA if it pops up
        logging.warning(f"login_to_website(): If a CAPTCHA appeared on {organization}, please solve it in the browser, then press Enter.")
        input("Press Enter once you’ve completed any challenge...")

        await self.page.wait_for_timeout(random.uniform(3, 6))
        # check for failure
        if "login" in self.page.url.lower():
            logging.error(f"login_to_website(): Login to {organization} failed; still on login page.")
            return False

        # save storage state for next time
        try:
            await self.page.context.storage_state(path=storage_path)
            logging.info(f"login_to_website(): Saved session state for {organization}.")
        except Exception as e:
            logging.warning(f"login_to_website(): Could not save session: {e}")

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


def uvic_rueda():
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
    db_handler.write_events_to_db(df, url=event_dict['url'], 
                                      source=event_dict['source'], 
                                      keywords=event_dict['dance_style'])


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

    # Add uvic wednesday rueda event. This event sometimes appears and then it dissapears. Lets just put it in.
    uvic_rueda()

    # Count events and urls after rd_ext.py
    db_handler.count_events_urls_end(start_df, file_name)

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
