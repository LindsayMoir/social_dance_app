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
import time
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from playwright.sync_api import sync_playwright
import random
from urllib.parse import urljoin, urlparse
from urllib.parse import parse_qs, unquote
import yaml

from config_runtime import get_config_path, load_config
from db import DatabaseHandler
from llm import LLMHandler
from credentials import get_credentials  # Import the utility function
from page_classifier import evaluate_step_ownership, is_social_url, resolve_prompt_type
from secret_paths import get_auth_file

# Module-level handler that will be initialized when needed
db_handler = None


def is_calendar_export_url(url: str) -> bool:
    """Return True for feed/export links that should not be fetched as event pages."""
    raw = str(url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    if (parsed.scheme or "").lower() == "webcal":
        return True
    path_low = (parsed.path or "").lower()
    query_low = (parsed.query or "").lower()
    query_map = parse_qs(parsed.query or "")
    action = str(query_map.get("action", [""])[0] or "").lower()
    cid_values = [unquote(str(value or "")).lower() for value in query_map.get("cid", [])]
    if path_low.endswith(".ics"):
        return True
    if path_low.endswith("/calendar/event") and action == "template":
        return True
    if path_low.endswith("/calendar/render") and any(
        value.startswith(("webcal:", "webcals:", "http://", "https://"))
        for value in cid_values
    ):
        return True
    return any(token in query_low for token in ("ical=1", "outlook-ical=1", "outlook_ical=1", "webcal="))


def _record_rd_ext_scrape_metric(
    db_handler: DatabaseHandler,
    *,
    link: str,
    parent_url: str,
    source: str,
    keywords: list[str] | str,
    access_attempted: bool,
    extraction_attempted: bool,
    extraction_succeeded: bool,
    extraction_skipped: bool,
    decision_reason: str,
    access_succeeded: bool,
    text_extracted: bool,
    keywords_found: bool,
    events_written: int,
    links_discovered: int = 0,
    links_followed: int = 0,
) -> None:
    """Persist rd_ext per-URL telemetry into the shared URL scrape metrics table."""
    try:
        db_handler.write_url_scrape_metric(
            {
                "run_id": os.getenv("DS_RUN_ID", "na"),
                "step_name": os.getenv("DS_STEP_NAME", "rd_ext"),
                "link": link,
                "parent_url": parent_url,
                "source": source,
                "keywords": keywords,
                "archetype": "edge_case",
                "extraction_attempted": extraction_attempted,
                "extraction_succeeded": extraction_succeeded,
                "extraction_skipped": extraction_skipped,
                "decision_reason": decision_reason,
                "handled_by": "rd_ext.py",
                "routing_reason": decision_reason,
                "access_attempted": access_attempted,
                "access_succeeded": access_succeeded,
                "text_extracted": text_extracted,
                "keywords_found": keywords_found,
                "events_written": int(events_written or 0),
                "ocr_attempted": False,
                "ocr_succeeded": False,
                "vision_attempted": False,
                "vision_succeeded": False,
                "fallback_used": False,
                "links_discovered": int(links_discovered or 0),
                "links_followed": int(links_followed or 0),
                "time_stamp": datetime.now(),
            }
        )
    except Exception as exc:
        logging.warning("_record_rd_ext_scrape_metric(): failed for %s: %s", link, exc)


def _safe_bool(value: object) -> bool:
    """Coerce config-like boolean values safely."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


async def _wait_for_login_completion(
    page,
    login_url: str,
    email_selector: str,
    pass_selector: str,
    timeout_ms: int = 30000,
) -> bool:
    """Wait for a login flow to leave the login page and hide login controls."""
    login_base = str(login_url or "").split("?", 1)[0].rstrip("/")
    deadline = time.monotonic() + max(1, timeout_ms) / 1000.0
    while time.monotonic() < deadline:
        current_url = str(getattr(page, "url", "") or "").split("?", 1)[0].rstrip("/")
        if current_url and current_url != login_base:
            return True
        try:
            email_handle = await page.query_selector(email_selector)
            pass_handle = await page.query_selector(pass_selector)
        except Exception:
            email_handle = None
            pass_handle = None
        if not email_handle and not pass_handle:
            return True
        await page.wait_for_timeout(1000)
    return False


def _get_login_probe_url(organization: str, login_url: str) -> str:
    """Return the best page to probe when checking whether saved auth is reusable."""
    org = str(organization or "").strip().lower()
    if org == "eventbrite":
        return "https://www.eventbrite.com/organizer/home/"
    return login_url


def is_social_media_url(url: str) -> bool:
    """
    Return True for Facebook/Instagram URLs that rd_ext must never crawl.
    """
    return is_social_url(url)

def get_db_handler():
    """Get or create the global db_handler instance."""
    global db_handler
    if db_handler is None:
        llm_handler = LLMHandler(get_config_path())
        db_handler = llm_handler.db_handler
    return db_handler


def extract_edge_case_url_series(df: pd.DataFrame) -> pd.Series:
    """
    Return the URL series from edge-case input data.

    Supports the expected named-column format and the legacy positional format.
    """
    if "url" in df.columns:
        return df["url"].astype(str)
    if df.shape[1] >= 3:
        return df.iloc[:, 2].astype(str)
    return pd.Series(dtype=str)


def validate_edge_case_social_url_ownership(df: pd.DataFrame) -> int:
    """
    Ensure rd_ext edge_cases input does not include social URLs owned by other steps.

    Returns:
        int: count of disallowed social rows detected.
    """
    urls = extract_edge_case_url_series(df)
    if urls.empty:
        return 0
    return int(urls.map(is_social_media_url).sum())


class ReadExtract:
    def __init__(self, config_path=None):
        self.config = load_config(config_path)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.logged_in = False

    def _next_weekday_date(self, weekday_name: str, include_today: bool = True) -> date:
        """
        Return the next date for the given weekday name.
        """
        weekday_map = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        target = weekday_map.get(str(weekday_name).strip().lower())
        if target is None:
            raise ValueError(f"Unsupported weekday: {weekday_name}")

        today = date.today()
        days_ahead = (target - today.weekday() + 7) % 7
        if days_ahead == 0 and not include_today:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    def _resolve_synthetic_event_base(self, rule: dict) -> dict:
        """
        Resolve base synthetic event payload from an inline `event` block.
        """
        inline_event = rule.get("event")
        if isinstance(inline_event, dict):
            return inline_event.copy()

        return {}
        

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
                browser = p.chromium.launch(
                    headless=_safe_bool(self.config.get('crawling', {}).get('headless', True))
                )
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
        self.browser = await self.playwright.chromium.launch(
            headless=_safe_bool(self.config.get('crawling', {}).get('headless', True))
        )
        # Set a realistic User-Agent to avoid being detected as a bot
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
        self.context = await self.browser.new_context(user_agent=user_agent)
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
        probe_url = _get_login_probe_url(organization, login_url)

        # ──────────── Quick bail if already logged in ────────────
        if self.logged_in:
            logging.info(f"login_to_website(): Already logged in to {organization}, reusing session.")
            return True

        # ──────────── 1) Try to reuse existing session ────────────
        try:
            ctx = await self.browser.new_context(storage_state=storage_path)
            page = await ctx.new_page()
            await page.goto(probe_url, wait_until="domcontentloaded", timeout=20000)
            # If we land off the login page, session is valid
            if login_url.split("?")[0].rstrip("/") not in str(page.url or "").split("?", 1)[0].rstrip("/"):
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
            completed = await _wait_for_login_completion(
                self.page,
                login_url=login_url,
                email_selector=email_selector,
                pass_selector=pass_selector,
                timeout_ms=30000,
            )
            if not completed:
                logging.error(
                    "login_to_website(): Eventbrite login did not finish after manual 2FA confirmation."
                )
                return False

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
        if is_social_media_url(url):
            logging.info("login_if_required(): Skipping social media URL (fb/ig): %s", url)
            return False
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
        if is_social_media_url(link):
            logging.info("extract_event_text: Skipping social media URL (fb/ig): %s", link)
            return None
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

        if is_social_media_url(url):
            logging.info("extract_from_url(): Skipping social media URL (fb/ig): %s", url)
            try:
                child_row = [url, '', 'rd_ext_social_skip', [], False, 1, datetime.now()]
                get_db_handler().write_url_to_db(child_row)
            except Exception as e:
                logging.warning("extract_from_url(): Failed recording social URL skip for %s: %s", url, e)
            return None

        # Check urls to see if they should be scraped
        if not get_db_handler().should_process_url(url):
            logging.info(f"extract_from_url(): Skipping URL {url} based on historical relevancy.")
            return None

        # Special handling for calendar/events pages - extract individual event links
        # This avoids LLM timeout issues by processing events individually instead of all at once
        # Supports: The Coda, The Loft, The Duke Saloon, and other venues with similar structure
        calendar_venues = {
            'gotothecoda.com/calendar': 'The Coda',
            'loftpubvictoria.com/events/month': 'The Loft',
            'thedukesaloon.com': 'The Duke Saloon',
        }

        for venue_url_pattern, venue_name in calendar_venues.items():
            if venue_url_pattern in url:
                logging.info(f"extract_from_url(): Detected {venue_name}, using calendar event extraction")
                event_data = await self.extract_calendar_events(url, venue_name=venue_name)
                if event_data:
                    results = {event_url: event_text for event_url, event_text in event_data}
                    logging.info(f"extract_from_url(): Returning {len(results)} {venue_name} events as dict for multi-event processing")
                    return results
                else:
                    logging.warning(f"extract_from_url(): No {venue_name} events extracted from {url}, returning None")
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
            if is_social_media_url(link):
                logging.info("extract_from_url(): Skipping discovered social media URL (fb/ig): %s", link)
                try:
                    child_row = [link, url, 'rd_ext_social_skip', [], False, 1, datetime.now()]
                    get_db_handler().write_url_to_db(child_row)
                except Exception as e:
                    logging.warning(
                        "extract_from_url(): Failed recording discovered social URL skip for %s: %s",
                        link,
                        e,
                    )
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
            if is_social_media_url(abs_url):
                logging.info("extract_links(): Skipping social media URL (fb/ig): %s", abs_url)
                continue
            # Same domain only
            if parsed.netloc != urlparse(base).netloc:
                continue
            # Skip calendar/ICS feeds
            # e.g., query params outlook-ical, webcal links, .ics endpoints
            if is_calendar_export_url(abs_url):
                logging.info(f"Skipping calendar feed link: {abs_url}")
                continue
            found.add(abs_url)

        return list(found)

    async def extract_calendar_events(self, calendar_url: str, venue_name: str = "Calendar") -> list:
        """
        Generic method to extract individual event links from a calendar/events listing page.

        This method works for any venue that displays events as individual pages:
        - The Coda: https://gotothecoda.com/calendar → /show/{id}
        - The Loft: https://loftpubvictoria.com/events/month/ → /event/{name}/{date}/
        - Others with similar structure

        Process:
        1. Navigates to the calendar/listing page
        2. Extracts all event links from the page
        3. Visits each event individually
        4. Extracts text from each event page
        5. Returns list of (event_url, event_text) tuples

        Args:
            calendar_url (str): The calendar/events listing page URL
            venue_name (str): Venue name for logging (e.g., "The Coda", "The Loft")

        Returns:
            list: List of tuples (event_url, event_text) for each event found
        """
        event_data = []

        try:
            logging.info(f"extract_calendar_events({venue_name}): Starting extraction from {calendar_url}")

            # Navigate to the calendar page
            if not await self.login_if_required(calendar_url):
                logging.error(f"extract_calendar_events({venue_name}): Login failed for {calendar_url}")
                return event_data

            await self.page.goto(calendar_url, timeout=15000)
            await self.page.wait_for_load_state("domcontentloaded")

            # Extract all event links from the page
            all_links = await self.page.query_selector_all("a")
            logging.info(f"extract_calendar_events({venue_name}): Found {len(all_links)} total links on page")

            # Look for links to individual event pages
            # Generic patterns: /show/, /event, /events/ (but exclude navigation links)
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
                    if is_social_media_url(href):
                        logging.info(
                            "extract_calendar_events(%s): Skipping social media event URL (fb/ig): %s",
                            venue_name,
                            href,
                        )
                        continue
                    if is_calendar_export_url(href):
                        logging.info(
                            "extract_calendar_events(%s): Skipping calendar export link: %s",
                            venue_name,
                            href,
                        )
                        continue
                    if ('/show/' in href or '/event' in href.lower() or '/events/' in href) and href != calendar_url:
                        unique_urls.add(href)

            event_links = list(unique_urls)
            logging.info(f"extract_calendar_events({venue_name}): Filtered to {len(event_links)} unique event URLs")

            if not event_links:
                logging.warning(f"extract_calendar_events({venue_name}): No event links found on {calendar_url}")
                return event_data

            # Process each event link
            for idx, event_url in enumerate(event_links, 1):
                try:
                    logging.info(f"extract_calendar_events({venue_name}): [{idx}/{len(event_links)}] Processing {event_url}")

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
                        logging.info(f"extract_calendar_events({venue_name}): Extracted {len(event_text)} chars from event {idx}")
                    else:
                        logging.warning(f"extract_calendar_events({venue_name}): No text extracted from {event_url}")

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logging.error(f"extract_calendar_events({venue_name}): Failed to process event link {idx}: {e}")
                    continue

            logging.info(f"extract_calendar_events({venue_name}): Successfully extracted {len(event_data)} events")
            return event_data

        except Exception as e:
            logging.error(f"extract_calendar_events({venue_name}): Failed to extract events: {e}")
            return event_data



    def add_configured_synthetic_events(self):
        """
        Add synthetic events defined in config.normalization.synthetic_events.
        """
        rules = (
            self.config
            .get("normalization", {})
            .get("synthetic_events", [])
        )
        if not isinstance(rules, list):
            logging.warning("add_configured_synthetic_events(): Expected list, got %s", type(rules).__name__)
            return

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not rule.get("enabled", True):
                continue

            recurrence = rule.get("recurrence", {})
            if recurrence.get("type") != "weekly_weekday":
                logging.warning(
                    "add_configured_synthetic_events(): Unsupported recurrence type '%s' for rule '%s'",
                    recurrence.get("type"),
                    rule.get("name", "unnamed_rule"),
                )
                continue

            event_dict = self._resolve_synthetic_event_base(rule)
            if not event_dict:
                logging.warning(
                    "add_configured_synthetic_events(): Missing event payload for rule '%s'",
                    rule.get("name", "unnamed_rule"),
                )
                continue

            weekday_name = recurrence.get("weekday", "")
            include_today = bool(recurrence.get("include_today", True))
            try:
                target_day = self._next_weekday_date(weekday_name, include_today=include_today)
            except ValueError as e:
                logging.warning(
                    "add_configured_synthetic_events(): %s (rule '%s')",
                    e,
                    rule.get("name", "unnamed_rule"),
                )
                continue

            event_dict['start_date'] = target_day.isoformat()
            event_dict['end_date'] = event_dict['start_date']

            df = pd.DataFrame([event_dict])
            get_db_handler().write_events_to_db(
                df,
                url=event_dict.get('url', ''),
                parent_url='',
                source=event_dict.get('source', ''),
                keywords=event_dict.get('dance_style', ''),
            )
            logging.info(
                "add_configured_synthetic_events(): Added synthetic event '%s' for %s.",
                rule.get("name", "unnamed_rule"),
                target_day.isoformat(),
            )

    def uvic_rueda(self):
        """
        Backward-compatible wrapper. Uses config-driven synthetic event rules.
        """
        self.add_configured_synthetic_events()


    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def _process_edge_case_urls(
    read_extract: ReadExtract,
    llm_handler: LLMHandler,
    db_handler: DatabaseHandler,
    df: pd.DataFrame,
) -> None:
    """Process edge-case URLs with a single shared browser session."""
    all_keywords = [str(k).strip().lower() for k in (llm_handler.get_keywords() or []) if str(k).strip()]
    await read_extract.init_browser()
    logging.info("run_rd_ext_edge_cases(): Browser initialized once for all URLs")
    try:
        for source, keywords, url, multiple in df.itertuples(index=False, name=None):
            multiple_flag = str(multiple).strip().lower() == "yes"
            logging.info(
                "run_rd_ext_edge_cases(): url=%s source=%s keywords=%s multiple=%s",
                url,
                source,
                keywords,
                multiple_flag,
            )
            if is_social_media_url(url):
                logging.info("run_rd_ext_edge_cases(): Skipping social media URL (fb/ig): %s", url)
                db_handler.write_url_to_db([url, "", source, [], False, 1, datetime.now()])
                _record_rd_ext_scrape_metric(
                    db_handler,
                    link=url,
                    parent_url="",
                    source=source,
                    keywords=keywords,
                    access_attempted=False,
                    extraction_attempted=False,
                    extraction_succeeded=False,
                    extraction_skipped=True,
                    decision_reason="social_media_skip",
                    access_succeeded=False,
                    text_extracted=False,
                    keywords_found=False,
                    events_written=0,
                )
                continue
            route = evaluate_step_ownership(url, current_step="rd_ext.py", explicit_edge_case=True)
            if not route.allow:
                logging.info(
                    "run_rd_ext_edge_cases(): Skipping URL owned by %s (%s): %s",
                    route.owner_step,
                    route.routing_reason,
                    url,
                )
                db_handler.write_url_to_db([url, "", source, [], False, 1, datetime.now(), route.routing_reason])
                _record_rd_ext_scrape_metric(
                    db_handler,
                    link=url,
                    parent_url="",
                    source=source,
                    keywords=keywords,
                    access_attempted=False,
                    extraction_attempted=False,
                    extraction_succeeded=False,
                    extraction_skipped=True,
                    decision_reason=route.routing_reason,
                    access_succeeded=False,
                    text_extracted=False,
                    keywords_found=False,
                    events_written=0,
                )
                continue
            extracted = await read_extract.extract_from_url(url, multiple_flag)

            # If multiple events were found (i.e. extracted is a dict), process each event separately.
            if isinstance(extracted, dict):
                followed_count = 0
                successful_events = 0
                for event_url, text in extracted.items():
                    event_route = evaluate_step_ownership(
                        event_url,
                        current_step="rd_ext.py",
                        explicit_edge_case=True,
                    )
                    if not event_route.allow:
                        logging.info(
                            "run_rd_ext_edge_cases(): Skipping extracted event URL owned by %s (%s): %s",
                            event_route.owner_step,
                            event_route.routing_reason,
                            event_url,
                        )
                        db_handler.write_url_to_db(
                            [event_url, url, source, [], False, 1, datetime.now(), event_route.routing_reason]
                        )
                        _record_rd_ext_scrape_metric(
                            db_handler,
                            link=event_url,
                            parent_url=url,
                            source=source,
                            keywords=keywords,
                            access_attempted=False,
                            extraction_attempted=False,
                            extraction_succeeded=False,
                            extraction_skipped=True,
                            decision_reason=event_route.routing_reason,
                            access_succeeded=False,
                            text_extracted=False,
                            keywords_found=False,
                            events_written=0,
                        )
                        continue
                    followed_count += 1
                    parent_url = url
                    prompt_type = resolve_prompt_type(event_url, fallback_prompt_type=url)
                    normalized_text = str(text or "")
                    found_keywords = any(kw in normalized_text.lower() for kw in all_keywords)
                    llm_result = llm_handler.process_llm_response(
                        event_url,
                        parent_url,
                        normalized_text,
                        source,
                        keywords,
                        prompt_type=prompt_type,
                    )
                    llm_success = bool(llm_result)
                    llm_events_written = int(getattr(llm_result, "events_written", int(llm_success)))
                    successful_events += llm_events_written
                    _record_rd_ext_scrape_metric(
                        db_handler,
                        link=event_url,
                        parent_url=url,
                        source=source,
                        keywords=keywords,
                        access_attempted=True,
                        extraction_attempted=True,
                        extraction_succeeded=llm_success,
                        extraction_skipped=False,
                        decision_reason="llm_success" if llm_success else ("no_keywords" if not found_keywords else "llm_no_events"),
                        access_succeeded=bool(normalized_text),
                        text_extracted=bool(normalized_text),
                        keywords_found=found_keywords,
                        events_written=llm_events_written,
                    )
                _record_rd_ext_scrape_metric(
                    db_handler,
                    link=url,
                    parent_url="",
                    source=source,
                    keywords=keywords,
                    access_attempted=True,
                    extraction_attempted=True,
                    extraction_succeeded=successful_events > 0,
                    extraction_skipped=False,
                    decision_reason="multi_event_processed" if successful_events > 0 else "multi_event_no_events",
                    access_succeeded=bool(extracted),
                    text_extracted=bool(extracted),
                    keywords_found=successful_events > 0,
                    events_written=successful_events,
                    links_discovered=len(extracted),
                    links_followed=followed_count,
                )
            else:
                parent_url = ""
                prompt_type = resolve_prompt_type(url, fallback_prompt_type=url)
                normalized_text = str(extracted or "")
                found_keywords = any(kw in normalized_text.lower() for kw in all_keywords)
                llm_result = None
                if normalized_text:
                    llm_result = llm_handler.process_llm_response(
                        url,
                        parent_url,
                        normalized_text,
                        source,
                        keywords,
                        prompt_type=prompt_type,
                    )
                llm_success = bool(llm_result)
                events_written = int(getattr(llm_result, "events_written", int(llm_success))) if llm_result else 0
                _record_rd_ext_scrape_metric(
                    db_handler,
                    link=url,
                    parent_url="",
                    source=source,
                    keywords=keywords,
                    access_attempted=True,
                    extraction_attempted=True,
                    extraction_succeeded=llm_success,
                    extraction_skipped=False,
                    decision_reason="llm_success" if llm_success else ("no_text" if not normalized_text else ("no_keywords" if not found_keywords else "llm_no_events")),
                    access_succeeded=bool(normalized_text),
                    text_extracted=bool(normalized_text),
                    keywords_found=found_keywords,
                    events_written=events_written,
                )
    finally:
        await read_extract.close()
        logging.info("run_rd_ext_edge_cases(): Browser closed")


def run_rd_ext_edge_cases() -> None:
    """Thin script runner for rd_ext edge-case processing."""
    from logging_config import setup_logging

    setup_logging("rd_ext")
    config = load_config()
    runtime_config_path = get_config_path()

    logging.info("\n\nrd_ext.py starting...")
    start_time = datetime.now()
    logging.info("run_rd_ext_edge_cases(): starting at %s", start_time)

    read_extract = ReadExtract(runtime_config_path)
    llm_handler = LLMHandler(runtime_config_path)
    local_db_handler = llm_handler.db_handler

    # Set module-level handler for helper functions using get_db_handler().
    global db_handler
    db_handler = local_db_handler

    file_name = os.path.basename(__file__)
    start_df = local_db_handler.count_events_urls_start(file_name)

    df = pd.read_csv(config["input"]["edge_cases"])
    disallowed_social = validate_edge_case_social_url_ownership(df)
    if disallowed_social > 0:
        msg = (
            f"edge_cases.csv contains {disallowed_social} social URL(s) (fb/ig), "
            "which are owned by fb.py/images.py and must not be processed by rd_ext.py."
        )
        logging.error("run_rd_ext_edge_cases(): %s", msg)
        raise ValueError(msg)

    asyncio.run(_process_edge_case_urls(read_extract, llm_handler, local_db_handler, df))
    read_extract.add_configured_synthetic_events()
    local_db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info("run_rd_ext_edge_cases(): finished at %s", end_time)
    logging.info("run_rd_ext_edge_cases(): total time=%s", end_time - start_time)


if __name__ == "__main__":
    run_rd_ext_edge_cases()
