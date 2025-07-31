"""
clean_up.py

This module defines the CleanUp class, which performs data cleanup on event records 
that do not have URLs. The process:

1. Retrieves events without URLs from the database.
2. For each such event, performs a (blocking) Google search using the event name to find 
   a relevant URL (fuzzy matching + ranking).
3. Uses Playwright's async API to extract text from the found URL.
4. Checks if the extracted text is relevant (LLMHandler).
5. If relevant, queries the LLM for structured event info.
6. Merges with the existing DB record.
7. Updates the DB and writes the found URL to the URLs table.

Dependencies:
    - pandas, fuzzywuzzy, logging, googleapiclient, etc.
    - db.DatabaseHandler, fb.FacebookEventScraper, gs.GoogleSearch, 
      llm.LLMHandler, credentials.get_credentials
"""

import asyncio
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime, timedelta
from dotenv import load_dotenv
from googleapiclient.discovery import build
import json
import logging
import numpy as np
import os
import pandas as pd
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import random
from rapidfuzz import fuzz
import requests
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from tabulate import tabulate
from typing import List, Dict, Any
import unicodedata
import urllib.parse
import yaml

from db import DatabaseHandler
from llm import LLMHandler
from credentials import get_credentials
import os

USER_AGENTS = [
    # Chrome on Windows 10
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",

    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",

    # Firefox on Windows 10
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",

    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13.5; rv:126.0) Gecko/20100101 Firefox/126.0",

    # Edge on Windows 11
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.2535.79",

    # Safari on macOS Ventura
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",

    # Brave on Windows (uses Chrome user agent)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",

    # Opera on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 OPR/109.0.5097.38",

    # Vivaldi on Windows (also Chromium-based)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Vivaldi/6.7.3329.29",

    # Generic latest Chrome on Linux (useful for headless scraping profiles)
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
]


class CleanUp:
    """
    CleanUp class for managing event data cleanup operations.
    """
    def __init__(self, config):
        """
        Initialize the class with configuration, database handler, LLM handler, and Google API credentials.
        Args:
            config (dict): Configuration dictionary for initializing handlers and connections.
        Raises:
            ConnectionError: If the database connection cannot be established.
        Attributes:
            config (dict): The configuration dictionary.
            db_handler (DatabaseHandler): Handler for database operations.
            llm_handler (LLMHandler): Handler for language model operations.
            conn (Any): Active database connection object.
            api_key (str): Google API key retrieved from credentials.
            cse_id (str): Google Custom Search Engine ID.
            browser (Any): Reference to the async browser instance (initialized later).
            context (Any): Reference to the browser context (initialized later).
            logged_in_page (Any): Reference to the logged-in browser page (initialized later).
        """
        self.config = config
        self.db_handler = DatabaseHandler(config)
        self.llm_handler = LLMHandler(config_path="config/config.yaml")

        # Establish database connection
        self.conn = self.db_handler.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        
        # Get Brave API key from .env
        load_dotenv()
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        if not self.brave_api_key:
            raise ValueError("BRAVE_API_KEY not found in environment variables.")

        # Retrieve Google API credentials using credentials.py
        _, self.api_key, self.cse_id = get_credentials('Google')

        # We'll store references to the async browser/page once we init them
        self.browser = None
        self.context = None
        self.logged_in_page = None


    def brave_search(self, event_name):
        """
        Performs a Brave Search API query using the provided event name and location.
        Returns a DataFrame with columns ['event_name', 'url'] or an empty one on failure.
        """
        location = self.config['location']['epicentre']
        query = f"{event_name} {location} address"
        logging.info(f"brave_search(): Searching for '{query}' using Brave Search API.")
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"Accept": "application/json", "X-Subscription-Token": self.brave_api_key}
        params = {"q": query, "count": self.config['search']['gs_num_results']}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({"event_name": item.get("title", ""), "url": item.get("url", "")})
                logging.info(f"brave_search(): Found result: {results}")
            return pd.DataFrame(results)
        
        except Exception as e:
            logging.error(f"brave_search(): Failed to fetch results for {query}: {e}")
            return pd.DataFrame(columns=["event_name", "url"])


    async def login_to_facebook(self, page, browser) -> bool:
            """
            Asynchronously logs into Facebook using credentials, handles login prompts,
            and saves session state for future reuse. Manual steps may be required for captcha.

            Args:
                page (playwright.async_api.Page): The Playwright page instance.
                browser (playwright.async_api.Browser): The Playwright browser instance.

            Returns:
                bool: True if login is successful, False otherwise.
            """
            try:
                context = await browser.new_context(storage_state="auth.json")
                page = await context.new_page()
                await page.goto("https://www.facebook.com/", timeout=60000)

                # If "login" not in the URL, we assume we're already logged in from storage_state
                if "login" not in page.url.lower():
                    logging.info("Loaded existing session (async). Already logged into Facebook.")
                    self.logged_in_page = page
                    return True
            except Exception:
                logging.info("No valid saved session found. Proceeding with manual async login...")

            # Fallback: manual login
            email, password, _ = get_credentials('Facebook')  # or however you retrieve credentials

            await page.goto("https://www.facebook.com/login", timeout=60000)
            await page.fill("input[name='email']", email)
            await page.fill("input[name='pass']", password)
            await page.click("button[name='login']")
            await page.wait_for_timeout(5000)

            async def check_additional_login(p):
                email_visible = await p.is_visible("input[name='email']")
                pass_visible = await p.is_visible("input[name='pass']")
                if email_visible and pass_visible:
                    logging.info("Additional login prompt detected. Re-filling credentials (async).")
                    await p.fill("input[name='email']", email)
                    await p.fill("input[name='pass']", password)
                    await p.click("button[name='login']")
                    await p.wait_for_timeout(5000)
                    return True
                return False

            # Attempt to handle subsequent prompts
            attempts = 0
            while attempts < 3 and (await check_additional_login(page)):
                attempts += 1

            # Pause for manual captcha/challenge
            logging.warning("Please solve any captcha/challenge in the browser. Pausing...")
            print("Press Enter here once done.")
            await asyncio.get_running_loop().run_in_executor(None, input)  # non-blocking approach

            # Check if we ended up logged in
            await page.wait_for_timeout(5000)
            if "login" in page.url.lower():
                logging.error("Async login failed. Check credentials or captcha challenges.")
                return False

            # Save session
            try:
                context = page.context
                await context.storage_state(path="auth.json")
                logging.info("Session state saved (async).")
            except Exception as e:
                logging.warning(f"Could not save session state (async): {e}")

            self.logged_in_page = page
            logging.info("Async login to Facebook successful.")
            return True
    

    async def extract_text_from_fb_url(self, url: str) -> str | None:
        """
        Asynchronously extracts text content from a Facebook event URL using
        Playwright and BeautifulSoup.

        Args:
            url (str): The Facebook event URL.

        Returns:
            str or None: Extracted text content.
        """
        if not self.browser or not self.logged_in_page:
            logging.error("Browser or logged_in_page not initialized in CleanUp.")
            return None

        fb_status = await self.login_to_facebook(self.logged_in_page, self.browser)
        if not fb_status:
            logging.warning("extract_text_from_fb_url: Could not log in to Facebook (async).")
            return None

        logging.info("extract_text_from_fb_url (async): Logged into Facebook.")
        extracted_text = await self.extract_event_text(url)
        return extracted_text
    

    async def extract_event_text(self, link: str) -> str | None:
        """
        Asynchronously extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            link (str): The event link.

        Returns:
            str or None: The extracted relevant text content, or None if not found.
        """
        if not self.logged_in_page:
            logging.error("No logged_in_page available. Did you call login_to_facebook (async)?")
            return None

        try:
            page = self.logged_in_page  # or create a fresh page if you prefer
            await page.goto(link, timeout=10000)

            # Look for "See more" buttons
            more_buttons = await page.query_selector_all("text=/See more/i")
            for mb in more_buttons:
                try:
                    await mb.wait_for_element_state("stable", timeout=3000)
                    await mb.click()
                    await page.wait_for_timeout(random.randint(3000, 5000))
                    logging.info(f"Clicked 'See more' button in URL: {link}")
                except Exception as e:
                    logging.warning(f"Could not click 'See more' in {link}: {e}")

            if not more_buttons:
                logging.debug(f"No 'See more' buttons found in URL: {link}")

            await page.wait_for_timeout(5000)
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            extracted_text = ' '.join(soup.stripped_strings)
            logging.info(f"(async) raw text from {link}: {extracted_text}")

            if extracted_text:
                # If you have a separate method to refine the text:
                extracted_text = self.extract_relevant_text(extracted_text, link)
                if extracted_text:
                    logging.info(f"(async) def extract_event_text(): relevant text from {link}: {extracted_text}")
                    return extracted_text
                else:
                    logging.info(f"(async) def extract_event_text(): No relevant text found in {link}.")
                    return None
            else:
                logging.info(f"(async) def extract_event_text(): No text extracted from {link}.")
                return None

        except Exception as e:
            logging.error(f"(async) def extract_event_text(): Failed to extract text from {link}: {e}")
            return None
    

    def extract_relevant_text(self, content, link):
        """
        Extracts a relevant portion of text from the given content based on specific patterns.

        This function searches for the first occurrence of "More About Discussion" in the content,
        then finds the last occurrence of a day of the week before this phrase, and finally extracts
        the text from this day up to the phrase "Guests See All".

        Args:
            content (str): The text content to be processed.

        Returns:
            str: The extracted relevant text if all patterns are found, otherwise None.

        Logs:
            Logs warnings if any of the required patterns ("More About Discussion", a day of the week,
            or "Guests See All") are not found in the expected positions within the content.
        """

        # Step 1: Find the first occurrence of "More About Discussion" (case-insensitive)
        mad_pattern = re.compile(r"More About Discussion", re.IGNORECASE)
        mad_match = mad_pattern.search(content)
        if not mad_match:
            logging.warning(f"'More About Discussion' not found in {link}.")
            return None

        mad_start = mad_match.start()

        # Step 2: In the text before "More About Discussion", find the last occurrence of a day of the week
        days_pattern = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)
        days_matches = list(days_pattern.finditer(content, 0, mad_start))
        if not days_matches:
            logging.warning(f"No day of the week found before 'More About Discussion' in {link}.")
            return None

        last_day_match = days_matches[-1]
        day_start = last_day_match.start()

        # Step 3: From the last day match, extract up to "Guests See All"
        gsa_pattern = re.compile(r"Guests See All", re.IGNORECASE)
        gsa_match = gsa_pattern.search(content, last_day_match.end())
        if not gsa_match:
            logging.warning(f"'Guests See All' not found after last day of the week in {link}.")
            return None

        gsa_end = gsa_match.end()

        # Extract the desired text
        extracted_text = content[day_start:gsa_end]

        return extracted_text
    

    async def process_events_without_url(self):
        """
        Asynchronously processes events without URLs by:
          1) Querying the DB for events with no URL.
          2) Doing a Google search to find candidate URLs.
          3) Extracting text from the best candidate URL using async Playwright.
          4) Checking relevance (LLMHandler).
          5) If relevant, fetching structured info from LLM.
          6) Merging/Updating DB records.
        """
        query = "SELECT * FROM events WHERE url = '' OR url IS NULL"
        no_urls_df = pd.read_sql(query, self.db_handler.conn)
        logging.info(
            f"def process_events_without_url(): Found {no_urls_df.shape[0]} events without URLs."
        )

        if no_urls_df.empty:
            logging.info("def process_events_without_url(): All events have a URL.")
            return

        # Optionally reduce dataframe size for testing
        if self.config['testing']['status']:
            no_urls_df = no_urls_df.head(self.config['testing']['nof_no_url_events'])

        delete_count = 0
        # Process each event
        for event_row in no_urls_df.itertuples(index=False):
            event_name = event_row.event_name

            # 1) Brave search (synchronous, but that's usually fine)
            results_df = self.serpapi_search(event_name)

            # 2) Find best URL
            best_url = self.find_best_url_for_event(event_name, results_df)
            if not best_url:
                logging.info(f"def process_events_without_url(): No URL found for event: {event_name}")
                continue

            # 3) Asynchronously extract text from the URL
            extracted_text = await self.extract_text_with_playwright_async(best_url)
            logging.info(f"def process_events_without_url(): Extracted text from URL: {best_url}")

            # 4) Check relevance
            source = event_row.source
            keywords_list = event_row.dance_style.split(",") if event_row.dance_style else []
            relevant = self.llm_handler.check_keywords_in_text(
                best_url, extracted_text, source, keywords_list
            )

            if relevant:
                # 5) If relevant, determine prompt type and possibly refine text (if facebook)
                prompt_type = "fb" if "facebook" in best_url.lower() or "instagram" in best_url.lower() else "default"
                if prompt_type == "fb":
                    # Refine extracted text
                    extracted_text = self.extract_text_from_fb_url(best_url)

                prompt = self.llm_handler.generate_prompt(best_url, extracted_text, prompt_type)
                llm_response = self.llm_handler.query_llm(best_url, prompt)

                if llm_response:
                    parsed_result = self.llm_handler.extract_and_parse_json(llm_response, best_url)
                    if parsed_result:
                        events_df = pd.DataFrame(parsed_result)
                        events_df["url"] = best_url
                        events_df.to_csv("debug/events_df.csv", index=False)

                        # 6) Merge data if new row is a good match
                        new_row = self.select_best_match(event_name, events_df)
                        if new_row is not None:
                            merged_row = self.merge_rows(event_row, new_row)
                            self.update_event_row(merged_row)

                            source = merged_row["source"]
                            keywords = merged_row["dance_style"]
                            # Write URL to db
                            self.db_handler.write_url_to_db(source, keywords, best_url, "", True, 1)

            else:
                logging.info(f"def process_events_without_url(): Event {event_name} and {event_row.event_id} is not relevant.")
                self.db_handler.delete_event_with_event_id(int(event_row.event_id))
                delete_count += 1

        logging.info("def process_events_without_url(): Deleted {delete_count} events without URLs.")
        logging.info("def process_events_without_url(): Finished processing events without URLs.")


    def find_best_url_for_event(self, original_event_name, results_df):
        """
        Finds the best URL for an event based on the original event name and a DataFrame of results.

        The function uses fuzzy string matching to compare the original event name with event names in the results DataFrame.
        It ranks URLs based on their source, giving preference to 'allevents' URLs over normal URLs, and normal URLs over 'facebook' or 'instagram' URLs.

        Parameters:
        original_event_name (str): The name of the original event to match against.
        results_df (pandas.DataFrame): A DataFrame containing event names and URLs to search through.

        Returns:
        str: The best matching URL if a match is found with a score of 80 or higher, otherwise an empty string.
        """

        logging.info(f"def find_best_url_for_event(): original_event_name: {original_event_name}")
        best_match = None
        best_score = 0
        best_rank = 0

        for row in results_df.itertuples(index=False):
            logging.info(F"def find_best_url_for_event(): row: {row}")
            score = fuzz.ratio(original_event_name, row.event_name)
            logging.info(F"def find_best_url_for_event(): score: {score}")
            if score < self.config['fuzzywuzzy']['hurdle']:
                continue

            url = str(row.url).lower()
            # Ranking logic: 'allevents' > normal > 'facebook'/'instagram'
            if "allevents" in url:
                rank = 3
            elif "facebook" in url or "instagram" in url:
                rank = 1
            else:
                rank = 2

            if (score > best_score) or (score == best_score and rank > best_rank):
                best_score = score
                best_rank = rank
                best_match = row

        return best_match.url if best_match is not None else ""
    

    async def extract_text_with_playwright_async(self, url: str) -> str:
        """
        Asynchronously extracts visible text content from a web page using Playwright.
        Navigates to the specified URL, waits for the page to render, and parses the HTML to extract all visible text.
        If a Google sign-in page or sign-in input is detected, or if any error occurs (including timeouts), returns an empty string.
        Args:
            url (str): The URL of the web page to extract text from.
        Returns:
            str: The extracted visible text from the web page, or an empty string if extraction fails or a sign-in page is detected.
        Raises:
            None: All exceptions are handled internally and logged.
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=self.config['crawling']['headless'])
                page = await browser.new_page()

                # Navigate to the URL with a 10-second timeout
                await page.goto(url, timeout=10000)

                # Wait an extra 3 seconds for the page to render
                await page.wait_for_timeout(3000)

                # Detect Google sign-in
                if "accounts.google.com" in page.url.lower():
                    logging.info("Google sign-in detected. Aborting extraction.")
                    await browser.close()
                    return ""

                # Check for sign-in input
                sign_in_input = await page.query_selector("input#identifierId")
                if sign_in_input:
                    logging.info("Google sign-in input detected. Aborting extraction.")
                    await browser.close()
                    return ""

                # Get the page's HTML content
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                extracted_text = " ".join(soup.stripped_strings)
                logging.info(f"def extract_text_with_playwright_async(): Extracted text from {url}: {extracted_text}")

                await browser.close()
                return extracted_text
            

        except PlaywrightTimeoutError:
            logging.error(f"Timeout while extracting text from {url}.")
            return ""
        except Exception as e:
            logging.error(f"Failed to extract text from {url}: {e}")
            return ""
        

    def select_best_match(self, original_event_name, events_df):
        """
        Selects the best matching event from a DataFrame based on fuzzy string similarity.

        Args:
            original_event_name (str): The name of the event to match.
            events_df (pd.DataFrame): DataFrame containing event data with an 'event_name' column.

        Returns:
            pandas.Series or None: The row from events_df with the highest fuzzy match score (>= 80),
            or None if no suitable match is found.
        """
        best_match = None
        best_score = 0
        for row in events_df.itertuples(index=False):
            score = fuzz.ratio(original_event_name, row.event_name)
            if score >= 80 and score > best_score:
                best_score = score
                best_match = row
        return best_match
    

    def merge_rows(self, original_row, new_row):
        """
        Merge two rows (namedtuples), preferring values from new_row if they are longer or if the original value is empty.

        For each field in the rows:
            - If the original value is empty or NaN, use the value from new_row.
            - If the new value is not empty or NaN and is longer (as a string) than the original value, use the new value.
            - Otherwise, keep the original value.

        Args:
            original_row (namedtuple): The original row to be updated.
            new_row (namedtuple): The new row with potential updated values.

        Returns:
            dict: A dictionary representing the merged row, with updated values where applicable.
        """
        merged = original_row._asdict()  # Convert namedtuple to dict
        for col in original_row._fields:
            orig_val = getattr(original_row, col)
            new_val = getattr(new_row, col, None) if hasattr(new_row, col) else None
            if pd.isna(orig_val) or orig_val == "":
                merged[col] = new_val
            elif pd.notna(new_val) and new_val != "":
                if len(str(new_val)) > len(str(orig_val)):
                    merged[col] = new_val
        return merged
    

    def update_event_row(self, merged_row):
        """
        Updates a row in the 'events' table with the values from the provided merged_row dictionary.
        Parameters:
            merged_row (dict): A dictionary containing the event data to update. Must include the 'event_id' key
                               to identify the row, and any other columns to be updated as key-value pairs.
        Returns:
            None
        Side Effects:
            Executes an UPDATE SQL query on the 'events' table using the provided database handler.
        """
        event_id = merged_row["event_id"]
        update_columns = [col for col in merged_row if col != "event_id"]
        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        update_params = {col: merged_row[col] for col in update_columns}
        update_params["event_id"] = event_id
        self.db_handler.execute_query(update_query, update_params)


    async def fix_incorrect_dance_styles(self):
        """
        Asynchronously fixes incorrect dance styles in the database by:
        1) Querying the DB for events.
        2) Identifying and updating incorrect dance styles.
        3) Updating the DB with the corrected dance styles.
        """
        events_df = self.fetch_events_from_db()
        if events_df.empty:
            logging.info("fix_incorrect_dance_styles(): No events found in the database.")
            return

        keywords_list = self._load_keywords()
        events_df["update"] = False  # Add update flag column

        events_df[["dance_style", "update"]] = events_df.apply(
            lambda row: self.update_dance_style(row, keywords_list), axis=1
        )

        self._process_updated_events(events_df)


    async def delete_events_outside_bc(self):
        """
        Deletes events outside of BC, Canada from the database.
        """
        sql = """
            SELECT event_id, location
            FROM events
            WHERE location IS NOT NULL OR location = ''
            """
        events_df = pd.read_sql(sql, self.conn)
        
        event_ids_to_be_deleted = []
        for event in events_df.itertuples():

            # Mark USA for deletion
            if event.location and "USA" in event.location:
                event_ids_to_be_deleted.append(event.event_id)
            # Prepare a regex pattern to match USA postal codes
            usa_postal_code_pattern = re.compile(r"\b\d{5}(?:[-\s]\d{4})?\b")
            if usa_postal_code_pattern.search(event.location):
                event_ids_to_be_deleted.append(event.event_id)
            if "Washington" in event.location:
                event_ids_to_be_deleted.append(event.event_id)
            if "Oregon" in event.location:
                event_ids_to_be_deleted.append(event.event_id)

            # Prepare a regex to match UK postal codes
            uk_postal_code_pattern = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b")
            if uk_postal_code_pattern.search(event.location):
                event_ids_to_be_deleted.append(event.event_id)

        # Have the database delete the list of event_id(s)
        if event_ids_to_be_deleted:
            self.db_handler.delete_multiple_events(event_ids_to_be_deleted)
            logging.info(f"delete_events_outside_bc(): Deleted {len(event_ids_to_be_deleted)} events outside of BC.")
        else:
            logging.info("delete_events_outside_bc(): No events found outside of BC.")


    async def delete_events_more_than_9_months_future(self):
        """
        Deletes events more than 270 days in the future from the database.
        """
        sql = """
            SELECT event_id, start_date
            FROM events
            WHERE start_date > CURRENT_DATE + INTERVAL '270 days'
            """
        # Have the database delete the list of event_id(s)
        events_df = pd.read_sql(sql, self.conn)
        event_ids_to_be_deleted = events_df['event_id'].tolist()

        if event_ids_to_be_deleted:
            self.db_handler.delete_multiple_events(event_ids_to_be_deleted)
            logging.info(f"delete_events_more_than_9_months_future(): Deleted {len(event_ids_to_be_deleted)}.")
        else:
            logging.info("delete_events_more_than_9_months_future(): No events found more than 9 months in the future.")


    def known_incorrect(self, known_incorrect_urls) -> None:
        """
        Deletes known incorrect events from the database.
        """
        conditions = " OR ".join(f"url ILIKE '%{kw}%'" for kw in known_incorrect_urls)
        sql = f"""
            DELETE FROM events
            WHERE {conditions};
        """
        rows_deleted = self.db_handler.execute_query(sql)
        if rows_deleted is not None:
            logging.info(
                f"known_incorrect(): Deleted {rows_deleted} event(s) matching keywords: {known_incorrect_urls}"
            )
        else:
            logging.error(
                f"known_incorrect(): Failed to delete events matching keywords: {known_incorrect_urls}"
            )


    def fetch_events_from_db(self):
        """Fetch all events from the database.
        """
        query = "SELECT * FROM events"
        events_df = pd.read_sql(query, self.db_handler.conn)
        logging.info(f"fetch_events_from_db(): Retrieved {events_df.shape[0]} events.")
        return events_df


    def _load_keywords(self):
        """
        Loads dance style keywords from a CSV file specified in the configuration, processes them by splitting comma-separated values, stripping whitespace, removing duplicates, and sorting the final list.

        Returns:
            list: A sorted list of unique dance style keywords as strings.
        """
        keywords_df = pd.read_csv(self.config['input']['data_keywords'])
        keywords_list = sorted(set(
            keyword.strip()
            for keywords in keywords_df["keywords"]
            for keyword in str(keywords).split(',')
        ))
        logging.info(f"_load_keywords(): Loaded {len(keywords_list)} keywords.")
        return keywords_list


    def update_dance_style(self, row, keywords_list):
        """
        Checks if any keywords from the provided list are present in the event's name, description, or source fields.
        If matches are found, returns a pandas Series with the matched keywords as a comma-separated string and a flag set to True,
        indicating that the dance style should be updated. Otherwise, returns the original dance style and update flag.

        Args:
            row (pd.Series): A pandas Series representing a row from the DataFrame, expected to contain 'event_name', 'description', 'source', 'dance_style', and 'update' fields.
            keywords_list (list of str): List of keywords to search for in the event's fields.

        Returns:
            pd.Series: A Series containing the updated dance style (comma-separated matched keywords or original value)
                       and a boolean flag indicating whether an update is needed.
        """
        event_name = row["event_name"] if isinstance(row["event_name"], str) else ""
        description = row["description"] if isinstance(row["description"], str) else ""
        source = row["source"] if isinstance(row["source"], str) else ""

        matches = [keyword for keyword in keywords_list if keyword.lower() in event_name.lower() 
                or keyword.lower() in description.lower() or keyword.lower() in source.lower()]

        if matches:
            return pd.Series([", ".join(matches), True])  # Update dance_style and flag for update
        return pd.Series([row["dance_style"], row["update"]])  # Keep original values


    def _process_updated_events(self, events_df):
        """
        Filters the provided DataFrame for rows marked as updated, processes them, and updates the corresponding records in the database.
        Args:
            events_df (pd.DataFrame): DataFrame containing event records, including an 'update' boolean column indicating which rows require updating.
        Behavior:
            - Filters rows where 'update' is True.
            - Drops the 'update' column from the filtered DataFrame.
            - Ensures 'address_id' column is filled with 0 where missing and cast to integer type.
            - Converts the filtered DataFrame to a list of dictionaries for batch database insertion.
            - Calls the database handler to update the 'events' table with the modified records.
            - Logs the number of records updated or if no updates are required.
        """
        updated_df = events_df[events_df["update"]].drop(columns=["update"])
        
        if updated_df.empty:
            logging.info("fix_incorrect_dance_styles(): No updates required.")
            return

        # Ensure that address_id are integers
        updated_df['address_id'] = updated_df['address_id'].fillna(0).astype(int)

        # Convert DataFrame to list of dictionaries for multiple inserts
        values = updated_df.to_dict(orient="records")
        table_name = "events"
        self.db_handler.multiple_db_inserts(table_name, values)
        logging.info(f"_process_updated_events(): Updated {len(values)} records in the database.")


    async def search_google_and_scrape_page(self, location: str) -> str:
        """
        Uses Playwright to search Google and scrape the entire results page.
        Stores and reuses session to avoid repeated CAPTCHAs.
        """
        try:
            async with async_playwright() as p:
                storage_path = "auth/google_auth.json"
                context_args = {}

                if os.path.exists(storage_path):
                    logging.info("Using saved Google session.")
                    context_args["storage_state"] = storage_path

                browser = await p.chromium.launch(headless=False)
                context = await browser.new_context(**context_args)
                page = await context.new_page()

                # Visit Google Search
                query = urllib.parse.quote(location)
                search_url = f"https://www.google.com/search?q={query}"
                logging.info(f"Navigating to Google search URL: {search_url}")
                await page.goto(search_url, timeout=20000)

                # Detect CAPTCHA
                if "sorry" in page.url.lower() or await page.query_selector("form[action^='/sorry/']"):
                    logging.warning("CAPTCHA detected. Please solve manually.")
                    print("\n🔐 CAPTCHA DETECTED 🔐")
                    print("Solve CAPTCHA in browser. Press Enter when complete.")
                    await asyncio.get_running_loop().run_in_executor(None, input)

                # Wait and scroll for more content
                await page.wait_for_timeout(2000)
                await page.keyboard.press("PageDown")
                await page.wait_for_timeout(2000)

                # Extract and save session
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                extracted_text = " ".join(soup.stripped_strings)

                await context.storage_state(path=storage_path)
                logging.info("Saved session to google_auth.json")
                await browser.close()

                return extracted_text

        except Exception as e:
            logging.error(f"search_google_and_scrape_page(): Exception: {e}")
            return ""


    def extract_location_snippet(self, full_text: str, location: str, buffer: int = 500) -> str:
        """
        Extracts a snippet of text surrounding a specified location string within the full text.
        Searches for the first occurrence of the location string (case-insensitive) in the full_text.
        If found, returns a substring of full_text that includes the location and a buffer of characters
        before and after it. If the location is not found, returns an empty string.
        Args:
            full_text (str): The complete text to search within.
            location (str): The location string to find in the text.
            buffer (int, optional): Number of characters to include before and after the location. Defaults to 500.
        Returns:
            str: A snippet of the full_text containing the location and surrounding context, or an empty string if not found.
        """
        index = full_text.lower().find(location.lower())
        if index == -1:
            logging.warning("extract_location_snippet(): Location string not found.")
            return ""
        return full_text[max(0, index - buffer):min(len(full_text), index + len(location) + buffer)]


    def parse_full_address(self, full_address):
        pattern = re.compile(
            r"(?P<street_number>\d+)?\s*"
            r"(?P<street_name>[\w\s]+?),\s*"
            r"(?P<city>[\w\s]+?),\s*"
            r"(?P<province_or_state>[A-Z]{2})\s*"
            r"(?P<postal_code>[A-Z]\d[A-Z]\s?\d[A-Z]\d)?"
        )
        match = pattern.search(full_address)
        if match:
            groups = match.groupdict()
            street_parts = groups.get("street_name", "").strip().split()
            street_name = " ".join(street_parts[:-1]) if len(street_parts) > 1 else groups.get("street_name")
            street_type = street_parts[-1] if len(street_parts) > 1 else None
            return {
                "street_number": groups.get("street_number"),
                "street_name": street_name,
                "street_type": street_type,
                "direction": None,
                "city": groups.get("city"),
                "province_or_state": groups.get("province_or_state"),
                "postal_code": groups.get("postal_code"),
            }
        return {}
    

    async def fix_null_addresses_in_events(self):
        """
        Fixes events in the database that have NULL address_id fields by updating them with valid address information.
        This method performs the following steps:
            1. Checks for the existence and recency (within 3 days) of a CSV file containing events with missing addresses.
            2. Loads and cleans the CSV data, ensuring only rows with non-empty 'full_address' are processed.
            3. Queries the database for events where address_id is NULL.
            4. For each entry in the CSV:
                - If an address_id is provided, updates matching events with this address_id and the full address.
                - If not, checks if the address already exists in the address table; if not, inserts it.
                - Updates events with the new or existing address_id and full address.
            5. Logs all updates to an output CSV file for auditing.
        Logging is used throughout to provide information and error messages. The method is asynchronous and intended to be run as part of a data cleanup or migration process.
        Returns:
            None
        """
        logging.info("Starting fix_null_addresses_in_events process...")

        nulls_csv = self.config['input']['nulls_addresses']

        if not os.path.exists(nulls_csv):
            logging.warning(f"CSV file {nulls_csv} does not exist. Skipping.")
            return

        modified_time = datetime.fromtimestamp(os.path.getmtime(nulls_csv))
        if datetime.now() - modified_time > timedelta(days=3):
            logging.info(f"CSV file {nulls_csv} is older than 3 days. Skipping update.")
            return
        else:
            logging.info(f"CSV file {nulls_csv} has been modified recently. Proceeding.")

        try:
            address_df = pd.read_csv(nulls_csv).fillna("").drop_duplicates()
            address_df = address_df[address_df['full_address'].str.strip() != ""]
        except Exception as e:
            logging.error(f"Could not load address CSV from {nulls_csv}: {e}")
            return

        query = """
            SELECT event_id, location, address_id 
            FROM events 
            WHERE address_id IS NULL
        """
        try:
            events_df = pd.read_sql(query, self.conn)
        except Exception as e:
            logging.error(f"Could not query events with NULL address_id: {e}")
            return

        if events_df.empty:
            logging.info("No events with NULL address_id found.")
            return

        update_log = []
        updated_count = 0
        for row in address_df.itertuples():
            full_address = getattr(row, 'full_address').strip()
            building_name = getattr(row, 'location').strip()
            csv_address_id = str(getattr(row, 'address_id')).strip()

            if not full_address:
                continue

            full_address_with_building = f"{building_name}, {full_address}" if building_name else full_address

            if csv_address_id and csv_address_id.lower() != "nan":
                logging.info(f"Using provided address_id {csv_address_id} for location match '{building_name}'")
                update_sql = """
                    UPDATE events 
                    SET location = :location, address_id = :address_id
                    WHERE address_id IS NULL AND LOWER(location) = LOWER(:building_name)
                    RETURNING event_id
                """
                result = self.db_handler.execute_query(update_sql, {
                    "location": full_address_with_building,
                    "address_id": int(csv_address_id),
                    "building_name": building_name
                })
                updated_event_ids = [r[0] for r in result] if result else []
                for event_id in updated_event_ids:
                    update_log.append({"event_id": event_id, "building_name": building_name, "full_address": full_address_with_building, "address_id": int(csv_address_id)})
                updated_count += len(updated_event_ids)
                continue

            check_sql = "SELECT address_id FROM address WHERE full_address = :full_address"
            result = self.db_handler.execute_query(check_sql, {"full_address": full_address_with_building})

            if result:
                address_id = result[0][0]
                logging.info(f"Address already exists for '{full_address_with_building}', using address_id {address_id}")
            else:
                parsed = self.parse_full_address(full_address)
                address_insert = {
                    "full_address": full_address_with_building,
                    "building_name": building_name if building_name else None,
                    "street_number": parsed.get("street_number"),
                    "street_name": parsed.get("street_name"),
                    "street_type": parsed.get("street_type"),
                    "direction": parsed.get("direction"),
                    "city": parsed.get("city"),
                    "met_area": None,
                    "province_or_state": parsed.get("province_or_state"),
                    "postal_code": parsed.get("postal_code"),
                    "country_id": "CA",
                    "time_stamp": datetime.now()
                }
                insert_sql = """
                    INSERT INTO address 
                    (full_address, building_name, street_number, street_name, street_type, direction, 
                     city, met_area, province_or_state, postal_code, country_id, time_stamp)
                    VALUES 
                    (:full_address, :building_name, :street_number, :street_name, :street_type, :direction, 
                     :city, :met_area, :province_or_state, :postal_code, :country_id, :time_stamp)
                    RETURNING address_id
                """
                inserted = self.db_handler.execute_query(insert_sql, address_insert)
                address_id = inserted[0][0] if inserted else None
                logging.info(f"Inserted new address for '{full_address_with_building}' with address_id {address_id}")

            update_sql = """
                UPDATE events 
                SET location = :location, address_id = :address_id
                WHERE address_id IS NULL AND LOWER(location) = LOWER(:building_name)
                RETURNING event_id
            """
            result = self.db_handler.execute_query(update_sql, {
                "location": full_address_with_building,
                "address_id": address_id,
                "building_name": building_name
            })
            updated_event_ids = [r[0] for r in result] if result else []
            for event_id in updated_event_ids:
                update_log.append({"event_id": event_id, "building_name": building_name, "full_address": full_address_with_building, "address_id": address_id})
            updated_count += len(updated_event_ids)

        log_df = pd.DataFrame(update_log)
        os.makedirs("output", exist_ok=True)
        log_path = os.path.join("output", "update_null_address_ids.csv")
        log_df.to_csv(log_path, index=False)
        logging.info(f"fix_null_addresses_in_events(): Updated {updated_count} event(s). Log saved to {log_path}.")


    def apply_deduplication_response(self, response, preview_mode=False):
        """
        Apply deduplication results to the database.
        Delete duplicate addresses and update events table to point to the canonical address.
        
        Args:
            response: LLM response with deduplication decisions
            preview_mode: If True, save to CSV instead of deleting from database
        """
        # Parse JSON string using robust LLM parsing methods
        if isinstance(response, str):
            response = response.strip()
            if not response or "no duplicates" in response.lower():
                logging.info("apply_deduplication_response(): LLM response indicates no duplicates.")
                return
            # Use the robust parsing methods from LLMHandler
            parsed_response = self.llm_handler.extract_and_parse_json(response, "address_dedup")
            if not parsed_response:
                logging.warning("apply_deduplication_response(): Failed to parse LLM response using robust parsing")
                return
            response = parsed_response

        if not isinstance(response, list):
            logging.warning("apply_deduplication_response(): Skipping due to malformed LLM response.")
            return

        for group in response:
            canonical_id = group.get("canonical_address_id")
            duplicates = group.get("duplicates", [])

            if not canonical_id or not duplicates:
                continue

            # Use db_handler instead of db
            full_address_result = self.db_handler.execute_query(
                "SELECT full_address FROM address WHERE address_id = :id",
                {"id": canonical_id}
            )
            if not full_address_result:
                logging.warning(f"apply_deduplication_response(): No full_address found for ID {canonical_id}")
                continue
            canonical_address = full_address_result[0][0]

            if preview_mode:
                # Save to CSV for review instead of deleting
                preview_data = []
                
                # Get column names using pandas
                import pandas as pd
                addr_columns_df = pd.read_sql("SELECT * FROM address LIMIT 1", self.conn)
                column_names = addr_columns_df.columns.tolist()
                
                # Add canonical address
                canonical_addr = self.db_handler.execute_query(
                    "SELECT * FROM address WHERE address_id = :id", {"id": canonical_id}
                )
                if canonical_addr:
                    addr_data = dict(zip(column_names, canonical_addr[0]))
                    addr_data['status'] = 'CANONICAL'
                    addr_data['group_id'] = f"semantic_{canonical_id}"
                    addr_data['reason'] = 'semantic_clustering'
                    preview_data.append(addr_data)
                
                # Add duplicate addresses
                for dup_id in duplicates:
                    dup_addr = self.db_handler.execute_query(
                        "SELECT * FROM address WHERE address_id = :id", {"id": dup_id}
                    )
                    if dup_addr:
                        addr_data = dict(zip(column_names, dup_addr[0]))
                        addr_data['status'] = 'PROPOSED_DUPLICATE'
                        addr_data['group_id'] = f"semantic_{canonical_id}"
                        addr_data['reason'] = 'semantic_clustering'
                        preview_data.append(addr_data)
                
                # Append to CSV file
                import os
                preview_df = pd.DataFrame(preview_data)
                csv_file = "output/address_duplicates.csv"
                if os.path.exists(csv_file):
                    preview_df.to_csv(csv_file, mode='a', header=False, index=False)
                else:
                    preview_df.to_csv(csv_file, mode='w', header=True, index=False)
                    
                logging.info(f"apply_deduplication_response(): Added {len(duplicates)} proposed duplicates to {csv_file} for review")
            else:
                # Original deletion logic
                total_events_updated = 0
                for dup_id in duplicates:
                    logging.info(f"apply_deduplication_response(): Repointing events from {dup_id} → {canonical_id}")

                    num_events_updated = self.db_handler.execute_query(
                        """
                        UPDATE events 
                        SET address_id = :canonical, location = :full_address 
                        WHERE address_id = :duplicate
                        """,
                        {
                            "canonical": canonical_id,
                            "duplicate": dup_id,
                            "full_address": canonical_address
                        }
                    )
                    total_events_updated += num_events_updated or 0
                    logging.info(f"apply_deduplication_response(): Updated {num_events_updated} events from {dup_id} to {canonical_id}")

                    logging.info(f"apply_deduplication_response(): Deleting address_id {dup_id} from address table")
                    self.db_handler.execute_query("DELETE FROM address WHERE address_id = :id", {"id": dup_id})

                logging.info(
                    f"apply_deduplication_response(): Cluster update summary → canonical_id {canonical_id}, "
                    f"total events updated: {total_events_updated}, duplicates removed: {len(duplicates)}"
                )


    def generate_prompt(self, subcluster: List[Dict]) -> str:
        """
        Generate a complete LLM prompt from the semantic deduplication prompt template
        and the given subcluster of address records.
        """
        template_path = self.config["prompts"]["fix_dup_addresses_semantic_clustering"]
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        def safe_json(record):
            clean = {}
            for k, v in record.items():
                if pd.isna(v):
                    clean[k] = ""
                elif isinstance(v, pd.Timestamp):
                    clean[k] = v.isoformat()
                else:
                    clean[k] = v
            return json.dumps(clean, ensure_ascii=False)

        data_lines = [safe_json(record) for record in subcluster]
        return f"{template}\n\n" + "\n".join(data_lines)

    
    def log_llm_response(self, log_file_path: str, subcluster: List[Dict], prompt: str, response: Any):
        """
        Logs the prompt and response from the LLM to the clustering log file.
        """
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write("--- PROMPT ---\n")
            f.write(prompt + "\n")
            f.write("--- RESPONSE ---\n")
            if isinstance(response, str):
                f.write(response.strip() + "\n")
            else:
                f.write(json.dumps(response, indent=2, ensure_ascii=False) + "\n")


    def log_cluster_header(self, log_file_path, cluster_index, total_clusters, address_records):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== Cluster {cluster_index + 1}/{total_clusters} ===\n")
            f.write("--- RECORDS ---\n")
            for record in address_records:
                f.write(json.dumps(record, default=str) + "\n")


    def subcluster_by_building_name(self, address_records: List[Dict], distance_threshold: int = 30) -> List[List[Dict]]:
        """
        Performs agglomerative clustering on building_name field using fuzzy matching.
        Records without a usable building_name are excluded from clustering and added as separate subclusters.
        """
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        from rapidfuzz import fuzz

        records_with_names = [r for r in address_records if r.get("building_name", "").strip()]
        records_without_names = [r for r in address_records if not r.get("building_name", "").strip()]

        names = [r["building_name"] for r in records_with_names]
        n = len(names)

        if n <= 1:
            return [records_with_names + records_without_names] if records_with_names else [address_records]

        dist_matrix = np.zeros((n, n))
        for x in range(n):
            for y in range(x + 1, n):
                score = fuzz.ratio(names[x], names[y])
                dist = 100 - score
                dist_matrix[x, y] = dist
                dist_matrix[y, x] = dist
        np.fill_diagonal(dist_matrix, 0)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average"
        )
        labels = clustering.fit_predict(dist_matrix)

        clusters = []
        for label in set(labels):
            group = [records_with_names[i] for i in range(n) if labels[i] == label]
            clusters.append(group)

        clusters.extend([[r] for r in records_without_names])
        return clusters
    

    def identical_building_name_with_missing_street_fields(self, records, log_file_path):
        def norm(val):
            return str(val or '').strip().lower()

        if len(records) <= 1:
            return False

        bnames = [norm(r.get("building_name")) for r in records]
        if len(set(bnames)) != 1:
            return False

        filled = [r for r in records if r.get("street_name") or r.get("street_number") or r.get("postal_code")]
        if not filled:
            return False

        canonical = filled[0]
        canonical_id = canonical["address_id"]
        duplicate_ids = [r["address_id"] for r in records if r["address_id"] != canonical_id]

        self.log_resolution(log_file_path, canonical_id, duplicate_ids, "identical building name, missing fields")
        return True
    

    def should_skip_cluster_by_rules(self, subcluster, log_file_path):
        from collections import Counter

        def normalize(val):
            return unicodedata.normalize("NFKC", str(val).strip().lower().replace("null", ""))

        streets = [normalize(r.get("street_number")) for r in subcluster]
        street_names = [normalize(r.get("street_name")) for r in subcluster]
        cities = [normalize(r.get("city")) for r in subcluster]

        all_unique = len(set(streets)) == len(streets)
        with open(log_file_path, "a") as f:
            if all_unique:
                f.write("--- STATUS ---\nSKIPPED: All street numbers are unique. No duplicates.\n")
                return True

            street_counts = Counter(streets)
            shared_streets = [k for k, v in street_counts.items() if v > 1 and k]
            for street in shared_streets:
                matching_names = {street_names[i] for i in range(len(street_names)) if streets[i] == street}
                matching_cities = {cities[i] for i in range(len(cities)) if streets[i] == street and cities[i]}
                if len(matching_names) > 1:
                    f.write(f"--- STATUS ---\nSKIPPED: Matching street number '{street}' has different street names.\n")
                    return True
                if len(matching_cities) > 1:
                    f.write(f"--- STATUS ---\nSKIPPED: Matching street number '{street}' has conflicting cities.\n")
                    return True
        return False
    

    def try_resolve_canonical_locally(self, subcluster, log_file_path):
        def score(r):
            return (
                int(bool(r.get("postal_code"))) +
                int(bool(r.get("street_name"))) +
                int(bool(r.get("time_stamp") and str(r["time_stamp"]).lower() != "nat"))
            )

        sorted_records = sorted(subcluster, key=lambda r: (-score(r), r.get("time_stamp", "")), reverse=True)
        best = sorted_records[0]
        others = [r for r in subcluster if r["address_id"] != best["address_id"]]

        reason = []
        if best.get("postal_code"): reason.append("has postal_code")
        if best.get("street_name"): reason.append("has street_name")
        if best.get("time_stamp") and str(best["time_stamp"]).lower() != "nat": reason.append("most recent time_stamp")

        if len(set(r.get("building_name", "") for r in subcluster)) > 1:
            msg = "SKIPPED: Multiple different building names detected. No duplicates."
            with open(log_file_path, "a") as f:
                f.write(f"--- STATUS ---\n{msg}\n")
            return True

        with open(log_file_path, "a") as f:
            f.write(f"--- STATUS ---SELECTED CANONICAL WITHOUT LLM: {best['address_id']}. Reason: {', '.join(reason)}\n")

        return True if reason else False

    def building_names_fuzzy_match(self, records: List[Dict], log_file_path: str) -> bool:
        """
        Checks if there's a fuzzy match between non-empty building names.
        Only returns True if fuzzy match ratio > 70 and fewer than 5 valid names exist.
        """
        from rapidfuzz import fuzz

        valid = [r for r in records if r.get("building_name", "").strip().lower() not in {"", "null"}]
        if len(valid) < 2:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write("--- STATUS ---\nNo valid building names. Skipping fuzzy match.\n")
            return False

        names = [r["building_name"].strip() for r in valid]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = fuzz.token_sort_ratio(names[i], names[j])
                if score > 70:
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write("--- STATUS ---\n")
                        f.write("Fuzzy match on building names above threshold. Sending to LLM.\n")
                    return True
        return False

    
    def has_conflicting_street_numbers_unless_same_building(self, records):
        def norm(val):
            return str(val).strip().lower() if val else ""

        # Filter out null values (both None and null strings)
        null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}
        street_numbers = set(norm(r.get("street_number")) for r in records 
                           if r.get("street_number") is not None and norm(r.get("street_number")) not in null_strings)
        building_names = [norm(r.get("building_name")) for r in records if norm(r.get("building_name"))]

        if len(street_numbers) <= 1:
            return False

        if len(set(building_names)) == 1:
            return False

        for i in range(len(building_names)):
            for j in range(i+1, len(building_names)):
                if fuzz.token_set_ratio(building_names[i], building_names[j]) > 70:
                    return False

        return True
    

    def submit_to_llm_and_log(self, subcluster, cluster_index, log_file_path):
        df = pd.DataFrame(subcluster)
        prompt_path = "prompts/fix_dup_addresses.txt"
        with open(prompt_path) as f:
            template = f.read()

        columns = [
            "address_id", "full_address", "building_name", "street_number", "street_name",
            "street_type", "direction", "city", "met_area", "province_or_state", "postal_code",
            "country_id", "time_stamp"
        ]
        df = df[columns].fillna("")
        table = tabulate(df.values.tolist(), headers=columns, tablefmt="github")
        prompt = f"""{template}\n\n{table}\n"""

        logging.info(f"Sending cluster {cluster_index + 1} to LLM...")
        response = self.llm_handler.query_llm("address_dedup", prompt)

        with open(log_file_path, "a") as f:
            f.write("--- PROMPT ---\n")
            f.write(f"{prompt_path}\n\n")
            f.write(f"{table}\n")
            f.write("--- RESPONSE ---\n")
            f.write(json.dumps(response, indent=2))
            f.write("\n")

        return response

    def cluster_addresses_semantically(self, df: pd.DataFrame, eps: float = 0.1, min_samples: int = 2) -> List[List[int]]:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = df['full_address'].astype(str).tolist()
        address_ids = df['address_id'].tolist()
        embeddings = model.encode(texts, show_progress_bar=False)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
        labels = clustering.labels_

        clusters = {}
        for label, addr_id in zip(labels, address_ids):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(addr_id)

        return list(clusters.values())

    def log_resolution(self, log_file_path, canonical_id, duplicates, reason):
        with open(log_file_path, "a") as f:
            f.write(f"--- STATUS ---\n")
            f.write(f"Canonical: {canonical_id}, Duplicates: {duplicates}. Reason: {reason}\n")

    def fetch_possible_duplicate_addresses(self):
        """
        Identify and retrieve potential duplicate addresses from the database based on similar text.
        
        Uses similarity on full_address, building_name, and location components to find potential duplicates.
        Groups by similar street names and postal codes to identify clusters of potential duplicates.
        
        Returns:
            pd.DataFrame: A DataFrame containing possible duplicate addresses with group_id.
        """
        # Simple approach without SIMILARITY function (requires pg_trgm extension)
        sql = """
        WITH PotentialDuplicates AS (
            SELECT a1.address_id as id1, a2.address_id as id2,
                   a1.full_address, a1.building_name, a1.street_name, a1.city, a1.postal_code
            FROM address a1
            JOIN address a2 ON a1.address_id < a2.address_id
            WHERE (
                -- Same street number and street name (exact match for duplicates)
                (LOWER(a1.street_number) = LOWER(a2.street_number) AND LOWER(a1.street_name) = LOWER(a2.street_name))
                -- Same building name (if both not empty)
                OR (COALESCE(a1.building_name, '') != '' AND COALESCE(a2.building_name, '') != '' 
                    AND LOWER(a1.building_name) = LOWER(a2.building_name))
            )
        ),
        AddressGroups AS (
            SELECT id1 as address_id FROM PotentialDuplicates
            UNION
            SELECT id2 as address_id FROM PotentialDuplicates
        )
        SELECT DENSE_RANK() OVER (ORDER BY a.city, a.street_name, a.postal_code) as group_id,
               a.address_id, a.full_address, a.building_name, a.street_number, 
               a.street_name, a.city, a.postal_code, a.country_id
        FROM address a
        JOIN AddressGroups ag ON a.address_id = ag.address_id
        ORDER BY group_id, a.address_id;
        """
        
        try:
            df = pd.read_sql(sql, self.conn)
            logging.info(f"fetch_possible_duplicate_addresses(): Found {len(df)} potential duplicate addresses")
            return df
        except Exception as e:
            logging.warning(f"fetch_possible_duplicate_addresses(): SIMILARITY function not available, using simpler approach: {e}")
            
            # Fallback to simpler grouping if SIMILARITY function not available
            simple_sql = """
            SELECT ROW_NUMBER() OVER (ORDER BY street_name, city, postal_code) as group_id,
                   address_id, full_address, building_name, street_number, 
                   street_name, city, postal_code, country_id
            FROM address 
            WHERE (street_name, city, postal_code) IN (
                SELECT street_name, city, postal_code 
                FROM address 
                GROUP BY street_name, city, postal_code 
                HAVING COUNT(*) > 1
            )
            ORDER BY street_name, city, postal_code, address_id;
            """
            df = pd.read_sql(simple_sql, self.conn)
            logging.info(f"fetch_possible_duplicate_addresses(): Found {len(df)} potential duplicate addresses (simple method)")
            return df

    def process_address_duplicates_with_llm(self):
        """
        Process duplicate addresses using LLM similar to event deduplication.
        Returns the number of addresses deleted.
        """
        df = self.fetch_possible_duplicate_addresses()
        if df.empty:
            logging.info("process_address_duplicates_with_llm(): No potential duplicates found.")
            return 0
            
        response_dfs = []
        
        for group_id in df['group_id'].unique():
            group = df[df['group_id'] == group_id]
            if len(group) <= 1:
                continue
                
            # Create prompt for address group
            address_text = "\n".join([
                f"Address ID {row['address_id']}: {row['full_address']} | Building: {row['building_name']} | Street: {row['street_number']} {row['street_name']}"
                for _, row in group.iterrows()
            ])
            
            # Load prompt from config
            prompt_path = self.config['prompts']['dedup_llm_address']
            with open(prompt_path, 'r') as f:
                prompt_template = f.read()
            prompt = f"{prompt_template}\n\n{address_text}"

            try:
                response = self.llm_handler.query_llm("address_dedup", prompt)
                if response:
                    # Parse LLM response using robust parsing methods
                    parsed = self.llm_handler.extract_and_parse_json(response, "address_dedup")
                    if parsed and isinstance(parsed, list):
                        response_df = pd.DataFrame(parsed)
                        response_df['address_id'] = response_df['address_id'].astype(int)
                        response_dfs.append(response_df)
                    else:
                        logging.warning(f"Failed to parse LLM response for group {group_id}")
            except Exception as e:
                logging.error(f"Error processing group {group_id}: {e}")
                
        if response_dfs:
            return self.merge_and_save_address_results(df, response_dfs)
        else:
            logging.warning("process_address_duplicates_with_llm(): No valid responses from LLM.")
            return 0
            
    def merge_and_save_address_results(self, df, response_dfs):
        """
        Merge address results and save to CSV with review capability.
        """
        response_df = pd.concat(response_dfs, ignore_index=True)
        df_merged = df.merge(response_df, on="address_id", how="left")
        
        # Save all results for review
        df_merged.to_csv("output/address_dedup_results.csv", index=False)
        
        # Create review CSV with canonical and duplicate addresses grouped
        review_df = df_merged[df_merged['Label'].notna()].copy()
        # Convert Label to int to handle string parsing from LLM
        review_df['Label'] = review_df['Label'].astype(int)
        review_df['status'] = review_df['Label'].map({0: 'CANONICAL', 1: 'PROPOSED_DUPLICATE'})
        review_df = review_df[['address_id', 'full_address', 'building_name', 'street_number', 'street_name', 
                              'city', 'postal_code', 'group_id', 'status']].copy()
        review_df['reason'] = 'llm_analysis'
        review_df = review_df.sort_values(['group_id', 'status'])
        # Append to existing file if it exists (from semantic clustering)
        csv_file = "output/address_duplicates.csv"
        if os.path.exists(csv_file):
            review_df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            review_df.to_csv(csv_file, mode='w', header=True, index=False)
        
        # Delete the duplicate addresses
        duplicate_ids = review_df[review_df['status'] == 'PROPOSED_DUPLICATE']['address_id'].tolist()
        logging.info(f"Found {len(duplicate_ids)} addresses marked for deletion: {duplicate_ids}")
        if duplicate_ids:
            logging.info(f"Deleting {len(duplicate_ids)} duplicate addresses from database...")
            deleted_count = 0
            for addr_id in duplicate_ids:
                # First update events to point to canonical address
                canonical_group = review_df[review_df['group_id'] == review_df[review_df['address_id'] == addr_id]['group_id'].iloc[0]]
                canonical_id = int(canonical_group[canonical_group['status'] == 'CANONICAL']['address_id'].iloc[0])
                canonical_address = canonical_group[canonical_group['status'] == 'CANONICAL']['full_address'].iloc[0]
                
                # Update events
                self.db_handler.execute_query(
                    "UPDATE events SET address_id = :canonical, location = :address WHERE address_id = :duplicate",
                    {"canonical": canonical_id, "address": canonical_address, "duplicate": addr_id}
                )
                
                # Delete duplicate address
                self.db_handler.execute_query("DELETE FROM address WHERE address_id = :id", {"id": addr_id})
                deleted_count += 1
                
            logging.info(f"Successfully deleted {deleted_count} duplicate addresses")
            return deleted_count
        else:
            logging.info("No duplicate addresses to delete")
            return 0

    def deduplicate_addresses_with_llm_semantic_clustering(self):
        """
        Deduplicates address records using an LLM by clustering semantically similar addresses
        using SentenceTransformer and sending each subcluster (if < 5 rows) to the LLM. Logs results only.
        """
        logging.info("Starting LLM-based address deduplication with semantic clustering...")
        os.makedirs("output", exist_ok=True)
        log_file_path = os.path.join("output", "llm_fix_semantic_clustering.txt")

        df = pd.read_sql("SELECT * FROM address", self.conn).fillna("")
        if len(df) <= 1:
            logging.info("Only one address row present. Nothing to deduplicate.")
            return

        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df.loc[:, col] = df[col].astype(str)

        clusters = self.cluster_addresses_semantically(df, eps=0.2)

        for i, group_ids in enumerate(clusters):
            batch = df[df['address_id'].isin(group_ids)].copy()
            if batch.empty:
                continue
            address_records = batch.to_dict(orient="records")
            self.log_cluster_header(log_file_path, i, len(clusters), address_records)
            subclusters = self.subcluster_by_building_name(address_records)

            final_subclusters = []
            for subcluster in subclusters:
                if len(subcluster) >= 5:
                    refined = self.subcluster_by_building_name(subcluster, distance_threshold=15)
                    final_subclusters.extend(refined)
                else:
                    final_subclusters.append(subcluster)

            for subcluster in final_subclusters:
                if len(subcluster) <= 1 or len(subcluster) >= 5:
                    continue

                # STEP 1: If building names match fuzzily → send to LLM
                if self.building_names_fuzzy_match(subcluster, log_file_path):
                    logging.info(f"Sending cluster {i + 1} to LLM due to building name match...")
                    prompt = self.generate_prompt(subcluster)
                    response = self.llm_handler.query_llm("address_dedup", prompt)
                    self.log_llm_response(log_file_path, subcluster, prompt, response)
                    self.apply_deduplication_response(response, preview_mode=False)
                    continue

                # STEP 2: Only if no building match, consider skipping by street rules
                if self.should_skip_cluster_by_rules(subcluster, log_file_path):
                    continue

                # STEP 3: Fallback: try resolving locally
                if self.try_resolve_canonical_locally(subcluster, log_file_path):
                    continue

                # STEP 4: Final fallback: LLM if conflicting street numbers
                if self.has_conflicting_street_numbers_unless_same_building(subcluster):
                    logging.info(f"Sending cluster {i + 1} to LLM due to conflicting street numbers...")
                    prompt = self.generate_prompt(subcluster)
                    response = self.llm_handler.query_llm("address_dedup", prompt)
                    self.log_llm_response(log_file_path, subcluster, prompt, response)
                    self.apply_deduplication_response(response, preview_mode=False)
                    continue

        logging.info("Finished LLM-based address deduplication and database updates.")

# ------------------------------------------------------------------------
# Main entry point - run asynchronously
# ------------------------------------------------------------------------

async def main():
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Set up logging
    # Ensure the 'logs' directory exists
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/clean_up_log.txt" ,
        filemode='a',  # Changed to append mode to preserve logs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logging.info("clean_up.py starting...")

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Initialize the CleanUp instance
    clean_up_instance = CleanUp(config)

    # Initialize the database handler
    db_handler = DatabaseHandler(config)
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    # Fix duplicate rows in the address table
    
    # First, clean up any existing 'null' strings in the database
    logging.info("Cleaning up null strings in address table...")
    clean_up_instance.db_handler.clean_null_strings_in_address()
    
    # Standardize postal code formats
    logging.info("Standardizing postal code formats...")
    clean_up_instance.db_handler.standardize_postal_codes()
    
    # Method 1: Semantic clustering first (free, local compute)
    logging.info("Starting semantic clustering address deduplication (free)...")
    clean_up_instance.deduplicate_addresses_with_llm_semantic_clustering()
    
    # Method 2: LLM-based deduplication for remaining candidates (costs money)
    logging.info("Starting LLM-based address deduplication for remaining candidates...")
    clean_up_instance.process_address_duplicates_with_llm()

    # Fix no urls in events
    await clean_up_instance.process_events_without_url()

    # Fix incorrect dance_styles
    await clean_up_instance.fix_incorrect_dance_styles()

    # Delete events outside of BC, Canada
    await clean_up_instance.delete_events_outside_bc()

    # Delete events more than 9 months in the future
    await clean_up_instance.delete_events_more_than_9_months_future()

    # Fix null addresses in events
    await clean_up_instance.fix_null_addresses_in_events()

    # Delete events that you know are not relevant
    bad_urls = [url.strip() for url in config['constants']['delete_known_bad_urls'].split(',') if url.strip()]
    logging.info(f"known_incorrect(): Deleting events with URLs containing: {bad_urls}")
    clean_up_instance.known_incorrect(bad_urls)

    db_handler.count_events_urls_end(start_df, file_name)
    logging.info(f"Wrote events and urls statistics to: {file_name}")

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")


if __name__ == "__main__":
    # Standard approach for async Python scripts
    asyncio.run(main())
