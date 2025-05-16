"""
fb.py

This module defines the FacebookEventScraper class for scraping Facebook event data
using Playwright and BeautifulSoup. It handles logging into Facebook, extracting event
link and text, processing URLs, and 
interacting with a database and Language Learning Model (LLM) to process and store
event data.

Classes:
    FacebookEventScraper:
        - Initializes with configuration, sets up Playwright for browser automation.
        - Logs into Facebook and maintains a session for scraping.
        - Extracts event link and content from Facebook event pages.
        - Processes Facebook URLs including fixing malformed URLs.
        - Interacts with LLMHandler for natural language processing tasks.
        - Interacts with DatabaseHandler to read/write URLs and events to the database.
        - Provides multiple driver methods for different scraping workflows:
            • driver_fb_urls: Processes Facebook URLs from the database.
            • driver_fb_search: Searches for Facebook events based on keywords.
            • driver_no_urls: Handles events without URLs, attempts to find and update them.

Usage Example:
    if __name__ == "__main__":
        # Load configuration and configure logging
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        logging.basicConfig(
            filename=config['logging']['log_file'],
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Initialize dependencies
        db_handler = DatabaseHandler(config)
        fb_scraper = FacebookEventScraper(config_path='config/config.yaml')
        llm_handler = LLMHandler(config_path='config/config.yaml')

        # Run scraping drivers
        fb_scraper.driver_fb_urls()
        fb_scraper.driver_fb_search()
        fb_scraper.driver_no_urls()

        # Clean up resource
        fb_scraper.browser.close()
        fb_scraper.playwright.stop()
        
        # Logging of process duration handled in __main__ block

Dependencies:
    - Playwright: For browser automation to navigate and scrape Facebook.
    - BeautifulSoup: To parse HTML content of event pages.
    - pandas: For data manipulation and CSV operations.
    - fuzzywuzzy: For fuzzy string matching.
    - requests: For HTTP requests when scraping non-JS rendered pages.
    - yaml: For configuration file parsing.
    - logging: For tracking the execution flow and errors.
    - re: For regular expression operations.
    - Other custom modules: llm (LLMHandler), db (DatabaseHandler).

Note:
    - The module assumes valid configuration in 'config/config.yaml'.
    - Logging is configured in the main section to record key actions and errors.
    - The class methods heavily rely on external services (Facebook, database, LLM),
      and their correct functioning depends on valid credentials and network access.
"""


from bs4 import BeautifulSoup
from datetime import datetime
from fuzzywuzzy import fuzz
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import random
import re
from sqlalchemy import text
import time
import yaml

# Import other classes
from credentials import get_credentials
from db import DatabaseHandler
from llm import LLMHandler

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
logging.info("\n\nfb.py starting...")

# Instantiate the class libraries
db_handler = DatabaseHandler(config)
llm_handler = LLMHandler(config_path='config/config.yaml')


class FacebookEventScraper():
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Start Playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.config['crawling']['headless'])

        # Create a single context & page, reusing 'facebook_auth.json'
        self.context = self.browser.new_context(storage_state="facebook_auth.json")
        self.page = self.context.new_page()
        # keep a stable reference for re‑use
        self.logged_in_page = self.page

        # Attempt login
        if self.login_to_facebook():
            logging.info("Facebook login successful.")
        else:
            logging.error("Facebook login failed.")

        # Run statistics tracking
        if config['testing']['status']:
            self.run_name = f"Test Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.run_description = "Test Run Description"
        else:
            self.run_name = "Facebook Event Scraper Run"
            self.run_description = f"Production {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.start_time = datetime.now()

        # **Tracking Variables**
        self.unique_urls = set()  # Store unique URLs
        self.unique_urls_count = 0  # Unique URLs processed
        self.total_url_attempts = 0  # Total number of URL contact attempts
        self.urls_with_extracted_text = 0  # URLs where text was successfully extracted
        self.urls_with_found_keywords = 0  # URLs where keywords were found
        self.events_written_to_db = 0  # Number of events successfully written to DB

        # Initialize run statistics
        self.start_time = datetime.now()
        self.total_url_attempts = 0
        self.urls_with_extracted_text = 0
        self.urls_with_found_keywords = 0
        self.events_written_to_db = 0

        # URL tracking
        self.urls_visited = set()

        # Get keywords
        self.keywords_list = llm_handler.get_keywords()
    

    def login_to_facebook(self):
        """
        Log into Facebook, reusing an existing session if possible.
        Detects and pauses for manual solve of any checkbox CAPTCHA.
        Saves the session to 'facebook_auth.json'.
        """
        storage = "facebook_auth.json"

        # 1) Try to reload an existing session
        try:
            ctx = self.browser.new_context(storage_state=storage)
            page = ctx.new_page()
            page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=30000)
            if "login" not in page.url.lower():
                logging.info("login_to_facebook: reused existing session, already logged in.")
                # update both page refs
                self.page = page
                self.logged_in_page = page
                return True
        except Exception:
            logging.info("login_to_facebook: no saved session, performing fresh login.")

        # 2) Fresh login
        self.page.goto("https://www.facebook.com/login", wait_until="domcontentloaded", timeout=30000)
        email, password, _ = get_credentials("Facebook")
        self.page.fill("input[name='email']", email)
        self.page.fill("input[name='pass']", password)
        self.page.click("button[name='login']")
        logging.info("login_to_facebook: credentials submitted, waiting…")

        # 3) Detect reCAPTCHA iframe
        try:
            self.page.wait_for_selector("iframe[src*='recaptcha']", timeout=10000)
            self.page.screenshot(path="debug/facebook_recaptcha.png", full_page=True)
            logging.info("login_to_facebook: reCAPTCHA detected—screenshot saved.")
            input("Please solve the reCAPTCHA in the browser, then press ENTER here…")
        except PlaywrightTimeoutError:
            logging.info("login_to_facebook: no reCAPTCHA detected.")

        # 4) Wait for the page to settle
        try:
            self.page.wait_for_load_state("networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            logging.warning("login_to_facebook: networkidle timeout—continuing anyway.")

        # 5) Verify login succeeded
        if "login" in self.page.url.lower():
            logging.error("login_to_facebook: still on login page—login probably failed.")
            return False

        # 6) Save session
        try:
            self.page.context.storage_state(path=storage)
            logging.info("login_to_facebook: session saved to facebook_auth.json.")
        except Exception as e:
            logging.warning(f"login_to_facebook: could not save session: {e}")

        # make sure future scrapes reuse this page
        self.logged_in_page = self.page
        return True
    

    def extract_event_links(self, search_url):
        """
        Extracts event links from a Facebook page (search or group) by regex.
        If it’s a group URL, auto-navigate to the /events/ sub-page.
        Always reuses the single logged-in page.
        """
        # 0) Ensure login
        if not self.login_to_facebook():
            logging.error(f"extract_event_links(): login failed, skipping {search_url}")
            return set()

        # 0b) If this is a group URL, go straight to its events tab
        if "/groups/" in search_url and not search_url.rstrip("/").endswith("/events"):
            search_url = search_url.rstrip("/") + "/events/"

        try:
            self.total_url_attempts += 1
            page = self.logged_in_page

            timeout_value = random.randint(8000, 12000)
            logging.info(f"extract_event_links(): Navigating to {search_url} (timeout {timeout_value}ms)")
            page.goto(search_url, timeout=timeout_value, wait_until="domcontentloaded")
            page.wait_for_timeout(random.randint(4000, 7000))

            # Scroll to load all events
            for _ in range(self.config['crawling']['scroll_depth']):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.randint(1500, 3000))

            content = page.content()
            links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))

            if links:
                self.urls_with_extracted_text += 1
                self.unique_urls.update(links)

            logging.info(f"extract_event_links(): Found {len(links)} event links on {search_url}")
            return links

        except Exception as e:
            logging.error(f"extract_event_links(): Error on {search_url}: {e}")
            return set()
    

    def extract_event_text(self, link):
        """
        Extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            link (str): The event link.

        Returns:
            str: The extracted relevant text content, or None if no relevant text is found.
        """
        self.total_url_attempts += 1  # Count URL contact attempts

        # 1) Reuse the logged‑in page if valid, otherwise open a fresh one
        try:
            if not self.logged_in_page or self.logged_in_page.url == "about:blank":
                raise Exception("Logged-in page invalid")
            page = self.logged_in_page
        except Exception:
            logging.warning("extract_event_text(): Logged-in page invalid, opening a new one.")
            page = self.context.new_page()
            self.logged_in_page = page

        # 2) Navigate only to DOMContentLoaded, then wait a bit for JS
        timeout_value = random.randint(10000, 15000)
        logging.info(f"extract_event_text(): Navigating to {link} (timeout {timeout_value} ms)...")
        try:
            page.goto(link, wait_until="domcontentloaded", timeout=timeout_value)
        except Exception as e:
            logging.warning(f"extract_event_text(): Navigation failed for {link}: {e}")
            return None

        # Give React/JS a few seconds to hydrate
        page.wait_for_timeout(random.randint(4000, 7000))

        # 3) Click any “See more” buttons
        for btn in page.query_selector_all("text=/See more/i"):
            try:
                btn.click()
                page.wait_for_timeout(random.randint(3000, 6000))
                logging.info(f"extract_event_text(): Clicked 'See more' on {link}")
            except Exception as e:
                logging.warning(f"extract_event_text(): Could not click 'See more' on {link}: {e}")

        # Final short pause before scraping HTML
        page.wait_for_timeout(random.randint(3000, 5000))

        # 4) Pull content and parse
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        extracted_text = ' '.join(soup.stripped_strings)

        if not extracted_text:
            logging.warning(f"extract_event_text(): No text extracted from {link}.")
            return None

        # 5) Update counters and check for keywords
        self.urls_with_extracted_text += 1
        logging.info(f"extract_event_text(): Raw text length {len(extracted_text)} chars for {link}")

        found = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
        if not found:
            logging.info(f"extract_event_text(): No keywords found in text for {link}.")
            return None

        self.urls_with_found_keywords += 1

        # 6) For FB event pages, run your regex‑based slice
        if 'facebook.com/events/' in link:
            chunk = self.extract_relevant_text(extracted_text, link)
            if chunk:
                logging.info(f"extract_event_text(): Extracted relevant snippet ({len(chunk)} chars) from {link}")
                logging.info(f"extract_event_text(): Snippet: {chunk}")
                return chunk
            else:
                logging.warning(f"extract_event_text(): Snippet extraction failed for {link}, returning full text.")
                logging.info(f"extract_event_text(): Full text: {extracted_text}")
                return extracted_text

        # 7) Otherwise return everything
        return extracted_text

    
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
        logging.info(f"def extract_relevant_text(): Extracting relevant text from {link}.")

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
            logging.warning(f"def extract_relevant_text(): 'Guests See All' not found after last day of the week in {link}.")
            return None

        gsa_end = gsa_match.end()

        # Extract the desired text
        extracted_text = content[day_start:gsa_end]

        return extracted_text
    
        
    def scrape_events(self, keywords):
        """
        Logs into Facebook once, performs searches for keywords, and extracts event link and text.

        Args:
            keywords (list): List of keywords to search for.

        Returns:
            tuple: The last search_url used and a list of tuples containing event link and extracted text.
        """
        base_url = self.config['constants']['fb_base_url']
        location = self.config['constants']['location']
        extracted_text_list = []

        for keyword in keywords:
            search_url = f"{base_url} {location} {keyword}"
            event_links = self.extract_event_links(search_url)
            logging.info(f"def scrape_events: Used {search_url} to get {len(event_links)} event_links\n")

            self.total_url_attempts += len(event_links)  # Update total URL attempts

            for link in event_links:
                if link in self.urls_visited:
                    continue  # Skip already visited URLs
                else:
                    self.unique_urls_count += 1  # Increment unique URL count

                    if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                        logging.info("def scrape_events(): Reached the URL visit limit. Stopping the scraping process.")
                        return search_url, extracted_text_list

                    extracted_text = self.extract_event_text(link)
                    self.urls_visited.add(link)

                    if extracted_text:
                        self.urls_with_extracted_text += 1  # Increment URLs with extracted text

                        # Check for keywords in the extracted text
                        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
                        if found_keywords:
                            self.urls_with_found_keywords += 1  # Increment URLs with found keywords
                            extracted_text_list.append((link, extracted_text))
                            logging.debug(f"Visited URL: {link}. Total visited: {len(self.urls_visited)}")
                        else:
                            logging.info(f"def scrape_events(): No keywords found in extracted text for URL: {link}.")
                    else:
                        logging.info(f"def scrape_events(): No text extracted for URL: {link}.")

        logging.info(f"def scrape_events(): Extracted text from {len(extracted_text_list)} events.")

        # Checkpoint history. Write extracted_text_list to a csv file
        extracted_text_df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])
        output_path = self.config['checkpoint']['extracted_text']
        if os.path.exists(output_path):
            extracted_text_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            extracted_text_df.to_csv(output_path, index=False)
        logging.info(f"def scrape_events(): Extracted text data written to {output_path}.")

        return search_url, extracted_text_list
        

    def fix_facebook_event_url(self, malformed_url):
        """
        Extracts and fixes a malformed Facebook event URL.
        
        Args:
            malformed_url (str): The potentially malformed URL.
        
        Returns:
            str or None: A corrected URL in the format 
                        'https://www.facebook.com/events/<event_id>/' 
                        if found, otherwise None.
        """
        # Regular expression pattern to capture the Facebook event URL structure.
        pattern = re.compile(
            r"(https?://)?(?:www\.)?facebook\.com/events/(\d+)/",
            re.IGNORECASE
        )
        
        match = pattern.search(malformed_url)
        if match:
            event_id = match.group(2)  # Extracted event id
            # Construct a properly formatted URL
            fixed_url = f"https://www.facebook.com/events/{event_id}/"
            return fixed_url
        
        return None


    def process_fb_url(self, url, source, keywords):
        """
        Processes a single Facebook URL: extracts text, interacts with LLM, and updates the database.

        Args:
            url (str): The Facebook URL to process.
            source (str): The organization name associated with the URL.
            keywords (str): Keywords associated with the URL.

        Returns:
            None
        """
        # Sanitize URL if malformed
        if url.startswith("http://https") or url.startswith("https://https"):
            url = self.fix_facebook_event_url(url)

        # Default values
        relevant = False
        increment_crawl_try = 1

        if not url:
            logging.warning("process_fb_url(): No valid URL provided for processing.")
            return

        self.total_url_attempts += 1  # Increment total attempts

        # Extract text (Avoid logging in every time)
        extracted_text = self.extract_event_text(url)
        if not extracted_text:
            logging.info(f"process_fb_url(): No text extracted for URL: {url}. Marking as processed.")
            db_handler.update_url(url, relevant, increment_crawl_try)
            return

        self.urls_with_extracted_text += 1  # Increment extracted text count

        # Check for keywords in extracted text
        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
        if not found_keywords:
            logging.info(f"process_fb_url(): No keywords found in text for {url}. Marking as processed.")
            db_handler.update_url(url, relevant, increment_crawl_try)
            return

        self.urls_with_found_keywords += 1  # Increment keyword match count
        logging.info(f"process_fb_url(): Keywords found in text for {url}.")

        # Generate prompt & query LLM only if keywords were found
        prompt = llm_handler.generate_prompt(url, extracted_text, 'fb')
        llm_response = llm_handler.query_llm(prompt)

        # If LLM says "No events found," update URL as non-relevant
        if llm_response is None or "No events found" in llm_response:
            logging.info(f"process_fb_url(): LLM determined no valid events for {url}. Marking as processed.")
            db_handler.update_url(url, relevant, increment_crawl_try)
            return

        # Extract parsed results from LLM response
        parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
        if not parsed_result:
            logging.warning(f"process_fb_url(): LLM returned an empty response for {url}. Marking as processed.")
            db_handler.update_url(url, relevant, increment_crawl_try)
            return

        # Convert parsed data into a DataFrame & Write to Database
        events_df = pd.DataFrame(parsed_result)
        if events_df.empty:
            logging.warning(f"process_fb_url(): Parsed result was empty for {url}. No data written.")
            db_handler.update_url(url, relevant, increment_crawl_try)
            return

        # Ensure URL field is filled
        if events_df['url'].values[0] == '':
            events_df.loc[0, 'url'] = url

        # Write events to DB
        db_handler.write_events_to_db(events_df, url, source, keywords)
        logging.info(f"process_fb_url(): Events successfully written to DB for {url}.")

        self.events_written_to_db += len(events_df)  # Increment event count

        # Mark URL as relevant
        relevant = True
        db_handler.update_url(url, relevant, increment_crawl_try)
        logging.info(f"process_fb_url(): URL marked as relevant and updated: {url}.")
        return
    

    def driver_fb_search(self):
        """
        1. Reads in the keywords CSV file.
        2. Creates search terms for Facebook searches using each keyword.
        3. Scrapes events from search results.
        4. For each extracted Facebook URL, processes it using process_fb_url.
        5. Writes run statistics to the database.
        """
        # Read in keywords and append a column for processed status
        keywords_df = pd.read_csv(self.config['input']['data_keywords'])
        keywords_df['processed'] = keywords_df['processed'] = False

        for idx, row in keywords_df.iterrows():
            keywords_list = row['keywords'].split(',')
            source = row.get('source', '')

            # Scrape the events
            search_url, extracted_text_list = self.scrape_events(keywords_list)
            logging.info(f"def driver_fb_search(): Extracted text based on search_url: {search_url}.")
            
            self.total_url_attempts += len(extracted_text_list)  # Update total URL attempts

            if extracted_text_list:
                for url, extracted_text in extracted_text_list:
                    if url not in self.urls_visited and 'facebook.com' in url:
                        logging.info(f"def driver_fb_search(): Processing Facebook URL: {url}")

                        if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                            logging.info("def driver_fb_search(): Reached crawl limit. Stopping processing.")
                            break

                        self.urls_with_extracted_text += 1  # Increment extracted text count

                        # Check for keywords in the extracted text
                        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
                        if found_keywords:
                            logging.info(f"def driver_fb_search(): Keywords found in text for {url}.")
                            self.urls_with_found_keywords += 1  # Increment URLs with found keywords

                            # Set prompt and process LLM response
                            prompt = 'fb'
                            llm_response = llm_handler.process_llm_response(url, extracted_text, source, keywords_list, prompt)

                            # If events were successfully extracted and written to the DB
                            if llm_response:
                                self.events_written_to_db += 1
                                logging.info(f"def driver_fb_search(): Events successfully written to DB for {url}.")

                        else:
                            logging.info(f"def driver_fb_search(): No keywords found in extracted text for URL: {url}.")
                    else:
                        logging.info(f"def driver_fb_search(): URL already visited: {url}")

            # Checkpoint the keywords
            keywords_df.loc[idx, 'processed'] = True
            keywords_df.to_csv(self.config['checkpoint']['fb_search'], index=False)
            logging.info(f"def driver_fb_search(): Keywords checkpoint updated.")


    def driver_fb_urls(self):
        """
        1. Gets all of the URLs from the urls table where the link is like '%facebook%'.
        2. For each URL, processes it and then scrapes any event links by hitting the /events/ subpage.
        3. Writes every processed URL—including event links—to the checkpoint CSV.
        """
        # 1) Load or initialize the checkpoint dataframe
        if config['checkpoint']['fb_urls_cp_status']:
            fb_urls_df = pd.read_csv(config['checkpoint']['fb_urls_cp'])
        else:
            query = text("""
                SELECT *
                FROM urls
                WHERE link ILIKE :link_pattern
            """)
            params = {'link_pattern': '%facebook%'}
            fb_urls_df = pd.read_sql(query, db_handler.conn, params=params)
            logging.info(f"def driver_fb_urls(): Retrieved {fb_urls_df.shape[0]} Facebook URLs from the database.")

        # 2) Add checkpoint columns
        fb_urls_df['processed'] = False
        fb_urls_df['events_processed'] = False
        fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)

        # 3) Iterate each base Facebook URL
        if fb_urls_df.shape[0] > 0:
            for idx, row in fb_urls_df.iterrows():
                base_url = row['link']
                source = row['source']
                keywords = row['keywords']
                logging.info(f"def driver_fb_urls(): Processing base URL: {base_url}")

                # Skip if already done
                if base_url in self.urls_visited:
                    continue

                # Process the base URL itself (writes any events found on that exact page)
                self.process_fb_url(base_url, source, keywords)
                self.urls_visited.add(base_url)

                # Mark as processed
                fb_urls_df.loc[fb_urls_df['link'] == base_url, 'processed'] = True
                fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                logging.info(f"def driver_fb_urls(): Base URL marked processed: {base_url}")

                # Honor the run limit
                if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                    break

                # 4) Now scrape *all* event links by auto-navigating to /events/
                fb_event_links = self.extract_event_links(base_url)
                if not fb_event_links:
                    logging.info(f"driver_fb_urls(): No events tab or no events found on {base_url}")

                # 5) Process each event link
                for event_url in fb_event_links:
                    if event_url in self.urls_visited:
                        continue

                    self.process_fb_url(event_url, source, keywords)
                    self.urls_visited.add(event_url)

                    # Add or update checkpoint row
                    if event_url not in fb_urls_df['link'].values:
                        new_row = pd.DataFrame({
                            'link': [event_url],
                            'source': [source],
                            'keywords': [keywords],
                            'processed': [True],
                            'events_processed': [True]
                        })
                        fb_urls_df = pd.concat([fb_urls_df, new_row], ignore_index=True)
                    else:
                        fb_urls_df.loc[fb_urls_df['link'] == event_url, 'processed'] = True
                        fb_urls_df.loc[fb_urls_df['link'] == event_url, 'events_processed'] = True

                    fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                    logging.info(f"def driver_fb_urls(): Event URL marked processed: {event_url}")

                    if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                        break

                # 6) Finally mark that we've scraped events for the base URL
                fb_urls_df.loc[fb_urls_df['link'] == base_url, 'events_processed'] = True
                fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                logging.info(f"def driver_fb_urls(): Events_scraped flag set for base URL: {base_url}")

        else:
            logging.warning("def driver_fb_urls(): No Facebook URLs returned from the database.")


    def write_run_statistics(self):
        """
        Writes run statistics to the database.
        """
        try:
            elapsed_time = str(self.end_time - self.start_time)
            time_stamp = datetime.now()

            run_data = pd.DataFrame([{
                "run_name": self.run_name,
                "run_description": self.run_description,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "elapsed_time": elapsed_time,
                "python_file_name": "fb.py",
                "unique_urls_count": len(self.urls_visited),
                "total_url_attempts": self.total_url_attempts,
                "urls_with_extracted_text": self.urls_with_extracted_text,
                "urls_with_found_keywords": self.urls_with_found_keywords,
                "events_written_to_db": self.events_written_to_db,
                "time_stamp": time_stamp
            }])

            run_data.to_sql("runs", db_handler.get_db_connection(), if_exists="append", index=False)
            logging.info(f"write_run_statistics(): Run statistics written to database for {self.run_name}.")

        except Exception as e:
            logging.error(f"write_run_statistics(): Error writing run statistics: {e}")


    def run(self):
        """
        Runs the Facebook scraper in the specified mode.

        Args:
            mode (str): "search" for keyword-based scraping, "urls" for URL-based processing.
        """
        self.driver_fb_search()
        self.driver_fb_urls()

        self.end_time = datetime.now()

        # Write run statistics to the database
        self.write_run_statistics()


if __name__ == "__main__":
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before fb.py
    start_df = db_handler.count_events_urls_start(file_name)

    # Initialize scraper
    fb_scraper = FacebookEventScraper(config_path='config/config.yaml')

    # Call run()
    fb_scraper.run()

    # Close the browser and Playwright
    fb_scraper.browser.close()
    fb_scraper.playwright.stop()

    # Count events and urls after fb.py
    db_handler.count_events_urls_end(start_df, file_name)

    # End time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Elapsed time
    elapsed_time = end_time - start_time
    logging.info(f"__main__: Elapsed time: {elapsed_time}")