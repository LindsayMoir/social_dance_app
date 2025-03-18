"""
fb.py

This module defines the FacebookEventScraper class for scraping Facebook event data
using Playwright and BeautifulSoup. It handles logging into Facebook, extracting event
link and text, performing Google searches related to events, processing URLs, and 
interacting with a database and Language Learning Model (LLM) to process and store
event data.

Classes:
    FacebookEventScraper:
        - Initializes with configuration, sets up Playwright for browser automation.
        - Logs into Facebook and maintains a session for scraping.
        - Extracts event link and content from Facebook event pages.
        - Uses Google Custom Search API for supplemental searches.
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
    - googleapiclient.discovery: For performing Google Custom Searches.
    - requests: For HTTP requests when scraping non-JS rendered pages.
    - yaml: For configuration file parsing.
    - logging: For tracking the execution flow and errors.
    - re: For regular expression operations.
    - Other custom modules: llm (LLMHandler), db (DatabaseHandler).

Note:
    - The module assumes valid configuration in 'config/config.yaml'.
    - Logging is configured in the main section to record key actions and errors.
    - The class methods heavily rely on external services (Facebook, Google, database, LLM),
      and their correct functioning depends on valid credentials and network access.
"""


from bs4 import BeautifulSoup
from datetime import datetime
from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright
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

        # Create a single context & page
        self.context = self.browser.new_context(storage_state="auth.json")
        self.logged_in_page = self.context.new_page()

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
        Logs into Facebook using stored credentials and avoids opening multiple tabs.
        """
        page = self.logged_in_page  # Reuse the same page

        # Attempt to use saved session state
        try:
            page.goto("https://www.facebook.com/", timeout=60000)

            # If the Facebook search bar is visible, we are already logged in
            if page.is_visible("div[aria-label='Search Facebook']"):
                logging.info("Already logged in.")
                return True
        except Exception as e:
            logging.warning(f"Session state not found. Proceeding with manual login. Error: {e}")

        # Manual login required
        email, password, _ = get_credentials('Facebook')

        page.goto("https://www.facebook.com/", timeout=30000)
        if page.is_visible("input[name='email']") and page.is_visible("input[name='pass']"):
            page.fill("input[name='email']", email)
            page.fill("input[name='pass']", password)
            page.click("button[name='login']")
        else:
            logging.info("Login page not detected, assuming already logged in.")

        # Wait for potential login redirects
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(10000)

        # Handle extra login prompts
        attempts = 0
        while attempts < 3:
            if page.is_visible("input[name='email']") and page.is_visible("input[name='pass']"):
                logging.info("Re-entering credentials due to login prompt.")
                page.fill("input[name='email']", email)
                page.fill("input[name='pass']", password)
                page.click("button[name='login']")
                page.wait_for_timeout(5000)
            else:
                break
            attempts += 1

        # If Captcha appears, wait for user input
        if "login" in page.url.lower():
            logging.warning("Solve the captcha or challenge in the browser, then press Enter.")
            input("Press Enter after solving the captcha...")

        # Save session state after successful login
        self.context.storage_state(path="auth.json")

        logging.info("Login to Facebook successful. Session saved.")
        return True
    

    def extract_event_links(self, search_url):
        """
        Extracts event links from a search page using Playwright and regex.

        Args:
            search_url (str): The Facebook events search URL.

        Returns:
            set: A set of extracted event links.
        """
        try:
            self.total_url_attempts += 1  # Count each attempt

            # Check if the logged-in page is still valid; otherwise, create a new one
            try:
                if not self.logged_in_page or self.logged_in_page.url == "about:blank":
                    raise Exception("Logged-in page is no longer valid.")
                page = self.logged_in_page  # Use the logged-in page if it's valid
            except Exception:
                logging.warning("def extract_event_links(): Logged-in page is closed or invalid, opening a new one.")
                page = self.context.new_page()  # Open a new page in the existing context

            # Randomize timeout to avoid bot detection
            timeout_value = random.randint(8000, 12000)  # 8 to 12 seconds
            logging.info(f"def extract_event_links(): Navigating to {search_url} with timeout {timeout_value} ms.")

            # Try navigating to the page
            page.goto(search_url, timeout=timeout_value)
            page.wait_for_timeout(random.randint(4000, 7000))  # Random wait before interacting

            # Scroll down gradually
            for _ in range(self.config['crawling']['scroll_depth']):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.randint(1500, 3000))  # Random scroll wait

            # Extract page content
            content = page.content()
            links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))

            # Update statistics
            if links:
                self.urls_with_extracted_text += 1  # Count URLs where extraction was successful
                self.unique_urls.update(links)  # Add new event URLs to the unique set

            logging.info(f"def extract_event_links(): Extracted {len(links)} event links from {search_url}.")
            return links

        except Exception as e:
            logging.error(f"def extract_event_links(): Error extracting links from {search_url}: {e}")
            return set()

        finally:
            # If we opened a new page, close it to prevent excessive tabs
            if page != self.logged_in_page:
                page.close()
                logging.info(f"def extract_event_links(): Closed extra page for URL: {search_url}.")


    def fb_group_event_links(self, url):
        """
        Extracts event links from a Facebook group's 'Events' tab.

        Args:
            url (str): The Facebook group URL.

        Returns:
            set: A set of extracted event links.
        """
        fb_group_events = set()

        try:
            self.total_url_attempts += 1  # Count each attempt

            # Ensure the logged-in page is valid
            try:
                if not self.logged_in_page or self.logged_in_page.url == "about:blank":
                    raise Exception("Logged-in page is no longer valid.")
                page = self.logged_in_page  # Reuse the logged-in page
            except Exception:
                logging.warning("fb_group_event_links(): Logged-in page is closed or invalid, opening a new one.")
                page = self.context.new_page()  # Open a new page in the existing context

            # Navigate to the group's page
            timeout_value = random.randint(10000, 15000)  # Randomized timeout
            logging.info(f"fb_group_event_links(): Navigating to {url} with timeout {timeout_value} ms.")
            page.goto(url, timeout=timeout_value)
            page.wait_for_timeout(random.randint(4000, 6000))  # Randomized wait

            # Wait for the "Events" tab to appear
            events_tab_selector = "a[href*='/events' i]"
            try:
                page.wait_for_selector(events_tab_selector, timeout=random.randint(5000, 10000))
                logging.info(f"fb_group_event_links(): Found 'Events' tab for group: {url}. Clicking it.")
                page.click(events_tab_selector)
                page.wait_for_timeout(random.randint(3000, 5000))  # Randomized wait
            except Exception:
                logging.info(f"fb_group_event_links(): No 'Events' tab found for group: {url}.")
                return fb_group_events  # Exit early if no Events tab exists

            # Scroll and gather links on the "Events" tab
            for _ in range(self.config['crawling']['scroll_depth']):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(random.randint(1500, 3000))  # Random scroll wait

            # Extract event links
            content = page.content()
            fb_group_events = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))

            # Update statistics
            if fb_group_events:
                self.urls_with_extracted_text += 1  # Count URLs where extraction was successful
                self.unique_urls.update(fb_group_events)  # Add new event URLs to the unique set

            logging.info(f"fb_group_event_links(): Extracted {len(fb_group_events)} event links "
                        f"from the 'Events' tab of group: {url}")

        except Exception as e:
            logging.error(f"fb_group_event_links(): Error extracting event links from {url}. Error: {e}")

        finally:
            # Close any extra pages if a new one was created
            if page != self.logged_in_page:
                page.close()
                logging.info(f"fb_group_event_links(): Closed extra page for URL: {url}.")

        return fb_group_events
    

    def extract_event_text(self, link):
        """
        Extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            link (str): The event link.

        Returns:
            str: The extracted relevant text content, or None if no relevant text is found.
        """
        self.total_url_attempts += 1  # Count URL contact attempts
        try:
            # Check if the logged-in page is still valid; otherwise, create a new one
            try:
                if not self.logged_in_page or self.logged_in_page.url == "about:blank":
                    raise Exception("Logged-in page is no longer valid.")
                page = self.logged_in_page  # Use the logged-in page if it's valid
            except Exception:
                logging.warning("def extract_event_text(): Logged-in page is closed or invalid, opening a new one.")
                page = self.context.new_page()  # Open a new page in the existing context

            timeout_value = random.randint(8000, 12000)  # Random timeout to avoid bot detection
            logging.info(f"def extract_event_text(): Navigating to {link} with timeout {timeout_value} ms.")
            page.goto(link, timeout=timeout_value)

            # Look for all buttons or links with text "See more" and click them
            more_buttons = page.query_selector_all("text=/See more/i")
            if more_buttons:
                for more_button in more_buttons:
                    try:
                        more_button.wait_for_element_state("stable", timeout=random.randint(2000, 4000))  # Ensure button is stable
                        more_button.click()
                        page.wait_for_timeout(random.randint(4000, 8000))  # Random wait after clicking
                        logging.info(f"Clicked 'See more' button in URL: {link}")
                    except Exception as e:
                        logging.warning(f"Could not click 'See more' button in URL {link}: {e}")
            else:
                logging.debug(f"No 'See more' buttons found in URL: {link}")

            # Randomized wait before extracting content
            page.wait_for_timeout(random.randint(5000, 7000))

            # Extract page content
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            extracted_text = ' '.join(soup.stripped_strings)

            if extracted_text:
                self.urls_with_extracted_text += 1  # Count URLs where text was extracted
                logging.info(f"def extract_event_text(): Extracted raw text ({len(extracted_text)} chars) from {link}")

                # Check for keywords in the extracted text
                found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
                if found_keywords:
                    self.urls_with_found_keywords += 1  # Count URLs where keywords were found

                    if extracted_text and 'facebook.com/events/' in link:
                        event_extracted_text = self.extract_relevant_text(extracted_text, link)
                        if event_extracted_text:
                            logging.info(f"def extract_event_text(): Extracted relevant event text from {link}: {len(event_extracted_text)} chars.")
                            return event_extracted_text
                        else:
                            logging.warning(f"def extract_event_text(): When regex ran, returned None in {link}.")
                            logging.info(f"def extract_event_text(): Returned original extracted_text not just event_extracted_text in {link}")
                            return extracted_text
                else:
                    logging.info(f"def extract_event_text(): No keywords found in extracted text for URL: {link}.")
                    return None
            else:
                logging.warning(f"def extract_event_text(): No text extracted from {link}.")
                return None
            
        # Force releasing any extra windows by putting a finally block in after a try and except block
        except Exception as e:
            logging.warning(f"def extract_event_text(): This {link}: {e} did not work. It could be for a variety of reasons. "
                            "It could be that the page is not loading, the page is not found, or the page is not in the right format.")
            return None

        finally:
            # If we opened a new page, close it to prevent excessive tabs
            if page != self.logged_in_page:
                page.close()
                logging.info(f"def extract_event_text(): Closed extra page for URL: {link}.")
    

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

        try:
            for keyword in keywords:
                search_url = f"{base_url} {location} {keyword}"
                event_links = self.extract_event_links(search_url)
                logging.info(f"def scrape_events: Used {search_url} to get {len(event_links)} event_links\n")

                self.total_url_attempts += len(event_links)  # Update total URL attempts

                for link in event_links:
                    if link in self.urls_visited:
                        continue  # Skip already visited URLs
                    else:
                        self.urls_visited.add(link)
                        self.unique_urls_count += 1  # Increment unique URL count

                        if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                            logging.info("def scrape_events(): Reached the URL visit limit. Stopping the scraping process.")
                            return search_url, extracted_text_list

                        extracted_text = self.extract_event_text(link)

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

        except Exception as e:
            logging.error(f"def scrape_events: Failed to scrape events: {e}")
            return None, []
        

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

                        self.urls_visited.add(url)  # Add to visited URLs

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
        1. Gets all of the urls from the urls table where the link is like '%facebook%'.
        2. For each url, extracts text and processes it.
        3. If valid events are found, writes them to the database; otherwise, updates the URL.
        4. Writes every processed URL—including event links—to the checkpoint CSV.
        """
        # Check and see if this is a checkpoint run
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

        # Add checkpoint columns for base URLs (these columns will also be used for event links)
        fb_urls_df['processed'] = False
        fb_urls_df['events_processed'] = False

        # Write initial checkpoint to disk
        fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
        
        if fb_urls_df.shape[0] > 0:
            for idx, row in fb_urls_df.iterrows():
                base_url = row['link']
                source = row['source']
                keywords = row['keywords']
                logging.info(f"def driver_fb_urls(): Processing URL: {base_url}")
                if base_url in self.urls_visited:
                    continue
                else:
                    self.urls_visited.add(base_url)
                    self.process_fb_url(base_url, source, keywords)

                    # Mark the base URL as processed (base processing done)
                    fb_urls_df.loc[fb_urls_df['link'] == base_url, 'processed'] = True

                    # Overwrite the checkpoint file after processing the base URL
                    fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                    logging.info(f"def driver_fb_urls(): Base URL {base_url} marked as processed.")

                    # Check urls_run_limit for the base URL
                    if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                        break

                    # Extract event links from the base URL
                    fb_event_links = self.extract_event_links(base_url)

                    # If this is a group URL, try to extract additional event links
                    if "facebook.com/groups" in base_url:
                        fb_group_events = self.fb_group_event_links(base_url)
                        if fb_group_events:
                            if fb_event_links:
                                fb_event_links.update(fb_group_events)
                            else:
                                fb_event_links = fb_group_events.copy()
                    
                    # Process each event link and update the checkpoint file
                    for event_url in fb_event_links:
                        if event_url in self.urls_visited:
                            continue
                        else:
                            self.urls_visited.add(event_url)
                            self.process_fb_url(event_url, source, keywords)

                            # If the event URL is not already in fb_urls_df, add a new row.
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
                                # If the event URL already exists, mark it as processed.
                                fb_urls_df.loc[fb_urls_df['link'] == event_url, 'processed'] = True
                                fb_urls_df.loc[fb_urls_df['link'] == event_url, 'events_processed'] = True
                            
                            # Write the updated dataframe to the checkpoint file.
                            fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                            logging.info(f"def driver_fb_urls(): Event URL {event_url} marked as processed.")

                            # Check urls_run_limit for event links
                            if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                                break

                    # Mark that the base URL has finished processing event links as well.
                    fb_urls_df.loc[fb_urls_df['link'] == base_url, 'events_processed'] = True
                    fb_urls_df.to_csv(self.config['checkpoint']['fb_urls'], index=False)
                    logging.info(f"def driver_fb_urls(): Base URL {base_url} event links marked as processed.")
        else:
            logging.warning("def driver_fb_urls(): No rows returned from the sql query.")


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