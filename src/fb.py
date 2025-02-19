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
            filemode='w',
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
import pandas as pd
from playwright.sync_api import sync_playwright
from pynput.keyboard import Controller, Key
import random
import re
import requests
from sqlalchemy import text
import time
import yaml

# Import other classes
from credentials import get_credentials
from db import DatabaseHandler
from llm import LLMHandler


class FacebookEventScraper():
    def __init__(self, config_path="config/config.yaml"):
        # Initialize base class (loads config)

        # Get config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Start time for manual intervention
        self.start_time = datetime.now()

        # Start Playwright and log into Facebook once
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.config['crawling']['headless'])

        self.context = self.browser.new_context()
        self.logged_in_page = self.context.new_page()

        # Attempt to log into Facebook and store the logged-in page for reuse
        if self.login_to_facebook(self.logged_in_page, self.browser):
            logging.info("Facebook login successful during initialization.")
        else:
            logging.error("Facebook login failed during initialization.")

        # Set up the set for urls visited
        self.urls_visited = set()

        # Create a keyboard controller
        self.keyboard = Controller()
    

    def login_to_facebook(self, page, browser):
        """
        Logs into Facebook using credentials from the configuration, handles potential additional
        login prompts, and saves session state for future reuse. 
        Manual intervention required for captcha challenges.
        
        Args:
            page (playwright.sync_api.Page): The Playwright page instance.
            browser (playwright.sync_api.Browser): The Playwright browser instance.
        
        Returns:
            bool: True if login is successful, False otherwise.
        """
        # Try to load saved authentication state if it exists
        try:
            context = browser.new_context(storage_state="auth.json")
            page = context.new_page()
            page.goto("https://www.facebook.com/", timeout=60000)

            # If this URL doesn't redirect to login, assume logged in
            if page.is_visible("div[aria-label='Search Facebook']"):
                logging.info("def login_to_facebook(): Successfully logged in.")
                return True
            
        except Exception as e:
            logging.info("def login_to_facebook(): No valid saved session found. Proceeding with manual login.")
        
        email, password, _ = get_credentials('Facebook')

        page.goto("https://www.facebook.com/", timeout=30000)
        if page.is_visible("input[name='email']") and page.is_visible("input[name='pass']"):
            page.fill("input[name='email']", email)
            page.fill("input[name='pass']", password)
            page.click("button[name='login']")
        else:
            logging.info("Already logged in or login page not detected.")

        # Wait briefly for potential navigation or additional login prompts
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(10000)  # 10 seconds

        # Detect and handle additional login prompts if they appear
        def check_additional_login(p):
            if p.is_visible("input[name='email']") and p.is_visible("input[name='pass']"):
                logging.info("Additional login prompt detected. Re-filling credentials.")
                p.fill("input[name='email']", email)
                p.fill("input[name='pass']", password)
                p.click("button[name='login']")
                # Wait for navigation or further prompts
                p.wait_for_timeout(5000)
                return True
            return False

        # Attempt to handle any subsequent login prompts up to a few times if necessary
        attempts = 0
        while attempts < 3 and check_additional_login(page):
            attempts += 1

        # Prompt for manual captcha/challenge resolution if necessary
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()

        if elapsed_time <= 120:
            logging.warning("def login_to_facebook(): Please solve any captcha or challenge in the browser, then press Enter here to continue...")
            input("def login_to_facebook(): After solving captcha/challenge (if any), press Enter to continue...")
        else:
            # Simulate pressing the Enter key
            time.sleep(5) 
            # Simulate pressing the Enter key
            self.keyboard.press(Key.enter)
            self.keyboard.release(Key.enter)
            logging.info("def login_to_facebook(): Automated Enter key press to continue after 2 minutes.")

        # Wait for navigation to complete after manual intervention
        page.wait_for_timeout(5000)

        # Check if login failed by examining the URL
        if "login" in page.url.lower():
            logging.error("Login failed. Please check your credentials or solve captcha challenges.")
            return False

        # Save session state for future reuse
        try:
            context = page.context
            context.storage_state(path="auth.json")
            logging.info("Session state saved for future use.")
        except Exception as e:
            logging.warning(f"Could not save session state: {e}")

        # Mark page as logged-in for future reuse
        self.logged_in_page = page
        logging.info("Login to Facebook successful.")
        return True
    

    def extract_event_links(self, search_url):
        """
        Extracts event links from a search page using Playwright and regex.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.
            search_url (str): The Facebook events search URL.

        Returns:
            set: A set of extracted event links.
        """
        # Use the stored already logged-in page
        page = self.logged_in_page  
        page.goto(search_url, timeout=10000)
        page.wait_for_timeout(5000)

        for _ in range(self.config['crawling']['scroll_depth']):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

        content = page.content()
        links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))
        logging.info(f"def extract_event_links(): Extracted {len(links)} event links from {search_url}.")

        return links
    

    def extract_event_text(self, link):
        """
        Extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            link (str): The event link.

        Returns:
            str: The extracted relevant text content, or None if no relevant text is found.
        """
        # Initialize event extracted text
        event_extracted_text = None

        try:
            page = self.logged_in_page
            page.goto(link, timeout=10000)

            # Look for all buttons or link with text "See more" (case insensitive) and click them
            more_buttons = page.query_selector_all("text=/See more/i")
            for more_button in more_buttons:
                try:
                    more_button.wait_for_element_state("stable", timeout=3000)  # Ensure button is stable
                    more_button.click()
                    # Randomize wait time after clicking each "See more"
                    page.wait_for_timeout(random.randint(10000, 20000))
                    logging.info(f"Clicked 'See more' button in URL: {link}")
                except Exception as e:
                    logging.warning(f"Could not click 'See more' button in URL {link}: {e}")

            if not more_buttons:
                logging.debug(f"No 'See more' buttons found in URL: {link}")

            page.wait_for_timeout(5000)
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            extracted_text = ' '.join(soup.stripped_strings)
            logging.info(f"def extract_event_text(): Extracted raw text from {link}: {extracted_text}")

            if extracted_text and 'facebook.com/events/' in link:
                event_extracted_text = self.extract_relevant_text(extracted_text, link)

            # Check if we got event extracted text
            if event_extracted_text:
                logging.info(f"Extracted relevant event text from {link}: {extracted_text}")
                return event_extracted_text
            else:
                logging.info(f"def extract_event_text(): No extracted_text for url: {link}")
                return None

        except Exception as e:
            logging.error(f"def extract_event_text(): Failed to extract text from {link}: {e}")
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

                for link in event_links:
                    if link in self.urls_visited:
                        continue  # Skip already visited URLs
                    else:
                        self.urls_visited.add(link)
                        if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                            logging.info("Reached the URL visit limit. Stopping the scraping process.")
                            return search_url, extracted_text_list
                        
                        extracted_text = self.extract_event_text(link)
                        if extracted_text:  # Only add if text was successfully extracted
                            extracted_text_list.append((link, extracted_text))
                        self.urls_visited.add(link)
                        logging.debug(f"Visited URL: {link}. Total visited: {len(self.urls_visited)}")

                        # Check if the limit has been reached
                        if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                            logging.info("Reached the URL visit limit. Stopping the scraping process.")
                            return search_url, extracted_text_list

                    # Get second-level link
                    second_level_links = self.extract_event_links(link)
                    for second_level_link in second_level_links:
                        if second_level_link in self.urls_visited:
                            continue  # Skip already visited URLs
                        else:
                            self.urls_visited.add(second_level_link)
                            if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                                logging.info("Reached the URL visit limit. Stopping the scraping process.")
                                return search_url, extracted_text_list
                            
                            second_level_text = self.extract_event_text(second_level_link)
                            if second_level_text:  # Only add if text was successfully extracted
                                extracted_text_list.append((second_level_link, second_level_text))
                            self.urls_visited.add(second_level_link)
                            logging.debug(f"Visited URL: {second_level_link}. Total visited: {len(self.urls_visited)}")

                            # Check if the limit has been reached
                            if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                                logging.info("Reached the URL visit limit. Stopping the scraping process.")
                                return search_url, extracted_text_list

        except Exception as e:
            logging.error(f"Failed to scrape events: {e}")

        logging.info(f"def scrape_events(): Extracted text from {len(extracted_text_list)} events.")

        # Return the last search_url processed and the list of extracted event data
        return search_url, extracted_text_list


    def extract_text_from_fb_url(self, url):
        """
        Extracts text content from a Facebook event URL using Playwright and BeautifulSoup.

        Args:
            url (str): The Facebook event URL.

        Returns:
            str: Extracted text content from the Facebook event page.
        """
        fb_status = fb_scraper.login_to_facebook(self.logged_in_page, self.browser)

        if fb_status:
            logging.info("def extract_text_from_fb_url(): Successfully logged into Facebook.")
            extracted_text = fb_scraper.extract_event_text(url)

            return extracted_text
    

    def google_search(self, query, num_results=10):
        """
        Perform a Google search and extract titles and URLs of the results.

        Args:
            query (str): The search query.
            api_key (str): Your Google API key.
            cse_id (str): Your Custom Search Engine ID.
            num_results (int): The number of search results to fetch (max 10 per request).

        Returns:
            list: A list of dictionaries containing 'title' and 'url' for each search result.
        """
        _, api_key, cse_id = get_credentials('Google')
        service = build("customsearch", "v1", developerKey=api_key)
        results = []
        
        try:
            response = service.cse().list(
                q=query,
                cx=cse_id,
                num=num_results  # Fetch up to `num_results` results (max 10 per request).
            ).execute()

            # Extract titles and link from the response
            if 'items' in response:
                for item in response['items']:
                    title = item.get('title')  # Get the title of the search result
                    url = item.get('link')  # Get the URL of the search result
                    results.append((title, url))
        except Exception as e:
             print(f"An error occurred: {e}")

        return results
    
    
    @staticmethod
    def extract_text_from_url(url):
        """
        Extracts text content from a given URL using requests and BeautifulSoup.

        Args:
            url (str): The URL of the webpage to scrape.

        Returns:
            str: Extracted text content from the webpage.
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            return text
        except Exception as e:
            logging.error(f"Failed to fetch content from {url}: {e}")
            return None
        

    @staticmethod
    def convert_facebook_url(original_url):
        """
        Captures the event ID from 'm.facebook.com' URLs and returns
        'https://www.facebook.com/events/<event_id>/'.

        Args:
            original_url (str): The original Facebook URL.

        Returns:
            str: Converted Facebook URL.
        """
        pattern = r'^https://m\.facebook\.com/events/([^/]+)/?.*'
        replacement = r'https://www.facebook.com/events/\1/'

        return re.sub(pattern, replacement, original_url)
    

    def scrape_and_process(self, query):
        """
        Scrapes Google Search results, processes them, and updates the database.

        Args:
            query (str): The search query.

        Returns:
            None
        """
        logging.info(f"Starting scrape and process for query: {query}.")
        extracted_text = ''
        results = self.google_search(query, self.config['crawling']['urls_run_limit'])

        # # We have to get thru all of them before we know if we need to bail out to Facebook
        for title, url in results:

            # Check and see if url has already been visited
            if url in self.urls_visited:
                pass

            else:
                # Check and see if we have hit our limit for urls to visit
                if self.config['crawling']['urls_run_limit'] > len(self.urls_visited):
                    return url, None, 'default'
                else:
                    # Add to self.urls_visited
                    self.urls_visited.add(url)

                    # Check if the query is similar to the title
                    similarity = fuzz.token_set_ratio(query, title)
                    if similarity > self.config['constants']['fuzzywuzzy_threshold'] and 'facebook' not in url:  # 80% threshold
                        logging.info(f"def scrape_and_process(): Relevant result found: Title: {title}, URL: {url}.")

                        # Scrape text from the non-Facebook URL
                        extracted_text = self.extract_text_from_url(url)
                        if extracted_text:
                            logging.info(f"def scrape_and_process(): Text extracted from url: {url}.")
                            if 'allevents.in' in url:
                                return url, extracted_text, 'allevents'
                            else:
                                return url, extracted_text, 'single_event'
            
        # Might be a facebook event
        for title, url in results:

            # Check and see if we have hit our limit for urls to visit
            if config['crawling']['urls_run_limit'] > len(self.urls_visited):
                return url, None, 'default'
                
            # Check if the query is similar to the title
            similarity = fuzz.token_set_ratio(query, title)
            if similarity > self.config['constants']['fuzzywuzzy_threshold'] and 'facebook' in url:  # 80% threshold
                logging.info(f"Relevant result found: Title: {title}, URL: {url}.")

                # Convert Facebook URL
                url = self.convert_facebook_url(url)
                logging.info(f"def scrape_and_process(): Extracting text from facebook url: {url}.")

                # Scrape text from the Facebook URL
                extracted_text = self.extract_text_from_fb_url(url)
                if extracted_text:
                    logging.info(f"def scrape_and_process(): Text extracted from facebook url: {url}."
                                 f"Extracted text: {extracted_text}")
                    return url, extracted_text, 'single_event'
                else:
                    logging.info(f"def scrape_and_process(): No relevant results found for: query: {query}.")
                    return url, None, 'default'
        
        logging.info(f"def scrape_and_process(): No relevant results found for: query: {query}.")        
        return None, None, 'default'
    

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

        # Establish default values
        update_other_link = 'No'
        relevant = False
        increment_crawl_try = 1

        if url:
            # Extract text and update db
            extracted_text = self.extract_text_from_fb_url(url)
            if not extracted_text:
                db_handler.update_url(url, update_other_link, relevant, increment_crawl_try)
                logging.info(f"def process_fb_url(): No text extracted for Facebook URL: {url}.")
                return
            
            # Check and see if keywords is a list. If it is not, convert it to a list
            if not isinstance(keywords, list):
                keywords_list = keywords.split(',')

            # Check and see if there are any of the keywords in keywords in the extracted_text
            found_keywords = [keyword for keyword in keywords_list if keyword in extracted_text]
            if found_keywords:
                logging.info(f"def process_fb_url(): Found keywords: {found_keywords} in extracted text for URL: {url}.")

                # Generate prompt and query LLM
                prompt = llm_handler.generate_prompt(url, extracted_text, 'fb')
                llm_response = llm_handler.query_llm(prompt)

                # If no events found delete the url from fb_urls table
                if "No events found" in llm_response:
                    db_handler.update_url(url, update_other_link, relevant, increment_crawl_try)
                    logging.info(f"def process_fb_url(): No valid events found for URL: {url}. URL deleted from urls table.")
                
                else:
                    # If parsed_result extract and write to db
                    parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
                    if parsed_result:
                        events_df = pd.DataFrame(parsed_result)

                        # If the URL field is empty, fill it with the current URL.
                        if events_df['url'].values[0] == '':
                            events_df.loc[0, 'url'] = url

                        db_handler.write_events_to_db(events_df, url, source, keywords)
                        logging.info(f"def process_fb_url(): Valid events found for Facebook URL: {url}. Events written to database.")

                        # Update url as relevant
                        relevant = True
                        db_handler.update_url(url, update_other_link, relevant, increment_crawl_try)
                        logging.info(f"def process_fb_url(): relevant and updated {url}")
            else:
                relevant = False
                update_other_link = ''
                increment_crawl_try = 1
                db_handler.update_url(url, update_other_link, relevant, increment_crawl_try)
                logging.info(f"def process_fb_url(): No keywords found in extracted text for URL: {url}.")
                return
        else:    
            logging.info(f"def process_fb_url(): No URL found for processing.")
            return


    def driver_fb_urls(self):
        """
        1. Gets all of the urls from fb_urls table.
        2. For each url, extracts text and processes it.
        3. If valid events are found, writes them to the database; otherwise, updates the URL.
        """
        # ********Temp start
        query =text("""
        SELECT * 
        FROM urls
        WHERE link ILIKE :link_pattern
        """)
        params = {'link_pattern': '%facebook%'}
        fb_urls_df = pd.read_sql(query, db_handler.conn, params=params)
        logging.info(f"def driver_fb_urls(): Retrieved {fb_urls_df.shape[0]} Facebook URLs from the database.")
        
        if fb_urls_df.shape[0] > 0:
            for _, row in fb_urls_df.iterrows():
                url = row['link']
                source = row['source']
                keywords = row['keywords']
                logging.info(f"def driver_fb_urls(): Processing URL: {url}")
                if url in self.urls_visited:
                    pass
                else:
                    self.urls_visited.add(url)
                    self.process_fb_url(url, source, keywords)
                    if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                        break   

                    # Get the event links from the facebook url
                    fb_event_links = self.extract_event_links(url)

                    # Get the event link from the event tab from the group page
                    if "facebook.com/groups" in url:
                        fb_group_events = self.fb_group_event_links(url)

                        # Merge fb_group_events with fb_event_links
                        if fb_group_events:
                            if fb_event_links:
                                fb_event_links.update(fb_group_events)
                            else:
                                fb_event_links = fb_group_events.union([url])

                    # Process the event links
                    for url in fb_event_links:
                        if url in self.urls_visited:
                            pass
                        else:
                            self.urls_visited.add(url)
                            self.process_fb_url(url, source, keywords)
                            if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                                break   
        else:
            logging.warning("def driver_fb_urls(): No rows returned from the sql query.")


    def fb_group_event_links(self, url):

        # Get the event link from the event tab from the group page
        fb_group_events = set()
        
        try:
            # Navigate to the group page's "Events" tab
            page = self.logged_in_page
            page.goto(url, timeout=10000)

            # Look for the "Events" tab and click it
            events_tab_selector = "a[href*='/events' i]"
            if page.is_visible(events_tab_selector):
                page.click(events_tab_selector)
                page.wait_for_timeout(3000)
                logging.info(f"Navigated to the 'Events' tab for group: {url}")

                # Scroll and gather links on the "Events" tab
                for _ in range(self.config['crawling']['scroll_depth']):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(2000)

                # Extract link matching the event URL pattern
                content = page.content()
                fb_group_events = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))
                logging.info(f"def fb_group_event_links(): Extracted {len(fb_group_events)} event links "
                             f"from the 'Events' tab of group: {url}")

            else:
                logging.info(f"No 'Events' tab found for group: {url}")
        except Exception as e:
            logging.error(f"Failed to extract event links from the 'Events' tab of group: {url}. Error: {e}")

        # Close the page
        page.close()

        return fb_group_events


    def driver_fb_search(self):
        """
        1. Reads in the keywords CSV file.
        2. Creates search terms for Facebook searches using each keyword.
        3. Scrapes events from search results.
        4. For each extracted Facebook URL, processes it using process_fb_url.
        """
        # Read in the keywords file
        keywords_df = pd.read_csv(self.config['input']['data_keywords'])

        for _, row in keywords_df.iterrows():
            keywords_list = row['keywords'].split(',')
            source = row.get('source', '')

            # Scrape the events
            search_url, extracted_text_list = self.scrape_events(keywords_list)
            logging.info(f"def driver_fb_search(): Extracted text based on search_url: {search_url}.")

            if extracted_text_list:
                # Save extracted text data to CSV
                extracted_text_df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])
                output_path = self.config['output']['fb_search_results']
                extracted_text_df.to_csv(output_path, index=False)
                logging.info(f"def driver_fb_search(): Extracted text data written to {output_path}.")

                for url, extracted_text in extracted_text_list:
                    # Check if URL has already been visited and if it is a facebook url
                    if url not in self.urls_visited and 'facebook.com' in url:
                        logging.info(f"def driver_fb_search(): Processing Facebook URL: {url}")

                        # Add to self.urls_visited and check to see if we are at our crawl limit
                        self.urls_visited.add(url)
                        if len(self.urls_visited) >= self.config['crawling']['urls_run_limit']:
                            break

                        # See if any of the keywords are in the extracted_text
                        found_keywords = [keyword for keyword in keywords_list if keyword in extracted_text]
                        if found_keywords:
                            logging.info(f"def driver_fb_search(): Found keywords: {keywords_list} in extracted text for URL: {url}.")
                        
                            # Set prompt and process_llm_response
                            prompt = 'fb'
                            llm_status = llm_handler.process_llm_response(url, extracted_text, source, keywords_list, prompt)
                            if llm_status:
                                prompt = 'default'
                                llm_status = llm_handler.process_llm_response(url, extracted_text, source, keywords_list, prompt)
                                logging.info(f"def driver_fb_search(): Processed Facebook URL via process_llm_response: {url}.")
                            else:
                                logging.info(f"def driver_fb_search(): No valid events found for URL: {url}.")
                        else:
                            logging.info(f"def driver_fb_search(): No keywords found in extracted text for URL: {url}.")
                    else:
                        logging.info(f"def driver_fb_search(): URL already visited: {url}")
            else:
                logging.info(f"def driver_fb_search(): No extracted text found for search_url: {search_url}.")
                
    
    def driver_no_urls(self):
        """
        Queries the database for events without URLs, scrapes the web for URLs and event data, processes the data using an LLM, 
        and updates the database with the new information.
        """
        query = "SELECT * FROM events WHERE url = :url"
        params = {'url': ''}
        result = db_handler.execute_query(query, params)

        if result:
            rows = result.fetchall()
            no_urls_df = pd.DataFrame(rows, columns=result.keys())

            logging.info(f"def driver_no_urls(): Retrieved {len(no_urls_df)} events without URLs.")

            # Reduce the number of events to process for testing
            if config['testing']['status']:
                no_urls_df = no_urls_df.head(self.config['crawling']['urls_run_limit'])

            for _, row in no_urls_df.iterrows():
                query_text = row['event_name']
                source = row['source']
                keywords = row['dance_style']
                url, extracted_text, prompt_type = self.scrape_and_process(query_text)

                if extracted_text:

                    # See if any of the keywords are in the extracted_text. Put keywords into a list
                    keywords_list = keywords.split(',')
                    found_keywords = [keyword for keyword in keywords_list if keyword in extracted_text]
                    if found_keywords:
                        logging.info(f"def driver_no_urls(): Found keywords: {found_keywords} in extracted text for URL: {url}.")
                        prompt = llm_handler.generate_prompt(url, extracted_text, prompt_type)
                        llm_response = llm_handler.query_llm(prompt)

                        if "No events found" in llm_response:
                            db_handler.delete_event_and_update_url(url, row['event_name'], row['start_date'])
                        else:
                            parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
                            events_df = pd.DataFrame(parsed_result)
                            logging.info(f"def driver_no_urls(): URL is: {url}")
                            events_df.to_csv(self.config['debug']['before_url_updated'], index=False)

                            if events_df['url'].values[0] == '':
                                events_df.loc[0, 'url'] = url
                                events_df.to_csv(self.config['debug']['after_url_updated'], index=False)

                            db_handler.write_events_to_db(events_df, url, source, keywords)
            else:
                logging.info("def driver_no_urls(): No events without URLs found in the database.")
                return None


if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Instantiate the class libraries
    db_handler = DatabaseHandler(config)
    fb_scraper = FacebookEventScraper(config_path='config/config.yaml')
    llm_handler = LLMHandler(config_path='config/config.yaml')

    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")

    logging.info(f"def __main__: Running driver_fb_urls.")
    fb_scraper.driver_fb_urls()

    # logging.info(f"def __main__: Running driver_fb_search.")
    fb_scraper.driver_fb_search()

    # logging.info(f"def __main__: Running driver_no_urls.")
    fb_scraper.driver_no_urls()

    # Close the browser and stop Playwright
    fb_scraper.browser.close()
    fb_scraper.playwright.stop()

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
