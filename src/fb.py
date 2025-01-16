from bs4 import BeautifulSoup
from datetime import datetime
from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import logging
import pandas as pd
from playwright.sync_api import sync_playwright
import re
import requests
import yaml

# Import other classes
from bh import BaseHandler
from db import DatabaseHandler
from llm import LLMHandler


# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Instantiate DatabaseHandler with the configuration dictionary
db_handler = DatabaseHandler(config)


class FacebookEventScraper(BaseHandler):
    def __init__(self, config_path="config/config.yaml"):
        # Initialize base class (loads config, sets up logging, etc.)
        super().__init__(config_path)

        # Start Playwright and log into Facebook once
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.config['crawling']['headless'])
        self.context = self.browser.new_context()
        self.logged_in_page = self.context.new_page()

        # Attempt to log into Facebook and store the logged-in page for reuse
        if not self.login_to_facebook(self.logged_in_page):
            logging.error("Facebook login failed during initialization.")


    def get_credentials(self, organization):
        """
        Retrieves credentials for a given organization from the keys file.

        Args:
            organization (str): The organization for which to retrieve credentials.

        Returns:
            tuple: appid_uid, key_pw, access_token for the organization.
        """
        keys_df = pd.read_csv(self.config['input']['keys'])
        keys_df = keys_df[keys_df['organization'] == organization]
        appid_uid, key_pw, cse_id = keys_df.iloc[0][['appid_uid', 'key_pw', 'cse_id']]
        logging.info(f"def get_credentials(): Retrieved credentials for {organization}.")
        return appid_uid, key_pw, cse_id
    

    def login_to_facebook(self, page):
        """
        Logs into Facebook using credentials from the self.configuration, and handles potential captcha challenges.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        # If already logged in, reuse the page
        if hasattr(self, 'logged_in_page') and self.logged_in_page:
            logging.info("Already logged into Facebook. Skipping login.")
            return True

        email, password, _ = self.get_credentials('Facebook')

        page.goto("https://www.facebook.com/login", timeout=60000)
        page.fill("input[name='email']", email)
        page.fill("input[name='pass']", password)
        page.click("button[name='login']")

        # Wait for navigation or potential challenge
        page.wait_for_timeout(10000)

        # Always prompt for manual captcha resolution
        logging.warning("Please solve any captcha or challenge in the browser, then press Enter here to continue...")
        input("After solving captcha/challenge (if any), press Enter to continue...")

        # Wait a bit more for navigation to complete after manual intervention
        page.wait_for_timeout(5000)

        # Check if login failed by examining the URL
        if "login" in page.url.lower():
            logging.error("Login failed. Please check your credentials.")
            return False

        # Mark page as logged-in for future reuse
        self.logged_in_page = page
        logging.info("def login_to_facebook(): Login successful.")
        return True
    

    def extract_event_links(self, page, search_url):
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
            page (playwright.sync_api.Page): The Playwright page instance.
            link (str): The event link.

        Returns:
            str: The extracted text content.
        """
        # Use the stored already logged-in page
        try:
            page = self.logged_in_page 
            page.goto(link, timeout=10000)
            page.wait_for_timeout(5000)
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            extracted_text = ' '.join(soup.stripped_strings)
            logging.info(f"def extract_event_text(): Extracted text from {link}.")
        except Exception as e:
            logging.error(f"Failed to extract text from {link}: {e}")
            extracted_text = None

        return extracted_text
    

    def scrape_events(self, keywords):
        """
        Logs into Facebook once, performs searches for keywords, and extracts event links and text.

        Args:
            keywords (list): List of keywords to search for.

        Returns:
            tuple: The last search_url used and a list of tuples containing event links and extracted text.
        """
        base_url = self.config['constants']['fb_base_url']
        location = self.config['constants']['location']
        extracted_text_list = []
        urls_visited = set()

        try:
            # Use the stored logged-in page
            page = self.logged_in_page

            for keyword in keywords:
                search_url = f"{base_url} {location} {keyword}"
                event_links = self.extract_event_links(page, search_url)
                logging.info(f"def scrape_events: Used {search_url} to get {len(event_links)} event_links\n")

                for link in event_links:
                    if link in urls_visited:
                        continue

                    extracted_text = self.extract_event_text(link)
                    extracted_text_list.append((link, extracted_text))
                    urls_visited.add(link)

                    # Get second-level links
                    second_level_links = self.extract_event_links(page, link)
                    for second_level_link in second_level_links:
                        if second_level_link in urls_visited:
                            continue

                        second_level_text = self.extract_event_text(second_level_link)
                        extracted_text_list.append((second_level_link, second_level_text))
                        urls_visited.add(second_level_link)

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
        # Use the stored logged-in page
        page = self.logged_in_page

        fb_status = fb_scraper.login_to_facebook(page)

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
        _, api_key, cse_id = self.get_credentials('Google')
        service = build("customsearch", "v1", developerKey=api_key)
        results = []
        
        #try:
        response = service.cse().list(
            q=query,
            cx=cse_id,
            num=num_results  # Fetch up to `num_results` results (max 10 per request).
        ).execute()

        # Extract titles and links from the response
        if 'items' in response:
            for item in response['items']:
                title = item.get('title')  # Get the title of the search result
                url = item.get('link')  # Get the URL of the search result
                results.append((title, url))

        # except Exception as e:
        #     print(f"An error occurred: {e}")

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
        results = self.google_search(query, 5)

        for title, url in results:
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
            
        # Check if the URL is a Facebook URL
        # We have to get thru all of them before we know if we need to bail out to Facebook
        for title, url in results:
            # Check if the query is similar to the title
            similarity = fuzz.token_set_ratio(query, title)
            if similarity > self.config['constants']['fuzzywuzzy_threshold'] and 'facebook' in url:  # 80% threshold
                logging.info(f"Relevant result found: Title: {title}, URL: {url}.")

                # Convert Facebook URL
                url = self.convert_facebook_url(url)
                logging.info(f"def scrape_and_process(): fb_url is: {url}")

                logging.info(f"def scrape_and_process(): Extracting text from facebook url: {url}.")
                extracted_text = self.extract_text_from_fb_url(url)
                if extracted_text:
                    logging.info(f"def scrape_and_process(): Text extracted from facebook url: {url}.")
                    return url, extracted_text, 'single_event'

        logging.info(f"def scrape_and_process(): No relevant results found for: query: {query}.")
        return url, None, 'default'
            

    def save_to_csv(self, extracted_text_list, output_path):
        """
        Saves extracted event links and text to a CSV file.

        Args:
            extracted_text_list (list): A list of tuples containing event links and extracted text.
            output_path (str): The file path to save the CSV.
        """
        df = pd.DataFrame(extracted_text_list, columns=['url', 'extracted_text'])
        df.to_csv(output_path, index=False)
        logging.info(f"def save_to_csv(): Extracted text data written to {output_path}")
    

    def process_fb_url(self, url, org_name, keywords):
        """
        Processes a single Facebook URL: extracts text, interacts with LLM, and updates the database.

        Args:
            url (str): The Facebook URL to process.
            org_name (str): The organization name associated with the URL.
            keywords (str): Keywords associated with the URL.

        Returns:
            None
        """
        # Sanitize URL if malformed (e.g., duplicate scheme)
        if url.startswith("http://https") or url.startswith("https://https"):
            url = url.replace("http://https", "https://").replace("https://https", "https://")

        extracted_text = self.extract_text_from_fb_url(url)
        if not extracted_text:
            db_handler.delete_url_from_fb_urls(url)
            logging.info(f"def process_fb_url(): No text extracted for Facebook URL: {url}. URL deleted from fb_urls table.")
            return

        prompt = llm_handler.generate_prompt(url, extracted_text, 'fb')
        llm_response = llm_handler.query_llm(prompt, url)

        if "No events found" in llm_response:
            db_handler.delete_url_from_fb_urls(url)
            logging.info(f"def process_fb_url(): No valid events found for URL: {url}. URL deleted from fb_urls table.")
        else:
            parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
            events_df = pd.DataFrame(parsed_result)

            # If the URL field is empty, fill it with the current URL.
            if events_df['url'].values[0] == '':
                events_df.loc[0, 'url'] = url

            db_handler.write_events_to_db(events_df, url, org_name, keywords)
            logging.info(f"def process_fb_url(): Valid events found for Facebook URL: {url}. Events written to database.")


    def driver_fb_urls(self):
        """
        1. Gets all of the urls from fb_urls table.
        2. For each url, extracts text and processes it.
        3. If valid events are found, writes them to the database; otherwise, deletes the URL.
        """
        query = "SELECT url, org_name, keywords FROM fb_urls"
        fb_urls_df = pd.read_sql(query, db_handler.get_db_connection())
        logging.info(f"def driver_fb_urls(): Retrieved {len(fb_urls_df)} Facebook URLs.")

        for _, row in fb_urls_df.iterrows():
            url = row['url']
            org_name = row.get('org_name', '')
            keywords = row.get('keywords', '')
            logging.info(f"def driver_fb_urls(): Processing URL: {url}")
            self.process_fb_url(url, org_name, keywords)


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
            org_name = row.get('org_name', '')
            keywords_str = row.get('keywords', '')

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
                    # If URL is a Facebook URL, process it using the shared helper
                    if 'facebook.com' in url:
                        logging.info(f"def driver_fb_search(): Processing Facebook URL: {url}")
                        self.process_fb_url(url, org_name, keywords_str)
                    else:
                        # Handle non-Facebook URLs if needed
                        pass

    
    def driver_no_urls(self):
        """
        Queries the database for events without URLs, scrapes the web for URLs and event data, processes the data using an LLM, 
        and updates the database with the new information.
        """
        query = "SELECT * FROM events WHERE url = %s"
        params = ('',)
        no_urls_df = pd.read_sql(query, db_handler.get_db_connection(), params=params)
        logging.info(f"def driver_no_urls(): Retrieved {len(no_urls_df)} events without URLs.")

        # Reduce the number of events to process for testing
        no_urls_df = no_urls_df.head(self.config['crawling']['urls_run_limit'])

        for _, row in no_urls_df.iterrows():
            query_text = row['event_name']
            url, extracted_text, prompt_type = self.scrape_and_process(query_text)

            if extracted_text:
                prompt = llm_handler.generate_prompt(url, extracted_text, prompt_type)
                llm_response = llm_handler.query_llm(prompt, url)

                if "No events found" in llm_response:
                    db_handler.delete_event_and_url(url, row['event_name'], row['start_date'])
                else:
                    parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
                    events_df = pd.DataFrame(parsed_result)
                    logging.info(f"def driver_no_urls(): URL is: {url}")
                    events_df.to_csv('before_url_updated.csv', index=False)

                    if events_df['url'].values[0] == '':
                        events_df.loc[0, 'url'] = url
                        events_df.to_csv('after_url_updated.csv', index=False)

                    db_handler.write_events_to_db(events_df, url)

        return None


if __name__ == "__main__":

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Instantiate the class libraries
    fb_scraper = FacebookEventScraper(config_path='config/config.yaml')
    llm_handler = LLMHandler(config_path='config/config.yaml')

    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")
    logging.info(f"def __main__: Running driver_fb_urls.")
    fb_scraper.driver_fb_urls()
    logging.info(f"def __main__: Running driver_fb_search.")
    fb_scraper.driver_fb_search()
    logging.info(f"def __main__: Running driver_no_urls.")
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
