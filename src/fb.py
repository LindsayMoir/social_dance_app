from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import logging
import os
import pandas as pd
from playwright.sync_api import sync_playwright
import re
import requests
import yaml

# Import other classes
from db import DatabaseHandler
from llm import LLMHandler
from scraper import EventSpider


class FacebookEventScraper:
    def __init__(self, config_path="config/config.yaml"):

        # Load configuration from a YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set up logging
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            filemode='w',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        logging.info("FacebookEventScraper initialized.")


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
        Logs into Facebook using credentials from the configuration.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        email, password, _ = self.get_credentials('Facebook')

        page.goto("https://www.facebook.com/login", timeout=60000)
        page.fill("input[name='email']", email)
        page.fill("input[name='pass']", password)
        page.click("button[name='login']")

        page.wait_for_timeout(5000)
        if "login" in page.url:
            logging.error("Login failed. Please check your credentials.")
            return False

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
        page.goto(search_url, timeout=60000)
        page.wait_for_timeout(5000)

        for _ in range(self.config['crawling']['scroll_depth']):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)

        content = page.content()
        links = set(re.findall(r'https://www\.facebook\.com/events/\d+/', content))
        logging.info(f"def extract_event_links(): Extracted {len(links)} event links from {search_url}.")

        return links
    

    def extract_event_text(self, page, link):
        """
        Extracts text from an event page using Playwright and BeautifulSoup.

        Args:
            page (playwright.sync_api.Page): The Playwright page instance.
            link (str): The event link.

        Returns:
            str: The extracted text content.
        """
        page.goto(link, timeout=60000)
        page.wait_for_timeout(5000)
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        extracted_text = ' '.join(soup.stripped_strings)
        logging.info(f"def extract_event_text(): Extracted text from {link}.")

        return extracted_text
    

    def scrape_events(self, keywords):
        """
        Logs into Facebook, performs searches for keywords, and extracts event links and text.

        Args:
            keywords (list): List of keywords to search for.

        Returns:
            list: A list of tuples containing event links and extracted text.
        """
        base_url = self.config['constants']['fb_base_url']
        location = self.config['constants']['location']
        extracted_text_list = []
        urls_visited = set()

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.config['crawling']['headless'])
                context = browser.new_context()
                page = context.new_page()

                if not self.login_to_facebook(page):
                    return []

                for keyword in keywords:
                    search_url = f"{base_url} {location} {keyword}"
                    event_links = self.extract_event_links(page, search_url)
                    logging.info(f"def scrape_events: Used {search_url} to get {len(event_links)} event_links\n")

                    for link in event_links:
                        if link in urls_visited:
                            continue

                        extracted_text = self.extract_event_text(page, link)
                        extracted_text_list.append((link, extracted_text))
                        urls_visited.add(link)

                        # Get second-level links
                        second_level_links = self.extract_event_links(page, link)
                        for second_level_link in second_level_links:
                            if second_level_link in urls_visited:
                                continue # Skip the rest of the loop for this link

                            second_level_text = self.extract_event_text(page, second_level_link)
                            extracted_text_list.append((second_level_link, second_level_text))
                            urls_visited.add(second_level_link)

                browser.close()

        except Exception as e:
            logging.error(f"Failed to scrape events: {e}")

        logging.info(f"def scrape_events(): Extracted text from {len(extracted_text_list)} events.")

        return search_url, extracted_text_list
    

    def extract_text_from_fb_url(self, url):
        """
        Extracts text content from a Facebook event URL using Playwright and BeautifulSoup.

        Args:
            url (str): The Facebook event URL.

        Returns:
            str: Extracted text content from the Facebook event page.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.config['crawling']['headless'])
            context = browser.new_context()
            page = context.new_page()

            fb_status = fb_scraper.login_to_facebook(page)

            if fb_status:
                logging.info("def extract_text_from_fb_url(): Successfully logged into Facebook.")
                extracted_text = fb_scraper.extract_event_text(page, url)

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


    def driver(self):
        """
        Queries the database for events without URLs, scrapes the web for URLs and event data, processes the data using an LLM, 
        and updates the database with the new information.

        Steps:
        1. Queries the database to get all events that do not have a URL in their 'url' column.
        2. For each event, performs a Google search using `seu_handler.scrape_and_process(query)` to get the URL and extracted text.
        3. Generates a prompt and queries the LLM to process the extracted text.
        4. Parses the LLM response to extract event data.
        5. Updates the event data with the URL and saves it to the database using `db_handler.write_events_to_db`.
        6. Deduplicates the data in the database.

        Returns:
            None
        """
        query = "SELECT * FROM events WHERE url = %s"
        params = ('',)
        no_urls_df = pd.read_sql(query, db_handler.get_db_connection(), params=params)
        logging.info(f"def driver(): Retrieved {len(no_urls_df)} events without URLs.")

        # Reduce the number of events to process for testing
        no_urls_df = no_urls_df.head(20)

        for _, row in no_urls_df.iterrows():
            query = row['event_name']
            url, extracted_text, prompt_type = self.scrape_and_process(query)

            if extracted_text:
                # Generate prompt, query LLM, and process the response.
                prompt = llm_handler.generate_prompt(url, extracted_text, prompt_type)
                llm_response = llm_handler.query_llm(prompt, url)

                if "No events found" in llm_response:
                    # Delete the events and urls from the events and urls tables where appropriate
                    db_handler.delete_event_and_url(url, row['event_name'], row['start_date'])

                else:
                    parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
                    events_df = pd.DataFrame(parsed_result)
                    logging.info(f"def driver(): URL is: {url}")
                    events_df.to_csv('before_url_updated.csv', index=False)

                    if events_df['url'].values[0] == '':
                        events_df.loc[0, 'url'] = url

                        events_df.to_csv('after_url_updated.csv', index=False)

                    db_handler.write_events_to_db(events_df, url)

        return None


if __name__ == "__main__":
    # Initialize shared dependencies
    config_path = "config/config.yaml"

    # Instantiate the class libraries
    event_spider = EventSpider(config_path=config_path)
    fb_scraper = FacebookEventScraper(config_path=config_path)
    db_handler = DatabaseHandler(fb_scraper.config)
    llm_handler = LLMHandler(config_path=config_path)

    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")
    fb_scraper.driver()
