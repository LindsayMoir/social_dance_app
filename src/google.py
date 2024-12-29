from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import requests
import pandas as pd
import yaml
import logging
from llm import LLMHandler
from db import DatabaseHandler

class EventScraper:
    def __init__(self, config_path="config/config.yaml"):
        """
        Initializes the EventScraper class.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        # Load configuration from a YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        # Set up logging
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.db_handler = DatabaseHandler(self.config)
        self.llm_handler = LLMHandler(config_path)

        logging.info("EventScraper initialized.")

    def google_search(self, query):
        """
        Performs a Google Search using Playwright and extracts the results.

        Args:
            query (str): The search query.

        Returns:
            list: A list of tuples containing the title and URL of the search results.
        """
        search_results = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # Set headless=True for silent execution
            context = browser.new_context()
            page = context.new_page()

            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={query}"
            page.goto(search_url)

            # Wait for results to load
            page.wait_for_selector("h3")

            # Extract titles and URLs from search results
            results = page.query_selector_all("div.yuRUbf a")
            for result in results:
                url = result.get_attribute("href")
                title_element = result.query_selector("h3")
                title = title_element.inner_text() if title_element else "No title"
                search_results.append((title, url))

            browser.close()

        logging.info(f"Google search completed for query: {query}. Found {len(search_results)} results.")
        return search_results

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

    def process_and_update(self, extracted_text, query, url):
        """
        Processes extracted text using LLM and updates the database.

        Args:
            extracted_text (str): The text extracted from a webpage.
            query (str): The event name query.
            url (str): The source URL of the event.

        Returns:
            None
        """
        logging.info(f"Processing extracted text for URL: {url} with query: {query}.")
        self.llm_handler.driver(self.db_handler, url, query, extracted_text, [query])

    def scrape_and_process(self, query):
        """
        Scrapes Google Search results, processes them, and updates the database.

        Args:
            query (str): The search query.

        Returns:
            None
        """
        logging.info(f"Starting scrape and process for query: {query}.")
        results = self.google_search(query)

        for title, url in results:
            if query.lower() in title.lower():
                logging.info(f"Relevant result found: Title: {title}, URL: {url}.")

                # Convert Facebook URL if applicable
                if 'facebook' in url:
                    url = self.convert_facebook_url(url)

                # Scrape text from the non-Facebook URL
                extracted_text = self.extract_text_from_url(url)
                if extracted_text:
                    # Process the extracted text and update the database
                    self.process_and_update(extracted_text, query, url)

if __name__ == "__main__":
    # Replace with actual configuration path
    config_path = "config/config.yaml"

    scraper = EventScraper(config_path)
    query = "Sundown Social: Dance with Cupid"
    scraper.scrape_and_process(query)
