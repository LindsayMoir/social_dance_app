# ebs.py

import logging
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright
import re
import yaml

from db import DatabaseHandler
from gs import GoogleSearch
from rd_ext import ReadExtract
from llm import LLMHandler


class EventbriteScraper():
    def __init__(self, config):
        """
        Initializes the EventbriteScraper with necessary handlers.

        Args:
            db_handler (DatabaseHandler): Instance to handle database operations.
        """
        self.config = config


    def eventbrite_search(self, query, org_name, keywords_list, prompt):
        """
        Searches Eventbrite for events based on the query, extracts event URLs,
        retrieves event details, and processes them using LLM.

        Parameters:
            query (str): The search query to enter in Eventbrite.
            org_name (str): The organization name related to the events.
            keywords_list (list): List of keywords associated with the events.
            prompt (str): The prompt to use for processing the extracted text.
        """
        with sync_playwright() as p:
            browser = self.launch_browser(p)
            context = browser.new_context()
            page = context.new_page()

            try:
                self.navigate_to_eventbrite(page)
                self.perform_search(page, query)
                event_urls = self.extract_event_urls(page)

                logging.info(f"Total unique event URLs found: {len(event_urls)}")

                for event_url in event_urls:
                    logging.info(f"Processing event URL: {event_url}")
                    self.process_event(event_url, org_name, keywords_list, prompt)

            except Exception as e:
                logging.error(f"An error occurred during Eventbrite search: {e}")
            finally:
                self.close_browser(page, context, browser)


    def launch_browser(self, playwright):
        """
        Launches the Playwright browser in headless mode.

        Args:
            playwright: Playwright instance.

        Returns:
            Browser instance.
        """
        return playwright.chromium.launch(headless=self.config['crawling']['headless'])
    

    def navigate_to_eventbrite(self, page):
        """
        Navigates to the Eventbrite homepage.

        Args:
            page: Playwright page instance.
        """
        page.goto("https://www.eventbrite.com/")
        logging.info("Navigated to Eventbrite.")


    def perform_search(self, page, query):
        """
        Performs a search on Eventbrite using the provided query.

        Args:
            page: Playwright page instance.
            query (str): Search query.
        """
        search_selector = "input#search-autocomplete-input"
        page.wait_for_selector(search_selector, timeout=10000)
        search_box = page.query_selector(search_selector)

        if search_box:
            search_box.fill(query)
            search_box.press("Enter")
            logging.info(f"Performed search with query: {query}")
            page.wait_for_load_state("networkidle", timeout=15000)
        else:
            logging.error("Search box not found on Eventbrite.")
            raise Exception("Search box not found.")
        

    def extract_event_urls(self, page):
        """
        Extracts unique event URLs from the search results.

        Args:
            page: Playwright page instance.

        Returns:
            set: A set of unique event URLs.
        """
        event_urls = set()
        event_links = page.query_selector_all("a[href*='/e/']")

        for link in event_links:
            href = link.get_attribute("href")
            if href:
                href = self.ensure_absolute_url(href)
                unique_id = self.extract_unique_id(href)
                if unique_id:
                    event_urls.add(href)
                    logging.debug(f"Found event URL: {href} with ID: {unique_id}")
                else:
                    logging.debug(f"URL does not match the expected pattern: {href}")

        return event_urls
    

    def ensure_absolute_url(self, href):
        """
        Ensures that the URL is absolute.

        Args:
            href (str): URL string.

        Returns:
            str: Absolute URL.
        """
        if not href.startswith("http"):
            href = f"https://www.eventbrite.com{href}"
        return href
    

    def extract_unique_id(self, url):
        """
        Extracts the unique identifier from the Eventbrite URL.

        Args:
            url (str): Eventbrite event URL.

        Returns:
            str or None: Unique identifier if pattern matches, else None.
        """
        pattern = r'/e/[^/]+-tickets-(\d+)(?:\?|$)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    

    def process_event(self, event_url, org_name, keywords_list, prompt):
        """
        Processes an individual event URL: extracts text, processes it with LLM,
        and writes to the database.

        Args:
            event_url (str): Event URL.
            org_name (str): Organization name.
            keywords_list (list): List of keywords.
            prompt (str): Prompt for LLM processing.
        """
        extracted_text = read_extract.extract_text_with_playwright(event_url)

        if extracted_text:
            success = llm_handler.process_llm_response(
                url=event_url,
                extracted_text=extracted_text,
                org_name=org_name,
                keywords_list=keywords_list,
                prompt=prompt
            )

            if success:
                logging.info(f"Successfully processed and wrote event: {event_url}")
                # Insert or update the URL in the database
                db_handler.write_url_to_db(
                    org_names=org_name,
                    keywords=keywords_list,
                    url=event_url,
                    other_links='',  # Populate as needed
                    relevant=True,
                    increment_crawl_trys=1
                )
            else:
                logging.warning(f"Failed to process event: {event_url}")
        else:
            logging.warning(f"No text extracted from event: {event_url}")

    def close_browser(self, page, context, browser):
        """
        Closes Playwright browser resources.

        Args:
            page: Playwright page instance.
            context: Playwright browser context.
            browser: Playwright browser instance.
        """
        page.close()
        context.close()
        browser.close()
        logging.info("Browser closed.")


    def driver(self):
        """
        Reads keywords from a DataFrame, constructs search queries,
        performs eventbrite searches for each query, and aggregates the results.

        Returns:
            pandas.DataFrame: A dataframe containing all search results.
        """
        all_results = []
        keywords_df = gs_instance.read_keywords()
        for _, row in keywords_df.iterrows():
            for keyword in row['keywords']:

                query = row['dance_style'] + '20%' + keyword
                org_name = ''
                keywords_list = row['keywords']
                prompt = 'default'
                
                results = self.eventbrite_search(query, org_name, keywords_list, prompt)

                if results:
                    all_results.extend(results)
                else:
                    logging.warning(f"def driver(): No results found for query: {query}")

        logging.info(f"def driver(): Driver completed with total {len(all_results)} results.")
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.config['output']['ebs_search_results'], index=False)

        return results_df


if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    db_handler = DatabaseHandler(config)
    ebs_instance = EventbriteScraper(config)
    gs_instance = GoogleSearch(config_path="config/config.yaml")
    llm_handler = LLMHandler(config_path='config/config.yaml')
    read_extract = ReadExtract(config_path='config/config.yaml') 

    # Start the driver
    results_df = ebs_instance.driver()

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")