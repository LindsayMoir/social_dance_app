# ebs.py

import asyncio
import logging
from datetime import datetime
import pandas as pd
import re
import yaml

from db import DatabaseHandler
from gs import GoogleSearch
from rd_ext import ReadExtract
from llm import LLMHandler


class EventbriteScraper:
    def __init__(self, config, read_extract, db_handler, gs_instance, llm_handler):
        """
        Initializes the EventbriteScraper with necessary handlers.

        Args:
            config (dict): Configuration dictionary.
            read_extract (ReadExtract): Instance of ReadExtract for text extraction.
            db_handler (DatabaseHandler): Instance to handle database operations.
            gs_instance (GoogleSearch): Instance to handle Google searches.
            llm_handler (LLMHandler): Instance to handle LLM processing.
        """
        self.config = config
        self.read_extract = read_extract
        self.db_handler = db_handler
        self.gs_instance = gs_instance
        self.llm_handler = llm_handler
        self.visited_urls = set()

    async def eventbrite_search(self, query, org_name, keywords_list, prompt):
        """
        Searches Eventbrite for events based on the query, extracts event URLs,
        retrieves event details, and processes them using LLM.

        Parameters:
            query (str): The search query to enter in Eventbrite.
            org_name (str): The organization name related to the events.
            keywords_list (list): List of keywords associated with the events.
            prompt (str): The prompt to use for processing the extracted text.
        """
        try:
            # Navigate to Eventbrite homepage
            await self.read_extract.page.goto("https://www.eventbrite.com/", timeout=10000)
            logging.info("Navigated to Eventbrite.")

            # Perform search
            await self.perform_search(query)

            # Extract event URLs
            event_urls = await self.extract_event_urls()

            logging.info(f"Total unique event URLs found: {len(event_urls)}")

            for event_url in event_urls:
                if event_url in self.visited_urls:
                    logging.info(f"URL already visited: {event_url}")
                    continue

                if len(self.visited_urls) >= self.config['crawling']['urls_run_limit']:
                    logging.info(f"Reached the maximum limit of {self.config['crawling']['urls_run_limit']} URLs.")
                    break

                logging.info(f"Processing event URL: {event_url}")
                self.visited_urls.add(event_url)
                await self.process_event(event_url, org_name, keywords_list, prompt)

        except Exception as e:
            logging.error(f"An error occurred during Eventbrite search: {e}")

    async def perform_search(self, query):
        """
        Performs a search on Eventbrite using the provided query.

        Args:
            query (str): Search query.
        """
        search_selector = "input#search-autocomplete-input"
        try:
            await self.read_extract.page.wait_for_selector(search_selector, timeout=10000)
            search_box = await self.read_extract.page.query_selector(search_selector)

            if search_box:
                await search_box.fill(query)
                await search_box.press("Enter")
                logging.info(f"Performed search with query: {query}")
                await self.read_extract.page.wait_for_load_state("networkidle", timeout=15000)
            else:
                logging.error("Search box not found on Eventbrite.")
                raise Exception("Search box not found.")
        except asyncio.TimeoutError:
            logging.error("Timeout while performing search on Eventbrite.")
            raise

    async def extract_event_urls(self):
        """
        Extracts unique event URLs from the search results.

        Returns:
            set: A set of unique event URLs.
        """
        event_urls = set()
        try:
            event_links = await self.read_extract.page.query_selector_all("a[href*='/e/']")

            for link in event_links:
                href = await link.get_attribute("href")
                if href:
                    href = self.ensure_absolute_url(href)
                    unique_id = self.extract_unique_id(href)
                    if unique_id:
                        event_urls.add(href)
                        logging.debug(f"Found event URL: {href} with ID: {unique_id}")
                    else:
                        logging.debug(f"URL does not match the expected pattern: {href}")

                if len(event_urls) >= self.config['crawling']['max_website_urls'] or len(self.visited_urls) >= self.config['crawling']['urls_run_limit']:
                    logging.info(f"Reached the maximum limit of {self.config['crawling']['max_website_urls']} event_urls or {self.config['crawling']['urls_run_limit']} visited URLs.")
                    break

            return event_urls
        except Exception as e:
            logging.error(f"Error extracting event URLs: {e}")
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

    async def process_event(self, event_url, org_name, keywords_list, prompt):
        """
        Processes an individual event URL: extracts text, processes it with LLM,
        and writes to the database.

        Args:
            event_url (str): Event URL.
            org_name (str): Organization name.
            keywords_list (list): List of keywords.
            prompt (str): Prompt for LLM processing.
        """
        try:
            extracted_text = await self.read_extract.extract_event_text(event_url)

            if extracted_text:
                success = self.llm_handler.process_llm_response(
                    url=event_url,
                    extracted_text=extracted_text,
                    org_name=org_name,
                    keywords_list=keywords_list,
                    prompt=prompt
                )

                if success:
                    logging.info(f"Successfully processed and wrote event: {event_url}")
                    # Insert or update the URL in the database
                    self.db_handler.write_url_to_db(
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
        except Exception as e:
            logging.error(f"Error processing event {event_url}: {e}")

    async def driver(self):
        """
        Reads keywords from a DataFrame, constructs search queries,
        performs Eventbrite searches for each query, and aggregates the results.

        Returns:
            pandas.DataFrame: A dataframe containing all search results.
        """
        all_results = []
        try:
            keywords_df = self.gs_instance.read_keywords()
            for _, row in keywords_df.iterrows():
                # Split the keywords and search for each keyword
                keywords = row['keywords'].split(',')
                for keyword in keywords:
                    query = keyword.strip()
                    org_name = ''  # Populate as needed
                    keywords_list = [kw.strip() for kw in row['keywords'].split(',')]
                    prompt = 'default'

                    # Perform the search
                    logging.info(f"driver(): Searching for query: {query}")
                    await self.eventbrite_search(query, org_name, keywords_list, prompt)

                    # Optionally, collect results if needed
                    # Since processing and database writing is handled in process_event,
                    # you might not need to collect them here unless required.

            logging.info(f"driver(): Driver completed with total {len(self.visited_urls)} processed URLs.")

            # Since results are processed and written to the database,
            # you might not need to return a DataFrame unless required.
            # If you need to aggregate results, consider modifying process_event to return data.
            return None

        except Exception as e:
            logging.error(f"Error in driver: {e}")
            return None


async def main():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Initialize handlers
    db_handler = DatabaseHandler(config)
    gs_instance = GoogleSearch(config_path="config/config.yaml")
    llm_handler = LLMHandler(config_path='config/config.yaml')

    # Initialize ReadExtract and browser
    read_extract = ReadExtract(config_path='config/config.yaml')
    await read_extract.init_browser()

    # Initialize EventbriteScraper
    ebs_instance = EventbriteScraper(
        config=config,
        read_extract=read_extract,
        db_handler=db_handler,
        gs_instance=gs_instance,
        llm_handler=llm_handler
    )

    # Start the driver
    await ebs_instance.driver()

    # Close ReadExtract's browser
    await read_extract.close()

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")


if __name__ == "__main__":
    asyncio.run(main())
