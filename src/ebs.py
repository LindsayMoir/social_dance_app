# ebs.py

"""
Eventbrite Scraper Script
This script defines the EventbriteScraper class and its associated methods to scrape event data from Eventbrite,
process the extracted text using a language model, and store the results in a database. The script also includes
a main function to initialize and run the scraper.

Classes:
    EventbriteScraper: Handles the scraping, processing, and storing of Eventbrite event data.

Functions:
    main(): Initializes and runs the Eventbrite scraper.

Usage:
    Run this script directly to start the Eventbrite scraping process.

Example:
    $ python ebs.py

Dependencies:
    - asyncio
    - logging
    - datetime
    - pandas
    - re
    - sys
    - yaml
    - db (DatabaseHandler)
    - rd_ext (ReadExtract)
    - llm (LLMHandler)

Configuration:
    The script reads configuration settings from 'config/config.yaml'.

Logging:
    Logs are written to the file specified in the configuration under 'logging.log_file'.

Inputs:
    - config/config.yaml: Configuration file containing settings for the scraper.
    - keyword.csv file from llm_handler.get_keywords() method.

Outputs:
    - Logs: Detailed logs of the scraping process.
    - Database: Run statistics and event data stored in the database.
"""
import asyncio
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import pandas as pd
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
import random
import re
import sys
import yaml

from db import DatabaseHandler
from llm import LLMHandler
from rd_ext import ReadExtract


class EventbriteScraper:
    def __init__(self, config, read_extract, db_handler, llm_handler):
        """
        Initializes the EventbriteScraper with necessary handlers.

        Args:
            config (dict): Configuration dictionary.
            read_extract (ReadExtract): Instance of ReadExtract for text extraction.
            db_handler (DatabaseHandler): Instance to handle database operations.
            llm_handler (LLMHandler): Instance to handle LLM processing.
        """
        self.config = config
        self.read_extract = read_extract
        self.db_handler = db_handler
        self.llm_handler = llm_handler
        self.visited_urls = set()
        self.keywords_list = llm_handler.get_keywords()

        # Run statistics tracking
        if config['testing']['status']:
            self.run_name = f"Test Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self.run_description = "Test Run Description"
        else:
            self.run_name = "ebs Run"
            self.run_description = f"Production {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.start_time = None
        self.end_time = None
        self.urls_contacted = 0
        self.urls_with_extracted_text = 0
        self.urls_with_found_keywords = 0
        self.events_written_to_db = 0


    async def eventbrite_search(self, query, source, keywords_list, prompt):
        """
        Searches Eventbrite for events matching the given query, filters and processes event URLs using LLM and database checks, and processes relevant events.

        Args:
            query (str): The search query to use on Eventbrite.
            source (str): The source identifier for the search.
            keywords_list (list): List of keywords to help with event relevance.
            prompt (str): The initial prompt or context for LLM processing.

        Workflow:
            1. Navigates to the Eventbrite homepage.
            2. Performs a search using the provided query.
            3. Extracts event URLs from the search results.
            4. For each unique event URL:
                - Skips URLs already visited or exceeding configured limits.
                - Uses LLM to determine if the URL is likely relevant.
                - Checks historical relevancy via the database handler.
                - If relevant and not previously processed, processes the event details.

        Returns:
            None

        Notes:
            - Uses LLM to filter URLs for relevance before processing.
            - Respects crawling limits set in the configuration.
            - Logs progress and errors throughout the process.
        """
        # Ensure event_urls is always defined
        event_urls = []

        # Navigate to Eventbrite (wait only for DOM, with longer timeout)
        to = random.randint(6000//2, int(6000 * 1.5))
        try:
            await self.read_extract.page.goto(
                "https://www.eventbrite.com/",
                wait_until="domcontentloaded",
                timeout=to
            )
            logging.info("def eventbrite_search(): Navigated to Eventbrite.")
        except Exception as e:
            logging.error(f"def eventbrite_search(): Error navigating to Eventbrite: {e}")
            return  # cannot proceed without homepage

        # Perform the search and extract URLs
        try:
            await self.perform_search(query)
            event_urls = await self.extract_event_urls()
            logging.info(f"def eventbrite_search(): Total unique event URLs found: {len(event_urls)}")
        except Exception as e:
            logging.error(f"def eventbrite_search(): Error during search/extraction: {e}")
            return  # nothing to process

        # Process each event URL
        counter = 0
        for event_url in event_urls:
            if event_url in self.visited_urls:
                logging.info(f"def eventbrite_search(): Already visited: {event_url}")
                continue

            if len(self.visited_urls) >= self.config['crawling']['urls_run_limit']:
                logging.info("def eventbrite_search(): Reached overall URL limit.")
                sys.exit()

            if counter >= self.config['crawling']['max_website_urls']:
                logging.info(
                    f"def eventbrite_search(): Reached max {self.config['crawling']['max_website_urls']} event URLs."
                )
                break

           # Check if the words in the url make it likely to be relevant
            prompt_type = 'relevant_dance_url'
            prompt = self.llm_handler.generate_prompt(event_url, event_url, prompt_type)

            # 1) Get the raw LLM output
            raw = self.llm_handler.query_llm(event_url, prompt)
            logging.info(f"def eventbrite_search(): Raw LLM output for {event_url} → {repr(raw)}")

            # 2) Convert to a proper boolean
            if isinstance(raw, str):
                is_relevant = raw.strip().lower() == "true"
            else:
                is_relevant = bool(raw)

            # 3) Act accordingly
            if not is_relevant:
                logging.info(f"def eventbrite_search(): Skipping URL {event_url} based on LLM response.")
                continue

            logging.info(f"def eventbrite_search(): LLM response indicates URL {event_url} is relevant.")

            # Check urls to see if they should be scraped
            if not self.db_handler.should_process_url(event_url):
                logging.info(f"def eventbrite_search(): Skipping URL {event_url} based on historical relevancy.")
                continue

            logging.info(f"def eventbrite_search(): Processing URL {event_url}")
            self.visited_urls.add(event_url)
            self.urls_contacted += 1
            counter += 1
            parent_url = query
            await self.process_event(event_url, parent_url, source, keywords_list, prompt, counter)


    async def perform_search(self, query):
        """
        Performs a search on Eventbrite using the provided query, retrying up to 3 times on failure.
        Logs and returns if all attempts fail, allowing the caller to continue.
        """
        search_selector = "input#search-autocomplete-input"
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                to = random.randint(20000 // 2, int(20000 * 1.5))
                await self.read_extract.page.wait_for_selector(search_selector, timeout=to)
                search_box = await self.read_extract.page.query_selector(search_selector)

                if not search_box:
                    logging.error(f"def perform_search(): Search box not found (attempt {attempt}).")
                    raise Exception("Search box not found.")

                await search_box.fill(query)
                await search_box.press("Enter")
                logging.info(f"def perform_search(): Performed search '{query}' (attempt {attempt}).")

                to = random.randint(15000 // 2, int(15000 * 1.5))
                await self.read_extract.page.wait_for_load_state("networkidle", timeout=to)
                return  # success

            except asyncio.TimeoutError:
                logging.warning(f"def perform_search(): Timeout on '{query}' (attempt {attempt}).")
            except Exception as e:
                logging.warning(f"def perform_search(): Error on '{query}' (attempt {attempt}): {e}")

            if attempt < max_retries:
                backoff = attempt * 2
                logging.warning(f"def perform_search(): Retrying '{query}' in {backoff}s ({attempt+1}/{max_retries})")
                await asyncio.sleep(backoff)

        # all attempts failed—log and return so caller can continue
        logging.error(f"def perform_search(): Giving up on '{query}' after {max_retries} attempts.")
        return  # or return False
    

    async def extract_event_urls(self):
        """ Extracts unique event URLs from the search results.

        Returns:
            set: A set of unique event URLs.
        """
        event_urls = set()
        try:
            event_link = await self.read_extract.page.query_selector_all("a[href*='/e/']")

            for link in event_link:
                href = await link.get_attribute("href")
                if href:
                    href = self.ensure_absolute_url(href)
                    unique_id = self.extract_unique_id(href)
                    if unique_id:
                        event_urls.add(href)
                        logging.debug(f"def extract_event_urls(): Found event URL: {href} with ID: {unique_id}")
                    else:
                        logging.debug(f"def extract_event_urls(): URL does not match the expected pattern: {href}")

                if len(self.visited_urls) >= self.config['crawling']['urls_run_limit']:
                    logging.info(f"def extract_event_urls(): Reached the maximum limit of {self.config['crawling']['max_website_urls']}"
                                 f" event URLs or {self.config['crawling']['urls_run_limit']} visited URLs."
                                 f"\nurls are: {event_urls}")

            return event_urls
        
        except Exception as e:
            logging.error(f"def extract_event_urls():  Error extracting event URLs: {e}")
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
        

    async def process_event(self, event_url, parent_url, source, keywords_list, prompt, counter):
        """Processes an individual event URL: extracts text, processes it with LLM,
        and writes to the database.
        
        Args:
            event_url (str): Event URL.
            source (str): Organization name.
            keywords_list (list): List of keywords.
            prompt (str): Prompt for LLM processing.
            counter (int): Counter for processed events.
        """
        extracted_text = await self.read_extract.extract_event_text(event_url)

        if extracted_text:
            self.urls_with_extracted_text += 1  # Count extracted text URLs

            # Check for keywords in the extracted text
            found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
            if found_keywords:
                self.urls_with_found_keywords += 1  # Count URLs with found keywords
                logging.info(f"def process_event(): Found keywords in text for URL {event_url}: {found_keywords}")

                # Process the extracted text with the LLM
                response = self.llm_handler.process_llm_response(event_url, parent_url, extracted_text, source, found_keywords, prompt)

                if response:
                    self.events_written_to_db += 1  # Count events written to the database
            else:
                logging.info(f"def process_event(): No keywords found in text for: {event_url}")
        else:
            logging.warning(f"def process_event(): No extracted text for event: {event_url}")


    async def driver(self):
        """ Reads keywords, performs searches, and processes extracted event URLs. """
        self.start_time = datetime.now()  # Record start time

        # Remove the output file if it exists
        output_path = self.config['output']['ebs_keywords_processed']
        if os.path.exists(output_path):
            os.remove(output_path)

        for keyword in self.keywords_list:
            query = keyword
            source = ''
            prompt = 'default'
            logging.info(f"driver(): Searching for query: {query}")
            await self.eventbrite_search(query, source, self.keywords_list, prompt)

            # Log the keyword and timestamp to a CSV file
            # 1) get the current timestamp as a string
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 2) build a one‐row DataFrame
            df_row = pd.DataFrame({
                "keyword":   [keyword],
                "timestamp": [ts]
            })

            # 3) append to CSV (write header only if file doesn't exist yet)
            write_header = not os.path.exists(output_path)
            df_row.to_csv(output_path, mode="a", header=write_header, index=False)
        
        self.end_time = datetime.now()  # Record end time
        logging.info(f"driver(): Completed processing {len(self.visited_urls)} unique URLs.")

        # Write run statistics to the database
        await self.write_run_statistics()


    async def write_run_statistics(self):
        """ Saves the run statistics to the database. """
        elapsed_time = str(self.end_time - self.start_time)  # Convert timedelta to a string
        python_file_name = __file__.split('/')[-1]

        # Create a DataFrame for the run statistics
        run_data = pd.DataFrame([{
            "run_name": self.run_name,
            "run_description": self.run_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": elapsed_time,  # Now stored as a string
            "python_file_name": python_file_name,
            "unique_urls_count": len(self.visited_urls),
            "total_url_attempts": self.urls_contacted,
            "urls_with_extracted_text": self.urls_with_extracted_text,
            "urls_with_found_keywords": self.urls_with_found_keywords,
            "events_written_to_db": self.events_written_to_db,
            "time_stamp": datetime.now()
        }])

        # Get the database connection engine
        engine = self.db_handler.get_db_connection()

        # Write the data to the "runs" table
        try:
            run_data.to_sql("runs", engine, if_exists="append", index=False)
            logging.info("write_run_statistics(): Run statistics written to database successfully.")
        except Exception as e:
            logging.error(f"write_run_statistics(): Error writing run statistics to database: {e}")


async def main():
    """ Main function to initialize and run the Eventbrite scraper. """
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Build log_file name
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging_file = f"logs/{script_name}_log.txt" 
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logging.info("\n\nebs.py starting...")

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Initialize the scraper components
    read_extract = ReadExtract(config_path='config/config.yaml')
    llm_handler = LLMHandler(config_path='config/config.yaml')
    db_handler = llm_handler.db_handler  # Use the DatabaseHandler from LLMHandler
    ebs_instance = EventbriteScraper(
        config=config,
        read_extract=read_extract,
        db_handler=db_handler,
        llm_handler=llm_handler
    )

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    # Start
    await read_extract.init_browser()
    await ebs_instance.driver()
    await read_extract.close()

    # Write the final event and url counts to the .csv
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

if __name__ == "__main__":
    asyncio.run(main())
