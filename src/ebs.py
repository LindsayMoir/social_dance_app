import asyncio
import logging
from datetime import datetime
import pandas as pd
import re
import sys
import yaml

from db import DatabaseHandler
from rd_ext import ReadExtract
from llm import LLMHandler


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

        # Prompt user for run name and description
        self.run_name = input("Enter run name: ").strip()
        self.run_description = input("Enter run description: ").strip()

        # Run statistics tracking
        self.start_time = None
        self.end_time = None
        self.urls_contacted = 0
        self.urls_with_extracted_text = 0
        self.urls_with_found_keywords = 0
        self.events_written_to_db = 0

    async def eventbrite_search(self, query, source, keywords_list, prompt):
        """ Searches Eventbrite for events based on the query, extracts event URLs,
        retrieves event details, and processes them using LLM.

        Parameters:
            query (str): The search query to enter in Eventbrite.
            source (str): The organization name related to the events.
            keywords_list (list): List of keywords associated with the events.
            prompt (str): The prompt to use for processing the extracted text. 
        """
        try:
            await self.read_extract.page.goto("https://www.eventbrite.com/", timeout=20000)
            logging.info(f"def eventbrite_search(): Navigated to Eventbrite.")

            await self.perform_search(query)
            event_urls = await self.extract_event_urls()
            logging.info(f"def eventbrite_search(): Total unique event URLs found: {len(event_urls)}")

            counter = 0
            for event_url in event_urls:
                if event_url in self.visited_urls:
                    logging.info(f"def eventbrite_search(): URL already visited: {event_url}")
                    continue

                if len(self.visited_urls) >= self.config['crawling']['urls_run_limit']:
                    logging.info(f"def eventbrite_search(): Reached the URL limit.")
                    sys.exit()

                elif counter >= self.config['crawling']['max_website_urls']:
                    logging.info(f"def eventbrite_search(): Reached the maximum limit of {self.config['crawling']['max_website_urls']} event URLs.")
                    break

                else:
                    logging.info(f"def eventbrite_search() Processing event URL: {event_url}")
                    self.visited_urls.add(event_url)
                    self.urls_contacted += 1  # Track URL contacts
                    counter += 1
                    await self.process_event(event_url, source, keywords_list, prompt, counter)

        except Exception as e:
            logging.error(f"def eventbrite_search(): Error during search: {e}")


    async def perform_search(self, query):
        """
        Performs a search on Eventbrite using the provided query.

        Args:
            query (str): Search query.
        """
        search_selector = "input#search-autocomplete-input"
        try:
            await self.read_extract.page.wait_for_selector(search_selector, timeout=20000)
            search_box = await self.read_extract.page.query_selector(search_selector)

            if search_box:
                await search_box.fill(query)
                await search_box.press("Enter")
                logging.info(f"def perform_search(): Performed search with query: {query}")
                await self.read_extract.page.wait_for_load_state("networkidle", timeout=15000)
            else:
                logging.error("def perform_search(): Search box not found on Eventbrite.")
                raise Exception("Search box not found.")
        except asyncio.TimeoutError:
            logging.error("def perform_search(): Timeout while performing search on Eventbrite.")
            raise


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
        

    async def process_event(self, event_url, source, keywords_list, prompt, counter):
        """ Processes an individual event URL: extracts text, processes it with LLM,
        and writes to the database.

        Args:
            event_url (str): Event URL.
            source (str): Organization name.
            keywords_list (list): List of keywords.
            prompt (str): Prompt for LLM processing.
        """
        try:
            extracted_text = await self.read_extract.extract_event_text(event_url)

            if extracted_text:
                self.urls_with_extracted_text += 1  # Count extracted text URLs

                # Check for keywords in the extracted text
                found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
                if found_keywords:
                    self.urls_with_found_keywords += 1  # Count URLs with found keywords
                    logging.info(f"def driver(): Found keywords in text for URL {event_url}: {found_keywords}")

                    # Process the extracted text with LLM
                    response = self.llm_handler.process_llm_response(event_url, extracted_text, source, keywords_list, prompt)

                    if response:
                        self.events_written_to_db += 1  # Count events written to the database
                else:
                    logging.info(f"def process_event(): No keywords found in text for: {event_url}")
            else:
                logging.warning(f"def process_event(): No extracted text for event: {event_url}")

        except Exception as e:
            logging.error(f"def process_event(): Error processing event {event_url}: {e}")

        finally:
            if self.read_extract.page and counter < self.config['crawling']['max_website_urls']:
                await self.read_extract.page.close()
                logging.info(f"def process_event(): Closed page for URL: {event_url}")
            else:
                logging.info(f"def process_event(): Page not closed for URL: {event_url}\n"
                             f"We are going onto the next keyword and that search query needs a page open")


    async def driver(self):
        """ Reads keywords, performs searches, and processes extracted event URLs. """
        self.start_time = datetime.now()  # Record start time
        self.keywords_list = ['salsa']  # Remember to remove this after testing
        for keyword in self.keywords_list:
            query = keyword
            source = ''
            prompt = 'default'
            logging.info(f"driver(): Searching for query: {query}")
            await self.eventbrite_search(query, source, self.keywords_list, prompt)

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

    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    db_handler = DatabaseHandler(config)
    llm_handler = LLMHandler(config_path='config/config.yaml')

    read_extract = ReadExtract(config_path='config/config.yaml')
    await read_extract.init_browser()

    ebs_instance = EventbriteScraper(
        config=config,
        read_extract=read_extract,
        db_handler=db_handler,
        llm_handler=llm_handler
    )

    await ebs_instance.driver()
    await read_extract.close()

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

if __name__ == "__main__":
    asyncio.run(main())
