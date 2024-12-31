import logging
import os
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import re
import yaml
import requests

# Import DatabaseHandler class
from db import DatabaseHandler
from llm import LLMHandler
from scraper import EventSpider
from srh_ext_upd import SearchExtractUpdate

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
        appid_uid, key_pw, access_token = keys_df.iloc[0][['appid_uid', 'key_pw', 'access_token']]
        logging.info(f"def get_credentials(): Retrieved credentials for {organization}.")
        return appid_uid, key_pw, access_token

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

                    print('\n************************************')
                    print(f"def scrape_events: Used {search_url} to get {len(event_links)} event_links\n")

                    logging.info(f"def scrape_events: Used {search_url} to get events")

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
        #no_urls_df = no_urls_df.head(3)

        for idx, row in no_urls_df.iterrows():
            query = row['event_name']
            url, extracted_text = seu_handler.scrape_and_process(query)

            if extracted_text:
                # Generate prompt, query LLM, and process the response.
                prompt = llm_handler.generate_prompt(url, extracted_text)
                llm_response = llm_handler.query_llm(prompt, url)

                if "No events found." in llm_response:
                    # Delete the events and urls from the events and urls tables where appropriate
                    db_handler.delete_event_and_url(url, row['event_name'], row['start_date'])

                else:
                    parsed_result = llm_handler.extract_and_parse_json(llm_response, url)
                    events_df = pd.DataFrame(parsed_result)
                    if row['url'] == '':
                        events_df.loc[idx, 'url'] = url
                    db_handler.write_events_to_db(events_df, url)

        #db_handler.dedup()


# Example Usage
if __name__ == "__main__":
    # Initialize shared dependencies
    config_path = "config/config.yaml"

    # Instantiate the class libraries
    event_spider = EventSpider(config_path=config_path)
    fb_scraper = FacebookEventScraper(config_path=config_path)
    db_handler = DatabaseHandler(fb_scraper.config)
    llm_handler = LLMHandler(config_path=config_path)
    seu_handler = SearchExtractUpdate(config_path=config_path)

    # Use the scraper
    logging.info(f"def __main__: Starting Facebook event scraping.")
    fb_scraper.driver()
