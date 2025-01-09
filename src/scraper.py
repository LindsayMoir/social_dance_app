"""
This program scrapes regular NON facebook pages and calendar pages for events.
"""

import base64
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import json
import logging
from openai import OpenAI
import os
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import random
import re
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, update, MetaData, text
import sys
import time
import yaml

print(os.getcwd())

from db import DatabaseHandler
from llm import LLMHandler

# Load configuration from a YAML file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='w',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("global: Working directory is: %s", os.getcwd())

# EventSpider class
class EventSpider(scrapy.Spider):
    name = "event_spider"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_links = set()  # To track visited URLs and avoid duplicate crawls

    def start_requests(self):
        """
        Initiates the scraping process by generating requests from URLs obtained from a database or a CSV file.

        This method connects to a database to fetch URLs marked as relevant or reads them from a CSV file 
        based on the configuration. It then iterates over the URLs, marking them as visited and yielding 
        Scrapy requests for each URL to be processed by the `parse` method.

        Raises:
            ConnectionError: If the database connection fails.

        Yields:
            scrapy.Request: A Scrapy request object for each URL to be crawled, with additional callback 
                            arguments including keywords, organization name, and the URL itself.
        """
        # Connect to the database
        conn = db_handler.get_db_connection()

        if conn is None:
            raise ConnectionError("Failed to connect to the databasein start_requests.")
        else:
            print("Connected to the database in start_requests", conn)

        if config['startup']['use_db']:
            # Read the URLs from the database
            query = "SELECT * FROM urls WHERE relevant = true;"
            urls_df = pd.read_sql_query(query, conn)
        else:
            # Load initial URLs from the 'urls.csv' file
            urls_df = pd.read_csv(config['input']['data_urls'])

        for _, row in urls_df.iterrows():
            org_name = row['org_names']
            keywords = row['keywords']
            url = row['links']

            if 'facebook' in url:
                db_handler.write_url_to_fb_table(url)
                logging.info(f"Skipping crawl for Facebook URL: {url}")
            else:
                # We need to write the url to the database, if it is not already there
                other_links, relevant, increment_crawl_trys = '', True, 1
                db_handler.write_url_to_db(org_name, keywords, url, other_links, relevant, increment_crawl_trys)

                if url not in self.visited_links:
                    self.visited_links.add(url)  # Mark the page link as visited
                    logging.info(f"Starting crawl for URL: {url}")
                    yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'keywords': keywords, 
                                                    'org_name': org_name, 
                                                    'url': url})

    def parse(self, response, keywords, org_name, url):
        """
        Args:
            response (scrapy.http.Response): The response object to parse.
            keywords (str): A comma separated string of keywords to check for relevance.
            org_name (str): The name of the organization to check for relevance.
            url (str): The URL of the current page being parsed.

        Returns:
            generator: A generator yielding scrapy.Request objects for further crawling.

        This function extracts links from the main page and handles Facebook links differently. 
        It also extracts iframe sources and updates the URL if relevant. 
        The function checks for relevance of the links and continues crawling if they are relevant.
        """
        # Get all of the subsidiary links on the page   
        page_links = response.css('a::attr(href)').getall()
        page_links = [response.urljoin(link) for link in page_links if link.startswith('http')]
        logging.info(f"def parse(): Found {len(page_links)} links on {response.url}")
        page_links = page_links[:config['crawling']['max_website_urls']]  # Limit the number of links
        logging.info(f"def parse(): Limiting the number of links to {config['crawling']['max_website_urls']}.")

        # Extract iframe sources
        iframe_links = response.css('iframe::attr(src)').getall()
        iframe_links = [response.urljoin(link) for link in iframe_links if link.startswith('http')]

        if iframe_links:
            db_handler.update_url(url, update_other_links='calendar', relevant=True, increment_crawl_trys=0)
            for calendar_url in iframe_links:
                self.fetch_google_calendar_events(calendar_url, url, org_name, keywords)

        # Put all links together
        all_links = set(page_links + [url])

        # identify any current or future facebook links. We want those out and processed by fb.py
        # Iterate over a copy of all_links to safely remove items during iteration
        for link in list(all_links):
            if 'facebook' in link or 'instagram' in link:
                db_handler.write_url_to_fb_table(link)
                all_links.remove(link)  # Safely remove from the original set
                logging.info(f"def parse(): Found a Facebook or Instagram URL, processing: {link}")
        
        logging.info(f"def parse() Found {len(all_links)} links on {response.url}")

        # Check for relevance and crawl further
        for link in all_links:
            if link not in self.visited_links:
                self.visited_links.add(link)  # Mark the page link as visited
                other_links, relevant, increment_crawl_trys = '', True, 1
                db_handler.write_url_to_db(org_name, keywords, url, other_links, relevant, increment_crawl_trys)
                if len(self.visited_links) >= config['crawling']['urls_run_limit']:
                    logging.info(f"def parse(): Maximum URL limit reached: {config['crawling']['urls_run_limit']} Stopping further crawling.")
                    break
                
                self.driver(link, keywords, org_name)
                logging.info(f"def parse() Starting crawl for URL: {link}")
                yield response.follow(url=link, callback=self.parse, cb_kwargs={'keywords': keywords, 
                                                                                'org_name': org_name, 
                                                                                'url': link})
    
    
    def driver(self, url, keywords, org_name):
        """
        Determine the relevance of a given URL based on its content, keywords, or organization name.

        Parameters:
        url (str): The URL to be evaluated.
        keywords (str): A comma separated string of keywords to check within the URL content.
        org_name (str): The name of the organization to check within the URL content.

        Returns:
        bool: True if the URL is relevant, False otherwise.
        """
        if 'facebook' in url:
            db_handler.write_url_to_fb_table(url)
            logging.debug(f"def driver(): URL {url} is a Facebook link and should never happen.")
            return
        
        # Process non-facebook links
        extracted_text = self.extract_text_with_playwright(url)

        # Check keywords in the extracted text
        keyword_status = self.check_keywords_in_text(url, extracted_text, keywords, org_name)

        if keyword_status:
            # Call the llm to process the extracted text
            llm_status = llm_handler.process_llm_response(url, extracted_text, org_name, keywords)

            if llm_status:
                # Mark the event link as relevant
                db_handler.update_url(url, update_other_links='', relevant=True, increment_crawl_trys=0)

            else:
                # Mark the event link as irrelevant
                db_handler.update_url(url, update_other_links='', relevant=False, increment_crawl_trys=0) 

        else:
            logging.info(f"def parse(): URL {url} marked as irrelevant since there are no keywords.")
            db_handler.update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)

        return

    def check_keywords_in_text(self, url, extracted_text, keywords, org_name):
        """
        Parameters:
        url (str): The URL of the webpage being checked.
        extracted_text (str): The text extracted from the webpage.
        keywords (str, optional): A comma-separated string of keywords to check in the extracted text. Defaults to None.
        org_name (str, optional): The name of the organization, used for further processing if keywords are found. Defaults to None.

        Returns:
        bool: True if the text is relevant based on the presence of keywords or 'calendar' in the URL, False otherwise.
        """
        # Check for keywords in the extracted text and determine relevance.
        if keywords:
            keywords_list = [kw.strip().lower() for kw in keywords.split(',')]
            if any(kw in extracted_text.lower() for kw in keywords_list):
                logging.info(f"def check_keywords_in_text: Keywords found in extracted text for URL: {url}")
                return llm_handler.process_llm_response(url, extracted_text, org_name, keywords_list)

        if 'calendar' in url:
            logging.info(f"def check_keywords_in_text: URL {url} marked as relevant because 'calendar' is in the URL.")
            return True

        logging.info(f"def check_keywords_in_text: URL {url} marked as irrelevant since there are no keywords, events, or 'calendar' in URL.")
        return False

    def extract_and_parse_json(self, result, url):
        """
        Parameters:
        result (str): The response string from which JSON needs to be extracted.
        url (str): The URL from which the response was obtained.
        Returns:
        list or None: Returns a list of events if JSON is successfully extracted and parsed, 
                      otherwise returns None.
        """
        if "No events found" in result:
            logging.info("def extract_and_parse_json(): No events found in result.")
            return None
        
        # Check if the response contains JSON
        if 'json' in result:
            start_position = result.find('[')
            end_position = result.rfind(']') + 1
            if start_position != -1 and end_position != -1:
                # Extract JSON part
                json_string = result[start_position:end_position]

                try:
                    # Convert JSON string to Python object
                    logging.info("def extract_and_parse_json(): JSON found in result.")
                    events_json =json.loads(json_string)
                    logging.info(f"def extract_and_parse_json(): For url: {url}, \nhere is the events_json: \n{events_json}.")
                    return events_json
                    
                except json.JSONDecodeError as e:
                    logging.error(f"def extract_and_parse_json(): Error parsing JSON: {e}")
                    return None

        logging.info("def extract_and_parse_json(): No JSON found in result.")
        
        return None
    
    def extract_text_with_playwright(self, url):
        """
        Extracts the text content from a web page using Playwright.
        This method uses the Playwright library to load a web page and extract its visible text content.
        It launches a headless Chromium browser, navigates to the specified URL, waits for the page to load,
        and then retrieves the text content from the page.
        Args:
            url (str): The URL of the web page to extract text from.
        Returns:
            str: The extracted text content from the web page, or None if an error occurs.
        Raises:
            Exception: If there is an issue with loading the page or extracting the text.
        """
        #try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=config['crawling']['headless'])
            page = browser.new_page()
            page.goto(url, timeout=30000)  # Wait up to 30 seconds for the page to load
            page.wait_for_timeout(9000)  # Additional wait for dynamic content

            # Extract the visible text
            content = page.content()  # Get the full HTML content
            soup = BeautifulSoup(content, 'html.parser')
            extracted_text = ' '.join(soup.stripped_strings)

            browser.close()
            return extracted_text
            
        # except Exception as e:
        #     logging.error(f"def extract_text_with_playwright(): Failed to extract text with Playwright for URL {url}: {e}")
        #     return None

    def fetch_google_calendar_events(self, calendar_url, url, org_name, keywords):
        """
        Fetch events from a Google Calendar and process them.

        This function extracts calendar IDs from the provided Google Calendar URL,
        fetches events using these IDs, and processes the events by writing them
        to a database and updating URLs.

        Args:
            calendar_url (str): The URL of the Google Calendar to fetch events from.
            keywords (str): A comma separated list of keywords to associate with the events.
            org_name (str): The name of the organization associated with the events.
            url (str): The URL to update after processing the events.

        Returns:
            None
        """
        calendar_id_pattern = r'src=([^&]+%40group.calendar.google.com)'
        calendar_ids = re.findall(calendar_id_pattern, calendar_url)

        if calendar_ids:
            decoded_calendar_ids = [id.replace('%40', '@') for id in calendar_ids]
            logging.info(f"def fetch_google_calendar_events(): Found {len(calendar_ids)} group.calendar.google.com IDs: {decoded_calendar_ids}")

            for calendar_id in decoded_calendar_ids:
                events_df = self.get_events(calendar_id)
                if not events_df.empty:
                    db_handler.write_events_to_db(events_df, calendar_url, org_name, keywords)
                    db_handler.update_url(calendar_url, update_other_links=url, relevant=True, increment_crawl_trys=1)
                    db_handler.update_url(url, update_other_links=calendar_url, relevant=True, increment_crawl_trys=1)
        else:
            start_idx = calendar_url.find("src=") + 4
            end_idx = calendar_url.find("&", start_idx)
            calendar_id = calendar_url[start_idx:end_idx] if end_idx != -1 else calendar_url[start_idx:]
            
            # Wrap base64 decoding in try/except
            try:
                padded_calendar_id = calendar_id + '=' * (4 - len(calendar_id) % 4)
                decoded_bytes = base64.b64decode(padded_calendar_id)
                calendar_id = decoded_bytes.decode('utf-8', errors='replace')
            except (UnicodeDecodeError, base64.binascii.Error) as e:
                logging.error(
                    f"def fetch_google_calendar_events(): Failed to decode calendar_id: \n{calendar_id} \n"
                    f"for calendar_url: {calendar_url} \nand url: \n{url} with exception: \n{e}"
                )
                calendar_id = None

            # Proceed only if calendar_id was successfully decoded
            if calendar_id:
                events_df = self.get_events(calendar_id)
                if not events_df.empty:
                    db_handler.write_events_to_db(events_df, calendar_url, org_name, keywords)
                    db_handler.update_url(calendar_url, update_other_links=url, relevant=True, increment_crawl_trys=1)
                    db_handler.update_url(url, update_other_links=calendar_url, relevant=True, increment_crawl_trys=1)

        return

    def get_events(self, calendar_id):
        """
        Args:
            calendar_id (str): The ID of the Google Calendar from which to fetch events.

        Returns:
            pd.DataFrame: A DataFrame containing the events fetched from the Google Calendar.
                            If no events are found, an empty DataFrame is returned.

        Raises:
            requests.exceptions.RequestException: If there is an issue with the HTTP request.

        Notes:
            - The function reads the API key from a CSV file specified in the configuration.
            - The date range for fetching events is determined by the 'days_ahead' value in the configuration.
            - The function handles pagination to fetch all events if there are multiple pages of results.
            - If an error occurs during the HTTP request, it logs the error and stops fetching events.
        """
        # Read the API key from the security file
        keys_df = pd.read_csv(config['input']['keys'])
        api_key = keys_df.loc[keys_df['organization'] == 'Google', 'key_pw'].values[0]
        days_ahead = config['date_range']['days_ahead']

        url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        params = {
            "key": api_key,
            "singleEvents": "true",
            "timeMin": datetime.now(timezone.utc).isoformat(),
            "timeMax": (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat(),
            "fields": "items, nextPageToken",
            "maxResults": 100
        }

        all_events = []
        while True:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                all_events.extend(data.get("items", []))
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break
                params["pageToken"] = next_page_token
            else:
                logging.error(f"def get_events(): Error: {response.status_code} - {response.text}")
                break

        df = pd.json_normalize(all_events)
        if df.empty:
            logging.info(f"def get_events(): No events found for calendar_id: {calendar_id}")
            return df

        return self.clean_events(df)

    def clean_events(self, df):
        """
        This function performs the following operations:
        1. Ensures required columns exist in the DataFrame.
        2. Moves values from 'start.date' and 'end.date' to 'start.dateTime' and 'end.dateTime' if necessary.
        3. Drops the 'start.date' and 'end.date' columns.
        4. Subsets the DataFrame to only useful columns.
        5. Extracts and converts the price from the 'description' column.
        6. Cleans the 'description' column by removing HTML tags and unnecessary whitespace.
        7. Splits 'start.dateTime' and 'end.dateTime' into separate date and time columns.
        8. Renames columns to more descriptive names.
        9. Adds a 'Type_of_Event' column based on keywords in the 'Description' column.
        10. Converts 'Start_Date' and 'End_Date' to date format.
        11. Extracts the day of the week from 'Start_Date' and adds it to the 'Day_of_Week' column.
        12. Reorders the columns for better readability.
        13. Sorts the DataFrame by 'Start_Date' and 'Start_Time'.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing event data.

        Returns:
        pandas.DataFrame: The cleaned and processed DataFrame with relevant event information.
        """
        # Avoid modifying the original DataFrame
        df = df.copy()

        # Ensure required columns exist
        required_columns = ['htmlLink', 'summary', 'start.date', 'end.date', 'location', 'start.dateTime', 'end.dateTime', 'description']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        # Move values from 'start.date' and 'end.date' to 'start.dateTime' and 'end.dateTime' if necessary
        df['start.dateTime'] = df['start.dateTime'].fillna(df['start.date'])
        df['end.dateTime'] = df['end.dateTime'].fillna(df['end.date'])

        # Drop the 'start.date' and 'end.date' columns
        df.drop(columns=['start.date', 'end.date'], inplace=True)

        # Subset df to only useful columns 
        df = df[['htmlLink', 'summary', 'location', 'start.dateTime', 'end.dateTime', 'description']]

        # Extract and convert the price
        df['Price'] = pd.to_numeric(df['description'].str.extract(r'\$(\d{1,5})')[0], errors='coerce')

        # Clean the description
        # Remove HTML tags and unnecessary whitespace
        df['description'] = df['description'].apply(
            lambda x: re.sub(r'\s{2,}', ' ', re.sub(r'<[^>]*>', ' ', str(x) if pd.notnull(x) else '')).strip()
        ).str.replace('&#39;', "'").str.replace("you're", "you are")

        # Function to split datetime into date and time
        def split_datetime(datetime_str):
            if 'T' in datetime_str:
                date_str, time_str = datetime_str.split('T')
                time_str = time_str[:8]  # Remove the timezone part
            else:
                date_str = datetime_str
                time_str = None
            return date_str, time_str

        # Apply the function to extract dates and times
        df['Start_Date'], df['Start_Time'] = zip(*df['start.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df['End_Date'], df['End_Time'] = zip(*df['end.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))

        # Drop columns
        df.drop(columns=['start.dateTime', 'end.dateTime'], inplace=True)

        # Rename columns
        df = df.rename(columns={
            'htmlLink': 'URL',
            'summary': 'Name_of_the_Event',
            'location': 'Location',
            'description': 'Description'
        })

        # Add 'Type_of_Event' column
        # Dictionary to map words of interest (woi) to 'Type_of_Event'
        event_type_map = {
            'class': 'class',
            'dance': 'social dance',
            'dancing': 'social dance',
            'weekend': 'workshop',
            'workshop': 'workshop'
        }

        # Function to determine 'Type_of_Event'
        def determine_event_type(name):
            name_lower = name.lower()
            if 'class' in name_lower and 'dance' in name_lower:
                return 'social dance'  # Priority rule
            for woi, event_type in event_type_map.items():
                if woi in name_lower:
                    return event_type
            return 'other'  # Default if no woi match

        # Apply the function to determine 'Type_of_Event'
        df['Type_of_Event'] = df['Description'].apply(determine_event_type)

        # Convert Start_Date and End_Date to date format
        df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.date
        df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce').dt.date

        # Extract the day of the week from Start_Date and add it to the Day_of_Week column
        df['Day_of_Week'] = pd.to_datetime(df['Start_Date']).dt.day_name()

        # Reorder the columns
        df = df[['URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week', 'Start_Date', 
                'End_Date', 'Start_Time', 'End_Time', 'Price', 'Location', 'Description']]

        # Sort the DataFrame by Start_Date and Start_Time
        df = df.sort_values(by=['Start_Date', 'Start_Time']).reset_index(drop=True)

        # Return the collected events as a pandas dataframe

        return df
   
# Run the crawler
if __name__ == "__main__":

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    # Instantiate class'
    db_handler = DatabaseHandler(config)
    llm_handler = LLMHandler(config_path="config/config.yaml")

    # Create the database tables
    db_handler.create_tables()

    # Run the crawler process

    process = CrawlerProcess(settings={
        "LOG_FILE": config['logging']['scrapy_log_file'],
        "LOG_LEVEL": "INFO",
        "DEPTH_LIMIT": config['crawling']['depth_limit'],
        "FEEDS": {
            "output/output.json": {
                "format": "json"
            }
        }
    })

    # Start the crawler process
    process.crawl(EventSpider)
    process.start()

    logging.info("__main__: Crawler process completed.")

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
