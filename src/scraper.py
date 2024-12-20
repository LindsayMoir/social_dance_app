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
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (create_engine, update, Table, Column, 
                        MetaData, Boolean, BigInteger, Text, TIMESTAMP)
from scrapy.crawler import CrawlerProcess
import sys
import time
import yaml

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

logging.info("Working directory is: %s", os.getcwd())


# Database connection
def get_db_connection():
    """
    Returns a SQLAlchemy engine for connecting to the PostgreSQL database.
    """
    try:
        # Read the database connection parameters from the config
        connection_string = (
            f"postgresql://{config['database']['user']}:"
            f"{config['database']['password']}@"
            f"{config['database']['host']}/"
            f"{config['database']['name']}"
        )
        # Create and return the SQLAlchemy engine
        conn = create_engine(connection_string)
        return conn
    
    except Exception as e:
        logging.error("Database connection failed: %s", e)
        return None


def url_in_table(url):
    """
    Check if a URL is already present in the 'urls' table.

    Parameters:
    url (str): The URL to check in the 'urls' table.

    Returns:
    bool: True if the URL is present in the table, False otherwise.
    """
    conn = get_db_connection()

    if conn is None:
        logging.error("Failed to connect to the database while checking URL in url_in_table.")
        return False

    query = 'SELECT * FROM urls WHERE links = %s'
    params = (url,)  # Tuple containing the parameter
    df = pd.read_sql(query, conn, params=params)

    if df.shape[0] > 0:
        logging.info(f"URL {url} is present in the 'urls' table def url_in_table.")
        return True
    else:
        logging.info(
            f"URL {url} is NOT present in the 'urls' table. "
            "This is highly unlikely and should be looked at"
        )
        return False


def update_url(url, update_other_links, relevant, increment_crawl_trys):
        """
        Updates the `relevant` column to True and increments the `crawl_trys` column for a specific URL in the database.

        Parameters:
        url (str): The URL to be updated.
        update_other_links (str): The value to update the `other_links` column. If 'No', the column will not be updated.
        increment_crawl_trys (int): The number by which to increment the `crawl_trys` column.
        relevant (bool): The value to set for the `relevant` column.

        Raises:
        ConnectionError: If the connection to the database fails.

        Returns:
        None
        """
        # Connect to the database
        conn = get_db_connection()

        if conn is None:
            raise ConnectionError("Failed to connect to the database.")

        # Reflect the table structure
        metadata = MetaData()
        metadata.reflect(bind=conn)
        table = metadata.tables['urls']

        # Build the update query
        query = (
            update(table)
            .where(table.c.links == url)
            .values(
                time_stamps=datetime.now(),
                other_links=update_other_links if update_other_links != 'No' else table.c.other_links,
                crawl_trys=table.c.crawl_trys + increment_crawl_trys,
                relevant=relevant
            )
        )

        # Execute the query
        with conn.begin() as connection:
            connection.execute(query)
        logging.info(f"def updated_url Updated URL: {url}")


def write_url_to_db(org_names, keywords, url, other_links, relevant, increment_crawl_trys):
        
        """
        Write or update a URL entry in the database.

        Args:
            url (str): The URL to write to the database.
            relevant (bool): Whether the URL is marked as relevant.
            other_links (str): Additional links or notes for the URL.
            increment_crawl_trys (int): Number of times the URL has been crawled.
        """
        #try:
        conn = get_db_connection()

        if conn is None:
            logging.error("Failed to connect to the database in write_url while trying to write {url}")

        try:
            update_url(url, other_links, relevant, increment_crawl_trys)
            
        except Exception as e:
            logging.info(f"Unable to update {url} \n in write_url. Assume the url or table does not exist: {e}")
            update_df = pd.DataFrame({
                                    "time_stamps": [datetime.now()],
                                    "org_names": [org_names],
                                    "keywords": [keywords],
                                    "links": [url],
                                    "other_links": [other_links],
                                    "relevant": [relevant],
                                    "crawl_trys": [increment_crawl_trys]})
            
            # If the table exists, this will append it. If the table does not exist, this will create it.
            update_df.to_sql('urls', conn, if_exists='append', index=False)


def write_events_to_db(df, url, keywords, org_name):
        """
        Parameters:
        df (pd.DataFrame): DataFrame containing event data with columns such as 'Start_Date', 'End_Date', 'Start_Time', 'End_Time', and optionally 'Price'.
        url (str): The URL from which the events were scraped.
        keywords (str): Keywords associated with the events.
        org_name (str): Name of the organization hosting the events.

        Returns:
        None

        Raises:
        None

        Notes:
        - Converts 'Start_Date' and 'End_Date' columns to date format.
        - Converts 'Start_Time' and 'End_Time' columns to time format.
        - Handles the 'Price' column by converting it to numeric and filling invalid values with NaN.
        - Adds 'Time_Stamp', 'Keyword', and 'Org_Name' columns to the DataFrame.
        - Reorders columns to place 'Org_Name' and 'Keyword' first.
        - Writes the DataFrame to the 'events' table in the database using SQLAlchemy.
        - Logs a warning if the 'Price' column is missing or empty.
        - Logs an error if the database connection fails.
        """
        df.to_csv('events.csv', index=False)

        # Convert date and time fields to the appropriate formats, tolerantly
        for col in ['Start_Date', 'End_Date']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        for col in ['Start_Time', 'End_Time']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

        # Check if the Price column exists
        if 'Price' in df.columns and not df['Price'].isnull().all():
            # Handle empty or invalid Price values
            df['Price'] = df['Price'].replace({'\$': '', '': None}, regex=True)
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Convert to float, set invalid to NaN
        else:
            logging.warning("The 'Price' column is missing or empty. Filling with NaN.")
            df['Price'] = float('nan')  # Add a Price column with NaN if it doesn't exist or is empty

        # Add additional columns
        df['Time_Stamps'] = datetime.now()
        df['Keywords'] = keywords
        df['Org_Names'] = org_name

        # Reorder the columns to make 'Org_Name', 'Keyword' columns first and second
        columns_order = [ 'Org_Names', 'Keywords'] + [
            col for col in df.columns if col not in ['Org_Names', 'Keywords']]
        df = df[columns_order]

        # Get database connection using SQLAlchemy
        conn = get_db_connection()  # Ensure this returns a SQLAlchemy engine
        if conn is None:
            logging.error("Failed to connect to the database.")
            return
        
        else:
            # Use Pandas to write the DataFrame to the database
            df.to_sql(
                name='events',
                con=conn,
                if_exists='append',  # Appends to the table or creates it if it doesn't exist
                index=False,         # Avoids creating a DataFrame index column in the table
                method='multi'       # Enables efficient bulk inserts
            )
            logging.info(f"Successfully wrote events to the database from URL: {url}")


def dedup():
        """
        This method connects to the database using SQLAlchemy, reads the 'events' 
        and 'urls' tables into pandas DataFrames, removes duplicate rows based on 
        specified columns, and then writes the deduplicated DataFrames back to the 
        database, replacing the original tables.

        For the 'events' table, duplicates are identified based on the following columns:
        - "Keyword"
        - "Type_of_Event"
        - "Name_of_the_Event"
        - "Day_of_Week"
        - "Start_Date"
        - "End_Date"

        For the 'urls' table, duplicates are identified based on the "links" column.

        The method logs the shape of the DataFrames before and after deduplication.

        Raises:
            Exception: If there is an error reading the 'events' table from the database.
        """
        # Get the SQLAlchemy connection
        conn = get_db_connection()

        # Read the events table into a pandas DataFrame
        try:
            df = pd.read_sql('SELECT * FROM events', conn)
            shape_before = df.shape
        except Exception as e:
            logging.error(f"Failed to read events table: {e}")
            return

        # Deduplicate the DataFrame based on specified columns and keep the last version
        deduplicated_df = df.drop_duplicates(
            subset=["Org_Name", "Keyword", "URL", "Type_of_Event", "Day_of_Week", "Start_Date", "End_Date"],
            keep="last"
        )
        shape_after = deduplicated_df.shape

        # This will drop the existing `events` table and replace it
        deduplicated_df.to_sql("events", conn, index=False, if_exists="replace")

        logging.info(f"Before duplicates removed from events table shape was {shape_before} and after {shape_after}.")

        # Read the urls table into a pandas DataFrame
        df = pd.read_sql('SELECT * FROM urls', conn)
        shape_before = df.shape

        # Deduplicate the DataFrame based on specified columns and keep the last version
        deduplicated_df = df.drop_duplicates(
            subset=["links"],
            keep="last"
        )
        shape_after = deduplicated_df.shape

        # This will drop the existing `urls` table and replace it
        deduplicated_df.to_sql("urls", conn, index=False, if_exists="replace")

        logging.info(f"Before duplicates removed from urls table shape was {shape_before} and after {shape_after}.")


def set_calendar_urls():
        """
        This function retrieves URLs from the 'urls' table in the database where the 'other_links' 
        column contains the word 'calendar'. It then marks these URLs as relevant for crawling 
        again by updating the 'relevant' column to True.

        The function performs the following steps:
        1. Establishes a connection to the database using SQLAlchemy.
        2. Executes a SQL query to select URLs from the 'urls' table where 'other_links' contains 'calendar'.
        3. Iterates over the retrieved URLs and checks if they are already marked as relevant.
        4. If a URL is not marked as relevant, it updates the URL to be relevant for crawling.
        5. Logs the number of URLs marked for crawling or indicates if no URLs were found.

        Raises:
            Exception: If there is an error while setting the calendar URLs.

        Logs:
            Information about URLs already marked as relevant.
            Information about the number of URLs marked for crawling.
            Errors encountered during the process.
        """
        # Get the SQLAlchemy connection
        conn = get_db_connection()

        try:
            # Read the URLs from the database with proper SQL syntax for PostgreSQL
            query = "SELECT * FROM urls WHERE other_links ILIKE %s"
            params = ("%calendar%",)  # Tuple containing the parameter
            urls_df = pd.read_sql_query(query, conn, params=params)

            # Update relevant column to True for each matching URL
            if urls_df.shape[0] > 0:
                for _, row in urls_df.iterrows():
                    if row['relevant'] == False:
                        logging.info(f"URL {row['links']} is already marked as relevant.")
                        continue
                    url = row['links']
                    update_url(url, update_other_links='No', relevant=True, increment_crawl_trys=0)
                logging.info(f"{len(urls_df)} calendar URLs marked for crawling again.")
            else:
                logging.info("No calendar URLs found to mark for crawling in set_calendar().")

        except Exception as e:
            logging.error(f"Error while setting calendar URLs: {e}")
            raise


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
        conn = get_db_connection()

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

            # We need to write the url to the database, if it is not already there
            write_url_to_db(org_name, keywords, url, other_links='', relevant=True, increment_crawl_trys=1)

            logging.info(f"Starting crawl for URL: {url}")
            yield scrapy.Request(url=url, callback=self.parse, cb_kwargs={'keywords': keywords, 
                                                                            'org_name': org_name, 
                                                                            'url': url})


    def parse(self, response, keywords, org_name, url):
        """
        Args:
            response (scrapy.http.Response): The response object to parse.
            keywords (list): A list of keywords to check for relevance.
            org_name (str): The name of the organization to check for relevance.
            url (str): The URL of the current page being parsed.

        Returns:
            generator: A generator yielding scrapy.Request objects for further crawling.

        This function extracts links from the main page and handles Facebook links differently. 
        It also extracts iframe sources and updates the URL if relevant. 
        The function checks for relevance of the links and continues crawling if they are relevant.
        """
        # Initialize page_links
        page_links = []
        facebooks_events_links = []
        iframe_links = []

        # Extract links from the main page but handle facebook links differently
        if 'facebook' in url:

            # Process the regular group facebook page

            # We want to see if there are any facebook event links in this url
            facebooks_events_links = self.fb_get_event_links(url)

            if facebooks_events_links:
                for event_link in facebooks_events_links:

                    # Write the event link to the database
                    write_url_to_db(org_name, keywords, event_link, other_links=url, relevant=True, increment_crawl_trys=1)

                    # Call playwright to extract the event details
                    extracted_text = self.fb_extract_text(event_link)

                    # Check keywords in the extracted text
                    keyword_status = self.check_keywords_in_text(event_link, extracted_text, keywords, org_name)

                    if keyword_status:
                        # Call the llm to process the extracted text
                        llm_status = self.process_llm_response(event_link, extracted_text, keywords, org_name)

                        if llm_status:
                            # Mark the event link as relevant
                            update_url(event_link, url, increment_crawl_trys=0, relevant=True)
                        else:
                            # Mark the event link as irrelevant
                            update_url(event_link, url, relevant=False, increment_crawl_trys=0)
                    else:
                        # Mark the event link as irrelevant
                        update_url(event_link, url, relevant=False, increment_crawl_trys=0)

            # Check if the URL contains 'login'
            if 'login' in url or '/groups/' not in url:
                logging.info(f"URL {url} marked as irrelevant due to Facebook login link.")
                update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)

            else:
                # Normal facebook text. I want to process this with playwright but the version that only returns text
                extracted_text = self.fb_extract_text(url)

                # Check keywords in the extracted text
                keyword_status = self.check_keywords_in_text(url, extracted_text, keywords, org_name)

                if keyword_status:
                    # Call the llm to process the extracted text
                    llm_status = self.process_llm_response(url, extracted_text, keywords, org_name)

                    if llm_status:
                        # Mark the event link as relevant
                        update_url(url, update_other_links='', relevant=True, increment_crawl_trys=0)
                    else:
                        # Mark the event link as irrelevant
                        update_url(url, update_other_links='', relevant=False, increment_crawl_trys=0) 

                else:
                    logging.info(f"URL {url} marked as irrelevant since there are no keywords and/or events in URL.")
                    update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)

        # Get all of the subsidiary links on the page   
        else:
            page_links = response.css('a::attr(href)').getall()
            page_links = [response.urljoin(link) for link in page_links if link.startswith('http')]
            page_links = page_links[:config['crawling']['max_urls']]  # Limit the number of links
            page_links = page_links + url  # Add the current URL to the list

        # Extract iframe sources
        iframe_links = response.css('iframe::attr(src)').getall()
        iframe_links = [response.urljoin(link) for link in iframe_links if link.startswith('http')]

        if iframe_links:
            update_url(url, update_other_links='calendar', relevant=True, increment_crawl_trys=0)
            for calendar_url in iframe_links:
                self.fetch_google_calendar_events(calendar_url, keywords, org_name, url)

        logging.info(f"Found {len(page_links)} page_links and {len(iframe_links)} iframe_links on {response.url}")

        # Put all links together and make sure that they get crawled and their subsequent levels get crawled
        all_links = set(page_links + iframe_links + facebooks_events_links + [url])

        # Check for relevance and crawl further
        for link in all_links:
            if link not in self.visited_links:
                self.visited_links.add(link)  # Mark the page link as visited
                if self.is_relevant(link, keywords, org_name):
                    logging.info(f"Starting crawl for URL: {link}")
                    yield response.follow(url=link, callback=self.parse, cb_kwargs={'keywords': keywords, 
                                                                                    'org_name': org_name, 
                                                                                    'url': link})
    

    def is_relevant(self, url, keywords, org_name):
        """
        Determine the relevance of a given URL based on its content, keywords, or organization name.

        Parameters:
        url (str): The URL to be evaluated.
        keywords (list of str): A list of keywords to check within the URL content.
        org_name (str): The name of the organization to check within the URL content.

        Returns:
        bool: True if the URL is relevant, False otherwise.
        """
        if 'facebook' in url:
            return self.handle_facebook_url(url, keywords, org_name)
        else:
            return self.handle_non_facebook_url(url, keywords, org_name)
        

    def handle_facebook_url(self, url, keywords, org_name):
        """
        This function processes a given Facebook URL to check its relevance based on certain criteria.
        It logs the process and updates the URL status accordingly.

        Parameters:
        url (str): The Facebook URL to be processed.

        Returns:
        bool: True if the URL is relevant, False otherwise.

        The function performs the following checks:
        1. If the URL contains 'login', it is marked as irrelevant.
        2. If the URL is a Facebook group link, it checks the group's relevance.
        3. Attempts to extract text from the URL and checks for relevant keywords in the extracted text.
        """
        # Check if the URL contains 'login'
        if 'login' in url:
            logging.info(f"URL {url} marked as irrelevant due to Facebook login link.")
            update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)
            return False

        if '/groups/' in url:
            return self.check_facebook_group(url)

        extracted_text = self.fb_extract_text(url)
        if not extracted_text:
            logging.error(f"Failed to extract text or login to Facebook for URL: {url}")
            update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)
            return False

        return self.check_keywords_in_text(url, extracted_text, keywords, org_name)
    

    def check_facebook_group(self, url):
        """
        Determines whether a Facebook url within a facebook group page is relevant based on the presence of event links.

        Args:
            url (str): The URL of the Facebook group to check for event links.

        Returns:
            bool: True if event links are found, False otherwise.
        """
        # Extract event links from the Facebook group
        logging.info(f"Attempting to extract event links from Facebook group: {url}")
        event_links = self.fb_get_event_links(url)
        if event_links:
            logging.info(f"Found {len(event_links)} event links on Facebook group page.")
            for event_link in event_links:
                logging.info(f"Event link: {event_link}")
            logging.info(f"URL {url} marked as relevant due to extracted Facebook event links.")
            return True

        logging.info(f"No event links found on Facebook group page: {url}")
        update_url(url, update_other_links='No', relevant=False, increment_crawl_trys=0)
        return False
    
    
    def fb_get_event_links(self, group_url):
        """
        Navigate to a Facebook group URL and extract event links.
        Args:
            group_url (str): The URL of the Facebook group from which to extract event links.

        Returns:
            list: A list of event links extracted from the Facebook group.

        Raises:
            Exception: If there is an error during the navigation or extraction process.

        Notes:
            - This function uses Playwright to automate the browser interaction.
            - If login is required, it will attempt to log in using stored cookies or provided credentials.
            - Cookies are saved after a successful login to avoid repeated logins in future sessions.
            - The function extracts all anchor tags from the page and filters those that contain '/events/' in their href attribute.
        """
        event_links = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()

            # Load cookies if available
            cookie_file = config['input']['fb_cookies']
            if os.path.exists(cookie_file):
                with open(cookie_file, "r") as f:
                    cookies = json.load(f)
                    context.add_cookies(cookies)

            page = context.new_page()
            page.goto(group_url, timeout=180000)

            # Get credentials for facebook
            keys_df = pd.read_csv(config['input']['keys'])
            email = keys_df.loc[keys_df['Organization'] == 'Facebook', 'App_ID'].values[0]
            password = keys_df.loc[keys_df['Organization'] == 'Facebook', 'Key'].values[0]

            # Perform login if necessary
            if 'login' in page.url:
                page.goto("https://www.facebook.com/login", timeout=180000)
                page.fill("input[name='email']", email)
                page.fill("input[name='pass']", password)
                page.click("button[name='login']")
                page.wait_for_selector("div[role='main']", timeout=180000)

            # Save cookies
            cookies = context.cookies()
            with open(cookie_file, "w") as f:
                json.dump(cookies, f)

            # Navigate back to the group URL
            page.goto(group_url, timeout=180000)

            # Extract event links
            logging.info("Extracting event links from the group page.")
            all_links = page.eval_on_selector_all("a", "elements => elements.map(e => e.href || '')")
            event_links = [link for link in all_links if '/events/' in link and 'facebook.com' in link]
            logging.info(f"Extracted {len(event_links)} event links.")
            for event_link in event_links:
                logging.info(f"Event link: {event_link}")

        return event_links
    

    def handle_non_facebook_url(self, url, keywords, org_name):
        """
        This method extracts text from the given URL using Playwright and checks if the extracted text contains 
        any of the specified keywords. If no text is extracted, it logs an informational message and returns False.

        Args:
            url (str): The URL to be processed.
            keywords (list of str): A list of keywords to check for in the extracted text.
            org_name (str): The name of the organization to check for in the extracted text.

        Returns:
            bool: True if the extracted text contains any of the specified keywords or the organization name, False otherwise.
        """
        # Extract text from the URL
        extracted_text = self.extract_text_with_playwright(url)
        if not extracted_text:
            logging.info(f"No text extracted for URL: {url}")
            return False

        return self.check_keywords_in_text(url, extracted_text, keywords, org_name)
    

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
                logging.info(f"Keywords found in extracted text for URL: {url}")
                return self.process_llm_response(url, extracted_text, keywords, org_name)

        if 'calendar' in url:
            logging.info(f"URL {url} marked as relevant because 'calendar' is in the URL.")
            return True

        logging.info(f"URL {url} marked as irrelevant since there are no keywords, events, or 'calendar' in URL.")
        return False
    

    def process_llm_response(self, url, extracted_text, keywords, org_name):
        """
        Generate a prompt, query a Language Learning Model (LLM), and process the response.

        This method generates a prompt based on the provided URL and extracted text, queries the LLM with the prompt,
        and processes the LLM's response. If the response is successfully parsed, it converts the parsed result into
        a DataFrame, writes the events to the database, and logs the relevant information.

        Args:
            url (str): The URL of the webpage being processed.
            extracted_text (str): The text extracted from the webpage.
            keywords (list): A list of keywords relevant to the events.
            org_name (str): The name of the organization associated with the events.

        Returns:
            bool: True if the LLM response is successfully processed and events are written to the database, False otherwise.
        """
        # Generate prompt, query LLM, and process the response.
        prompt = self.generate_prompt(url, extracted_text)
        llm_response = self.query_llm(prompt, url)

        if llm_response:
            parsed_result = self.extract_and_parse_json(llm_response, url)
            if parsed_result:
                events_df = pd.DataFrame(parsed_result)
                write_events_to_db(events_df, url, keywords, org_name)
                logging.info(f"URL {url} marked as relevant with events written to the database.")
                return True
        
        else:
            logging.error(f"Failed to process LLM response for URL: {url}")
            return False
    
    def generate_prompt(self, url, extracted_text):
        """
        Generate a prompt for a language model using extracted text and configuration details.

        Args:
            url (str): The URL of the webpage from which the text was extracted.
            extracted_text (str): The text extracted from the webpage.

        Returns:
            str: A formatted prompt string for the language model.
        """
        #Generate the LLM prompt using the extracted text and configuration details.
        txt_file_path = config['prompts']['is_relevant']
        with open(txt_file_path, 'r') as file:
            is_relevant_txt = file.read()

        # Define the date range for event identification
        start_date = datetime.now()
        end_date = start_date + pd.DateOffset(months=config['date_range']['months'])

        # Generate the full prompt
        prompt = (
            f"The following text was extracted from a webpage {url}:\n\n"
            f"{extracted_text}\n\n"
            f"Identify any events (social dances, classes, workshops) within the date range "
            f"{start_date} to {end_date}."
            f"{is_relevant_txt}"
        )

        return prompt


    def query_llm(self, prompt, url):
        """
        Args:
            prompt (str): The prompt to send to the LLM.
            url (str): The URL associated with the prompt.
        Returns:
            str: The response from the LLM if available, otherwise None.
        Raises:
            FileNotFoundError: If the keys file specified in the config is not found.
            KeyError: If the 'OpenAI' organization key is not found in the keys file.
            Exception: For any other exceptions that may occur during the API call.
        """
        # Read the API key from the security file
        keys_df = pd.read_csv(config['input']['keys'])
        api_key = keys_df.loc[keys_df['Organization'] == 'OpenAI', 'Key'].values[0]

        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()

        # Query the LLM
        response = client.chat.completions.create(
            model=config['llm']['url_evaluator'],
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )

        if response.choices:
            logging.info(f"LLM response received based on url {url}.")
            return response.choices[0].message.content.strip()
        
        return None


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
            logging.info("No events found in result.")
            return None
        
        # Check if the response contains JSON
        if 'json' in result:
            start_position = result.find('[')
            end_position = result.find(']') + 1
            if start_position != -1 and end_position != -1:
                # Extract JSON part
                json_string = result[start_position:end_position]

                try:
                    # Convert JSON string to Python object
                    logging.info("JSON found in result.")
                    events_json =json.loads(json_string)
                    return events_json
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON: {e}")
                    return None

        logging.info("No JSON found in result.")
        
        return None
    

    def fb_extract_text(self, group_url):
        """
        Args:
        group_url (str): The URL of the Facebook group page to extract text from.

    Returns:
        str: The extracted text content from the Facebook group page. Returns an empty string if extraction fails.

    Raises:
        Exception: If there is an error during the extraction process, such as a timeout or login failure.

    Notes:
        - This function uses Playwright to automate the browser interaction.
        - Facebook credentials are read from a CSV file located at '/mnt/d/OneDrive/Security/keys.csv'.
        - Cookies are loaded from and saved to a file specified in the configuration.
        - If direct access to the group URL fails, the function attempts to log in using the provided credentials.
        - A screenshot is saved for debugging purposes if there is a failure while waiting for the group content to load.
        """
        extracted_text = ""
        cookie_file = config['input']['fb_cookies']

        # Get Facebook credentials
        keys_df = pd.read_csv('/mnt/d/OneDrive/Security/keys.csv')
        email = keys_df.loc[keys_df['Organization'] == 'Facebook', 'App_ID'].values[0]
        password = keys_df.loc[keys_df['Organization'] == 'Facebook', 'Key'].values[0]

        with sync_playwright() as p:
            logging.info("Launching browser in headless mode.")
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()

            # Load cookies if available
            if os.path.exists(cookie_file):
                logging.info("Loading cookies from file.")
                with open(cookie_file, "r") as f:
                    cookies = json.load(f)
                    context.add_cookies(cookies)
                logging.info("Cookies loaded successfully.")

            page = context.new_page()

            # Attempt to navigate directly to the group URL
            logging.info(f"Attempting direct access to {group_url}")
            page.goto(group_url, timeout=180000)
            logging.info(f"Current page URL after navigation: {page.url}")

            # Check if login is needed
            if 'login' in page.url:
                logging.info("Direct access failed. Performing login...")
                page.goto("https://www.facebook.com/login", timeout=180000)
                page.fill("input[name='email']", email)
                page.fill("input[name='pass']", password)
                page.click("button[name='login']")
                page.wait_for_timeout(5000)  # Wait briefly for login to process

                # Verify login success
                if 'login' in page.url or 'recover' in page.url:
                    logging.error("Login failed. Still on login or reset page.")
                    browser.close()
                    return None

                logging.info("Login successful. Saving cookies for future use.")
                cookies = context.cookies()
                with open(cookie_file, "w") as f:
                    json.dump(cookies, f)

                # Navigate back to the group URL
                page.goto(group_url, timeout=180000)
                logging.info(f"Re-attempting navigation to group URL: {group_url}")
                logging.info(f"Current page URL after login: {page.url}")

            # Wait for the group content to load
            try:
                logging.info("Waiting for group page content to load...")
                page.wait_for_selector("div[role='main']", timeout=180000)
                extracted_text = page.inner_text("div[role='main']")
                logging.info("Successfully extracted text from the group page.")

            except Exception as e:
                logging.error(f"Timeout or failure while waiting for selector: {e}")
                # Capture a screenshot for debugging
                page.screenshot(path="facebook_group_debug.png")
                logging.info("Saved screenshot to 'facebook_group_debug.png' for debugging.")

            browser.close()
            logging.info("Browser closed.")

        return extracted_text

    
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
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=30000)  # Wait up to 30 seconds for the page to load
                page.wait_for_timeout(3000)  # Additional wait for dynamic content

                # Extract the visible text
                content = page.content()  # Get the full HTML content
                soup = BeautifulSoup(content, 'html.parser')
                extracted_text = ' '.join(soup.stripped_strings)

                browser.close()
                return extracted_text
            
        except Exception as e:
            logging.error(f"Failed to extract text with Playwright for URL {url}: {e}")
            return None


    def fetch_google_calendar_events(self, calendar_url, keywords, org_name, url):
        """
        Fetch events from a Google Calendar and process them.

        This function extracts calendar IDs from the provided Google Calendar URL,
        fetches events using these IDs, and processes the events by writing them
        to a database and updating URLs.

        Args:
            calendar_url (str): The URL of the Google Calendar to fetch events from.
            keywords (list): A list of keywords to associate with the events.
            org_name (str): The name of the organization associated with the events.
            url (str): The URL to update after processing the events.

        Returns:
            None
        """
        calendar_id_pattern = r'src=([^&]+%40group.calendar.google.com)'
        calendar_ids = re.findall(calendar_id_pattern, calendar_url)

        if calendar_ids:
            decoded_calendar_ids = [id.replace('%40', '@') for id in calendar_ids]
            logging.info(f"Found {len(calendar_ids)} group.calendar.google.com IDs: {decoded_calendar_ids}")

            for calendar_id in decoded_calendar_ids:
                events_df = self.get_events(calendar_id)
                if not events_df.empty:
                    write_events_to_db(events_df, calendar_url, keywords, org_name)
                    self.update_url(calendar_url, update_other_links=url, relevant=True, increment_crawl_trys=1, )
                    self.update_url(url, update_other_links=calendar_url, relevant=True, increment_crawl_trys=1)
        else:
            start_idx = calendar_url.find("src=") + 4
            end_idx = calendar_url.find("&", start_idx)
            calendar_id = calendar_url[start_idx:end_idx] if end_idx != -1 else calendar_url[start_idx:]
            calendar_id = base64.b64decode(calendar_id + '=' * (4 - len(calendar_id) % 4)).decode('utf-8')

            events_df = self.get_events(calendar_id)
            if not events_df.empty:
                write_events_to_db(events_df, calendar_url, keywords, org_name)
                update_url(calendar_url, update_other_links=url, relevant=True, increment_crawl_trys=1)
                update_url(url, update_other_links=calendar_url, relevant=True, increment_crawl_trys=1)


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
        api_key = keys_df.loc[keys_df['Organization'] == 'Google', 'Key'].values[0]
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
                logging.error(f"Error: {response.status_code} - {response.text}")
                break

        df = pd.json_normalize(all_events)
        if df.empty:
            logging.info(f"No events found for calendar_id: {calendar_id}")
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
    logging.info(f"Starting the crawler process at {start_time}")

    # Run the crawler process
    process = CrawlerProcess(settings={
        "FEEDS": {"output.json": {"format": "json"}},
        "LOG_LEVEL": "INFO",
        "DEPTH_LIMIT": config['crawling']['depth_limit'],
    })

    # First crawl
    process.crawl(EventSpider)
    process.start()
    logging.info("Crawler process completed.")

    # Run deduplication and set calendar URLs
    dedup()
    set_calendar_urls()

    # Get the end time
    end_time = datetime.now()
    logging.info(f"Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time}")
