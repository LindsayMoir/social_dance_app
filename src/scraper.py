"""
scraper.py

This module defines the EventSpider class for crawling and a ScraperManager class 
to orchestrate the crawl-and-check process. The EventSpider handles URL extraction,
dynamic content extraction, and Google Calendar event processing. The ScraperManager
runs the crawler (by launching a separate process), checks if the event counts meet 
thresholds, and if not, adjusts configuration and re-runs the crawl. After a successful 
run, it restores any files that were moved to a temporary folder.

Dependencies:
    - Scrapy, requests, pandas, yaml, logging, shutil, etc.
    - Local modules: DatabaseHandler (from db.py), LLMHandler (from llm.py), 
      ReadExtract (from rd_ext.py), and credentials (from credentials.py).
"""

import base64
from datetime import datetime, timedelta, timezone
import logging
import os
import pandas as pd
import re
import requests
import scrapy
from scrapy.crawler import CrawlerProcess
import shutil
import sys
import yaml
import subprocess

from credentials import get_credentials
from db import DatabaseHandler
from llm import LLMHandler
from rd_ext import ReadExtract

# --------------------------------------------------
# Global objects initialization.
# (These globals will be used by EventSpider.)
# --------------------------------------------------
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Instantiate required handlers.
db_handler = DatabaseHandler(config)
llm_handler = LLMHandler(config_path="config/config.yaml")
read_extract = ReadExtract("config/config.yaml")

# --------------------------------------------------
# EventSpider: Handles the crawling process
# --------------------------------------------------
class EventSpider(scrapy.Spider):
    name = "event_spider"

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.visited_link = set()  # Track visited URLs
        self.keywords_list = llm_handler.get_keywords()
        logging.info("\n\nscraper.py starting...")


    def start_requests(self):
        """
        Generate start requests from URLs either in the DB or CSV files.
        """
        conn = db_handler.get_db_connection()
        if conn is None:
            raise ConnectionError("Failed to connect to the database in start_requests.")
        logging.info(f"def start_requests(): Connected to the database: {conn}")

        if self.config['startup']['use_db']:
            query = "SELECT * FROM urls WHERE relevant = true;"
            urls_df = pd.read_sql_query(query, conn)
        else:
            urls_dir = self.config['input']['urls']
            csv_files = [os.path.join(urls_dir, f) for f in os.listdir(urls_dir) if f.endswith('.csv')]
            dataframes = [pd.read_csv(file) for file in csv_files]
            urls_df = pd.concat(dataframes, ignore_index=True)

        for _, row in urls_df.iterrows():
            source = row['source']
            keywords = row['keywords']
            url = row['link']
            logging.info(f"def start_requests(): Starting crawl for URL: {url}")
            yield scrapy.Request(url=url, callback=self.parse,
                                    cb_kwargs={'keywords': keywords, 'source': source, 'url': url})
                

    def parse(self, response, keywords, source, url):
        """
        Parse the response to extract links, iframes, and calendar info.
        """
        page_links = response.css('a::attr(href)').getall()
        page_links = [response.urljoin(link) for link in page_links if link.startswith('http')]
        logging.info(f"def parse(): Found {len(page_links)} links on {response.url}")
        page_links = page_links[: self.config['crawling']['max_website_urls']]
        logging.info(f"def parse(): Limiting to {self.config['crawling']['max_website_urls']} links on {response.url}")

        # Process iframes and extract Google Calendar emails via regex
        iframe_links = response.css('iframe::attr(src)').getall()
        iframe_links = [response.urljoin(link) for link in iframe_links if link.startswith('http')]
        logging.info(f"def parse(): Found {len(iframe_links)} iframe links on {response.url}")

        calendar_pattern = re.compile(r'"gcal"\s*:\s*"([a-zA-Z0-9_.+-]+@group\.calendar\.google\.com)"')
        calendar_emails = calendar_pattern.findall(response.text)
        logging.info(f"def parse(): Extracted Google Calendar emails: {calendar_emails}")

        iframe_links.extend(calendar_emails)
        if iframe_links:
            db_handler.update_url(url, relevant=True, increment_crawl_try=0)
            for calendar_url in iframe_links:
                self.fetch_google_calendar_events(calendar_url, url, source, keywords)

        # Combine main links and current URL, remove Facebook/Instagram URLs.
        all_links = {url} | set(page_links)

        # Filter out unwanted links (i.e. those that do not contain 'facebook' or 'instagram').
        filtered_links = {link for link in all_links if 'facebook' not in link and 'instagram' not in link}

        # Determine the unwanted links (those that were filtered out).
        unwanted_links = all_links - filtered_links

        # Write unwanted (Facebook/Instagram) links to the database.
        for link in unwanted_links:
            db_handler.write_url_to_db(source, keywords, link, relevant=False, increment_crawl_try=1)
            logging.info(f"def parse(): Recorded unwanted URL (Facebook/Instagram): {link}")

        logging.info(f"def parse(): {len(filtered_links)} links remain on {response.url}")
        logging.info(f"def parse(): Here is the current list of all_links. \n{filtered_links}")

        for link in filtered_links:
            if link in self.visited_link:
                pass
            else:
                self.visited_link.add(link)
                db_handler.write_url_to_db(source, keywords, link, None, 1)
                if len(self.visited_link) >= self.config['crawling']['urls_run_limit']:
                    logging.info(f"def parse(): Reached maximum URL limit: {self.config['crawling']['urls_run_limit']}. Stopping further crawl.")
                    break
                logging.info(f"def parse(): Crawling next URL: {link}")
                self.driver(link, keywords, source)
                yield response.follow(url=link, callback=self.parse,
                                      cb_kwargs={'keywords': keywords, 'source': source, 'url': link})
                

    def driver(self, url, keywords, source):
        """
        Evaluate URL relevance by extracting text and checking keywords.
        """
        extracted_text = read_extract.extract_text_with_playwright(url)
        if extracted_text:
            found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
            if found_keywords:
                logging.info(f"def driver(): Found keywords for URL {url}: {found_keywords}")
                prompt = 'default'
                llm_status = llm_handler.process_llm_response(url, extracted_text, source, keywords, prompt)
                if llm_status:
                    db_handler.write_url_to_db(source, found_keywords, url, True, 1)
                    logging.info(f"def driver(): URL {url} marked as relevant (LLM positive).")
                    return
                else:
                    logging.info(f"def driver(): URL {url} marked as irrelevant (LLM negative).")
            else:
                logging.info(f"def driver(): URL {url} marked as irrelevant (no keywords found).")
        else:
            logging.info(f"def driver(): URL {url} marked as irrelevant (no extracted text).")
        db_handler.write_url_to_db(source, keywords, url, False, 1)


    def fetch_google_calendar_events(self, calendar_url, url, source, keywords):
        """
        Fetch and process events from a Google Calendar.
        """
        logging.info(f"def fetch_google_calendar_events(): Inputs - calendar_url: {calendar_url}, URL: {url}, source: {source}, keywords: {keywords}")
        calendar_ids = self.extract_calendar_ids(calendar_url)
        if not calendar_ids:
            if self.is_valid_calendar_id(calendar_url):
                calendar_ids = [calendar_url]
            else:
                decoded_calendar_id = self.decode_calendar_id(calendar_url)
                if decoded_calendar_id:
                    calendar_ids = [decoded_calendar_id]
                else:
                    logging.error(f"def fetch_google_calendar_events(): Failed to extract valid Calendar ID from {calendar_url}")
                    return
        for calendar_id in calendar_ids:
            self.process_calendar_id(calendar_id, calendar_url, url, source, keywords)


    def extract_calendar_ids(self, calendar_url):
        pattern = r'src=([^&]+%40group.calendar.google.com)'
        ids = re.findall(pattern, calendar_url)
        return [id.replace('%40', '@') for id in ids]
    

    def decode_calendar_id(self, calendar_url):
        try:
            start_idx = calendar_url.find("src=") + 4
            end_idx = calendar_url.find("&", start_idx)
            calendar_id = calendar_url[start_idx:end_idx] if end_idx != -1 else calendar_url[start_idx:]
            if self.is_valid_calendar_id(calendar_id):
                return calendar_id
            padded_id = calendar_id + '=' * (4 - len(calendar_id) % 4)
            decoded = base64.b64decode(padded_id).decode('utf-8', errors='ignore')
            if self.is_valid_calendar_id(decoded):
                return decoded
            logging.error(f"def decode_calendar_id(): Decoded ID is not valid: {decoded}")
            return None
        except Exception as e:
            logging.error(f"def decode_calendar_id(): Exception for {calendar_url} - {e}")
            return None
        

    def is_valid_calendar_id(self, calendar_id):
        pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@(group\.calendar\.google\.com|gmail\.com)$')
        return bool(pattern.fullmatch(calendar_id))
    

    def process_calendar_id(self, calendar_id, calendar_url, url, source, keywords):
        logging.info(f"def process_calendar_id(): Processing calendar_id: {calendar_id} from {calendar_url}")
        events_df = self.get_calendar_events(calendar_id)
        if not events_df.empty:
            logging.info(f"def process_calendar_id(): Found {len(events_df)} events for calendar_id: {calendar_id}")
            db_handler.write_events_to_db(events_df, calendar_url, source, keywords)


    def get_calendar_events(self, calendar_id):
        _, api_key, _ = get_credentials('Google')
        days_ahead = self.config['date_range']['days_ahead']
        api_url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
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
            response = requests.get(api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                all_events.extend(data.get("items", []))
                if not data.get("nextPageToken"):
                    break
                params["pageToken"] = data.get("nextPageToken")
            else:
                logging.error(f"def get_calendar_events(): Error {response.status_code} for calendar_id: {calendar_id}")
                break
        df = pd.json_normalize(all_events)
        if df.empty:
            logging.info(f"def get_calendar_events(): No events found for calendar_id: {calendar_id}")
            return df
        return self.clean_calendar_events(df)
    

    def clean_calendar_events(self, df):
        df = df.copy()
        required_columns = ['htmlLink', 'summary', 'start.date', 'end.date', 'location', 'start.dateTime', 'end.dateTime', 'description']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        df['start.dateTime'] = df['start.dateTime'].fillna(df['start.date'])
        df['end.dateTime'] = df['end.dateTime'].fillna(df['end.date'])
        df.drop(columns=['start.date', 'end.date'], inplace=True)
        df = df[['htmlLink', 'summary', 'location', 'start.dateTime', 'end.dateTime', 'description']]
        df['Price'] = pd.to_numeric(df['description'].str.extract(r'\$(\d{1,5})')[0], errors='coerce')
        df['description'] = df['description'].apply(lambda x: re.sub(r'\s{2,}', ' ', re.sub(r'<[^>]*>', ' ', str(x))).strip()
                                                    ).str.replace('&#39;', "'").str.replace("you're", "you are")
        def split_datetime(dt_str):
            if 'T' in dt_str:
                date_str, time_str = dt_str.split('T')
                return date_str, time_str[:8]
            return dt_str, None
        df['Start_Date'], df['Start_Time'] = zip(*df['start.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df['End_Date'], df['End_Time'] = zip(*df['end.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df.drop(columns=['start.dateTime', 'end.dateTime'], inplace=True)
        df = df.rename(columns={'htmlLink': 'URL', 'summary': 'Name_of_the_Event', 'location': 'Location',
                                  'description': 'Description'})
        event_type_map = {
            'class': 'class',
            'classes': 'class',
            'dance': 'social dance',
            'dancing': 'social dance',
            'weekend': 'workshop',
            'workshop': 'workshop',
            'rehearsal': 'rehearsal'
        }
        def determine_event_type(row):
            name = row.get('Name_of_the_Event') or ''
            description = row.get('Description') or ''
            combined = f"{name} {description}".lower()
            if 'class' in combined and 'dance' in combined:
                return 'class, social dance'
            for word, event_type in event_type_map.items():
                if word in combined:
                    return event_type
            return 'other'
        df['Type_of_Event'] = df.apply(determine_event_type, axis=1)
        df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.date
        df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce').dt.date
        df['Day_of_Week'] = pd.to_datetime(df['Start_Date']).dt.day_name()
        df = df[['URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week', 'Start_Date', 
                 'End_Date', 'Start_Time', 'End_Time', 'Price', 'Location', 'Description']]
        df = df.sort_values(by=['Start_Date', 'Start_Time']).reset_index(drop=True)
        return df

# --------------------------------------------------
# ScraperManager: Orchestrates crawler runs, re-run checks, and restores temp files.
# --------------------------------------------------
class ScraperManager:
    def __init__(self, config, db_handler):
        self.config = config
        self.db_handler = db_handler
    

    def subset_calendar_urls(self, calendar_df, group_df, threshold=100):
        # Define the important sources
        important_sources = {
            "Salsa Caliente",
            "Red Hot Swing",
            "Victoria Latin Dance Association",
            "Victoria West Coast Swing Collective"
        }
        # Get sources present in group_df that are important and have less than threshold events.
        underperforming = group_df[
            (group_df['source'].isin(important_sources)) & (group_df['counted'] < threshold)
        ]['source'].tolist()

        # Also, for any important source not present in group_df at all, add them.
        for src in important_sources:
            if src not in group_df['source'].tolist():
                underperforming.append(src)

        subset_df = calendar_df[calendar_df['source'].isin(underperforming)]
        logging.info(f"def subset_calendar_urls(): Found {len(underperforming)} important sources under threshold or missing: {underperforming}")
        return subset_df


    def move_existing_urls_to_temp(self, urls_dir):
        temp_dir = os.path.join(urls_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        for filename in os.listdir(urls_dir):
            if filename.endswith(".csv"):
                src = os.path.join(urls_dir, filename)
                dst = os.path.join(temp_dir, filename)
                shutil.move(src, dst)
                logging.info(f"def move_existing_urls_to_temp(): Moved {filename} to {temp_dir}.")
        logging.info(f"def move_existing_urls_to_temp(): Completed moving CSV files from {urls_dir} to {temp_dir}.")


    def write_calendar_urls(self, calendar_df, urls_dir):
        output_file = os.path.join(urls_dir, "calendar_urls_subset.csv")
        calendar_df.to_csv(output_file, index=False)
        logging.info(f"def write_calendar_urls(): Wrote subset calendar URLs to {output_file}.")
        return output_file
    

    def run_crawler(self):
        """
        Runs the crawler by launching the 'scraper_crawl.py' script in a separate process.
        """
        logging.info("def run_crawler(): Launching crawl subprocess.")
        result = subprocess.run([sys.executable, 'src/scraper_crawl.py'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"def run_crawler(): Crawl subprocess failed with error: {result.stderr}")
            raise Exception("Crawl subprocess failed.")
        else:
            logging.info(f"def run_crawler(): Crawl subprocess completed successfully: {result.stdout}")


    def scraper_check_and_rerun(self, max_attempts=4, event_threshold=100):
        """
        Attempts to run the web scraper multiple times to ensure sufficient events are captured for important sources.
        Parameters:
        max_attempts (int): The maximum number of attempts to run the scraper. Default is 4.
        event_threshold (int): The minimum number of events required for each important source. Default is 100.
        Returns:
        bool: True if sufficient events are captured for all important sources within the maximum attempts, False otherwise.
        The function performs the following steps:
        1. Loads calendar URLs from the configuration file.
        2. Defines a set of important sources.
        3. Runs the scraper up to `max_attempts` times or until sufficient events are captured.
        4. Checks the number of events captured for each important source.
        5. If any important source has insufficient events, prepares a subset of URLs for re-running the scraper.
        6. Updates the configuration settings for the re-run.
        7. Moves existing URLs to a temporary location and writes the subset of URLs for the next run.
        8. Logs the process and results.
        If the scraper fails to capture sufficient events after the maximum attempts, an error is logged.
        """

        calendar_urls_file = self.config['input']['calendar_urls']
        calendar_df = pd.read_csv(calendar_urls_file)
        logging.info(f"def scraper_check_and_rerun(): Loaded calendar URLs from {calendar_urls_file}.")

        # Define important sources.
        important_sources = {
            "Salsa Caliente",
            "Red Hot Swing",
            "Victoria Latin Dance Association",
            "Victoria West Coast Swing Collective"
        }

        attempt = 1
        success = False

        while attempt <= max_attempts and not success:
            logging.info(f"def scraper_check_and_rerun(): Attempt {attempt} starting crawler process.")
            self.run_crawler()

            group_df = db_handler.groupby_source()
            logging.info(f"def scraper_check_and_rerun(): Groupby results:\n{group_df}")

            # Extract only the rows for important sources.
            important_df = group_df[group_df['source'].isin(important_sources)]
            missing_important = important_sources - set(important_df['source'].tolist())
            under_threshold = [src for src in important_df['source'] 
                            if important_df.loc[important_df['source'] == src, 'counted'].iloc[0] < event_threshold]

            if not missing_important and not under_threshold:
                logging.info("def scraper_check_and_rerun(): All important sources have sufficient events.")
                success = True
            else:
                logging.info(f"def scraper_check_and_rerun(): Insufficient events for important sources. "
                            f"Missing: {missing_important}, Under threshold: {under_threshold}. Preparing to re-run.")
                subset_df = self.subset_calendar_urls(calendar_df, group_df, threshold=event_threshold)
                if subset_df.empty:
                    logging.info("def scraper_check_and_rerun(): No URLs to re-run; exiting re-run loop.")
                    break

                self.config['startup']['use_db'] = False
                logging.info("def scraper_check_and_rerun(): Set config['startup']['use_db'] to False.")

                self.config['crawling'].update({
                    'depth_limit': 3,
                    'headless': True,
                    'max_crawl_trys': 3,
                    'max_website_urls': 5,
                    'scroll_depth': 3,
                    'urls_run_limit': 5 * len(subset_df)
                })
                logging.info("def scraper_check_and_rerun(): Updated crawling settings based on subset size.")

                urls_dir = self.config['input']['urls']
                self.move_existing_urls_to_temp(urls_dir)
                self.write_calendar_urls(subset_df, urls_dir)
                attempt += 1

        if not success:
            logging.error("def scraper_check_and_rerun(): Failed to capture sufficient events for important sources after maximum attempts.")
        else:
            logging.info("def scraper_check_and_rerun(): Successfully captured sufficient events for important sources.")
        return success


    def restore_temp_files(self):
        """
        Moves any CSV files from the temp directory back to the original URLs directory.
        """
        urls_dir = self.config['input']['urls']
        temp_dir = os.path.join(urls_dir, "temp")
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                src = os.path.join(temp_dir, filename)
                dst = os.path.join(urls_dir, filename)
                try:
                    shutil.move(src, dst)
                    logging.info(f"def restore_temp_files(): Moved {filename} from temp back to {urls_dir}.")
                except Exception as e:
                    logging.error(f"def restore_temp_files(): Failed to move {filename} back - {e}")
            logging.info("def restore_temp_files(): Completed restoring temp files.")
        else:
            logging.info("def restore_temp_files(): No temp directory found; nothing to restore.")


# --------------------------------------------------
# Main Block
# --------------------------------------------------
if __name__ == "__main__":
    start_time = datetime.now()

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Instantiate required handlers.
    db_handler = DatabaseHandler(config)
    llm_handler = LLMHandler(config_path="config/config.yaml")
    read_extract = ReadExtract("config/config.yaml")
    file_name = os.path.basename(__file__)
    start_df = db_handler.count_events_urls_start(file_name)

    # Let the manager handle the crawl.
    manager = ScraperManager(config, db_handler)
    success = manager.scraper_check_and_rerun(config['crawling']['max_attempts'], event_threshold=100)

    if success:
        manager.restore_temp_files()
    else:
        logging.error("__main__: Scraper did not run successfully; temp files were not restored.")

    logging.info("__main__: Crawler process completed.")
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
