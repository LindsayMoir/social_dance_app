"""
scraper.py

The EventSpider handles URL extraction,
dynamic content extraction, and Google Calendar event processing. 

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
from scrapy_playwright.page import PageMethod
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
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                cb_kwargs={'keywords': keywords, 'source': source, 'url': url},
                # ← Tell scrapy-playwright to spin up a browser for this request
                meta={
                    "playwright": True,
                    # optional: wait for network‐idle or body to render
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "body")
                    ],
                },
            )


    def parse(self, response, keywords, source, url):
        """
        1) Render page via Playwright and get HTML text.
        2) Identify keywords in the page and run LLM to decide relevance.
        3) Record the URL in the database (with metadata).
        4) Extract <a> links and iframe/calendar URLs.
        5) Fetch Google Calendar events where found.
        6) Filter out unwanted links, record them, and follow remaining links.
        """
        # 1) Get rendered page text
        extracted_text = response.text

        # 2) Keyword & LLM logic
        found_keywords = [kw for kw in self.keywords_list if kw in extracted_text.lower()]
        relevant    = False
        parent_url  = ''
        crawl_try   = 1
        time_stamp  = datetime.now()
        # build the initial record for this URL
        url_row = [url, parent_url, source, found_keywords, relevant, crawl_try, time_stamp]

        if found_keywords:
            logging.info(f"def parse(): Found keywords for URL {url}: {found_keywords}")
            prompt     = 'default'
            llm_status = llm_handler.process_llm_response(url, parent_url, extracted_text, source, keywords, prompt)
            if llm_status:
                # mark as relevant
                url_row[4] = True
                db_handler.write_url_to_db(url_row)
                logging.info(f"def parse(): URL {url} marked as relevant (LLM positive).")
            else:
                db_handler.write_url_to_db(url_row)
                logging.info(f"def parse(): URL {url} marked as irrelevant (LLM negative).")
        else:
            db_handler.write_url_to_db(url_row)
            logging.info(f"def parse(): URL {url} marked as irrelevant (no keywords).")

        # 3) Extract all <a href> links (limit to configured maximum)
        raw_links = response.css('a::attr(href)').getall()
        page_links = [
            response.urljoin(link)
            for link in raw_links
            if link and link.startswith(("http://", "https://"))
        ][: self.config['crawling']['max_website_urls']]
        logging.info(f"def parse(): Found {len(page_links)} links on {response.url}")

        # 4) Process iframes & extract Google Calendar addresses
        iframe_links = [
            response.urljoin(link)
            for link in response.css('iframe::attr(src)').getall()
            if link.startswith(("http://", "https://"))
        ]
        calendar_emails = re.findall(
            r'"gcal"\s*:\s*"([A-Za-z0-9_.+-]+@group\.calendar\.google\.com)"',
            response.text
        )
        if iframe_links or calendar_emails:
            for cal_url in iframe_links + calendar_emails:
                self.fetch_google_calendar_events(cal_url, url, source, keywords)
            # mark the page itself as relevant if calendar events fetched
            url_row = [cal_url, url, source, found_keywords, True, crawl_try, time_stamp]
            db_handler.write_url_to_db(url_row)

        # 5) Filter unwanted links and record them
        all_links      = {url} | set(page_links)
        filtered_links = {lnk for lnk in all_links if 'facebook' not in lnk and 'instagram' not in lnk}
        unwanted_links = all_links - filtered_links
        for link in unwanted_links:
            child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)
            logging.info(f"def parse(): Recorded unwanted URL: {link}")

        # 6) Follow each remaining link with Playwright rendering
        for link in filtered_links:
            if link in self.visited_link:
                continue
            self.visited_link.add(link)

            # record the child link before crawling
            child_row = [link, url, source, found_keywords, False, 1, datetime.now()]
            db_handler.write_url_to_db(child_row)

            if len(self.visited_link) >= self.config['crawling']['urls_run_limit']:
                logging.info(
                    f"def parse(): Reached URL run limit ({self.config['crawling']['urls_run_limit']}); stopping."
                )
                break

            logging.info(f"def parse(): Crawling next URL: {link}")
            yield scrapy.Request(
                url=link,
                callback=self.parse,
                cb_kwargs={'keywords': keywords, 'source': source, 'url': link},
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_selector", "body")
                    ],
                },
            )


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
            logging.warning(f"def decode_calendar_id(): Decoded ID is not valid: {decoded}")
            return None
        except Exception as e:
            logging.warning(f"def decode_calendar_id(): Exception for {calendar_url} - {e}")
            return None
        

    def is_valid_calendar_id(self, calendar_id):
        pattern = re.compile(r'^[a-zA-Z0-9_.+-]+@(group\.calendar\.google\.com|gmail\.com)$')
        return bool(pattern.fullmatch(calendar_id))
    

    def process_calendar_id(self, calendar_id, calendar_url, url, source, keywords):
        logging.info(f"def process_calendar_id(): Processing calendar_id: {calendar_id} from {calendar_url}")
        events_df = self.get_calendar_events(calendar_id)
        if not events_df.empty:
            logging.info(f"def process_calendar_id(): Found {len(events_df)} events for calendar_id: {calendar_id}")
            db_handler.write_events_to_db(events_df, calendar_id, calendar_url, source, keywords)


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
    

    def run_crawler(self):
        """
        Runs the crawler inline via Scrapy's CrawlerProcess (no external subprocess).
        """
        logging.info("def run_crawler(): Starting crawler in-process.")
        process = CrawlerProcess(settings={
            "LOG_FILE": self.config['logging']['scraper_log_file'],
            "LOG_LEVEL": "INFO",
            "DEPTH_LIMIT": self.config['crawling']['depth_limit'],
            "FEEDS": {
                "output/output.json": {"format": "json"}
            }
        })
        process.crawl(EventSpider, config=self.config)
        process.start()  # will block until crawl finishes
        logging.info("def run_crawler(): Crawler completed successfully.")


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
    scraper = EventSpider(config)
    file_name = os.path.basename(__file__)
    start_df = db_handler.count_events_urls_start(file_name)

    # Start the crawler.
    scraper.run_crawler()
    
    logging.info("__main__: Crawler process completed.")
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
