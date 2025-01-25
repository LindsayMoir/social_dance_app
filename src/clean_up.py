"""
clean_up.py

This module defines the CleanUp class, which performs data cleanup on event records 
that do not have URLs. The process involves:

1. Retrieving events without URLs from the database.
2. For each such event, performing a Google search using the event name to find 
   a relevant URL based on fuzzy matching and predefined preferences, where URL 
   quality is ranked (best: contains 'allevents', medium: does not contain 
   'facebook' or 'instagram', worst: contains 'facebook' or 'instagram').
3. Using Playwright (via ReadExtract) to asynchronously extract text from the found URL.
4. Checking if the extracted text contains relevant keywords using LLMHandler.
5. If relevant, generating a prompt, querying an LLM, and parsing the response 
   to obtain structured event information.
6. Merging the new event data with existing database records, updating fields 
   where necessary based on content length.
7. Updating the events table with the merged data and writing the found URL 
   to the URLs table.

Dependencies:
    - pandas: For data manipulation.
    - fuzzywuzzy: For fuzzy matching of event names.
    - logging: For logging operations.
    - DatabaseHandler from db.py: For database operations.
    - GoogleSearch from gs.py: For performing Google searches.
    - LLMHandler from llm.py: For interacting with the Language Learning Model.
    - ReadExtract from rd_ext.py: For asynchronous text extraction using Playwright.
    - credentials.py: For centralized credential retrieval.
    - Other standard libraries: yaml, requests, etc.
"""

import asyncio
from datetime import datetime
import logging
import random
import pandas as pd
from fuzzywuzzy import fuzz
import yaml
from googleapiclient.discovery import build

from db import DatabaseHandler
from gs import GoogleSearch
from llm import LLMHandler
from credentials import get_credentials
from test_bed.rd_ext import ReadExtract  # Import the asynchronous extraction class

class CleanUp:
    def __init__(self, config):
        self.config = config
        self.db_handler = DatabaseHandler(config)
        self.llm_handler = LLMHandler(config_path="config/config.yaml")

        # Establish database connection
        self.conn = self.db_handler.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        logging.info("def __init__(): Database connection established.")

        # Retrieve Google API credentials using credentials.py
        _, self.api_key, self.cse_id = get_credentials(self.config, 'Google')


    async def process_events_without_url(self):
        """
        Asynchronously processes events without URLs by finding appropriate URLs, 
        extracting text, and updating event records.
        """
        query = "SELECT * FROM events WHERE url = '' OR url IS NULL"
        no_urls_df = pd.read_sql(query, self.db_handler.conn)
        logging.info(f"def process_events_without_url(): Found {no_urls_df.shape[0]} events without URLs.")

        if no_urls_df.shape[0] == 0:
            logging.info('def process_events_without_url(): All events have a URL')
            return
        
        # Optionally reduce dataframe size for testing
        if self.config['testing']['status']:
            no_urls_df = no_urls_df.head(config['testing']['nof_no_url_events']) 

        # Initialize ReadExtract once for all extractions
        read_extract = ReadExtract("config/config.yaml")
        await read_extract.init_browser()

        for event_row in no_urls_df.itertuples(index=False):
            event_name = event_row.event_name

            # Use Google search to find the best URL
            results_df = self.google_search(event_name)
            best_url = self.find_best_url_for_event(event_name, results_df)

            if best_url:
                # Use the asynchronous ReadExtract instance to extract text
                extracted_text = await read_extract.extract_event_text(best_url)
                logging.info(f"def process_events_without_url(): Extracted text from URL: {best_url}")

                org_name = event_row.org_name
                keywords_list = event_row.dance_style.split(',') if event_row.dance_style else []

                relevant = self.llm_handler.check_keywords_in_text(best_url, extracted_text, org_name, keywords_list)

                if relevant:
                    # Determine prompt type based on URL
                    prompt_type = 'fb' if 'facebook' in best_url or 'instagram' in best_url else 'default'
                    prompt = self.llm_handler.generate_prompt(best_url, extracted_text, prompt_type)
                    llm_response = self.llm_handler.query_llm(prompt, best_url)

                    if llm_response:
                        parsed_result = self.llm_handler.extract_and_parse_json(llm_response, best_url)
                        if parsed_result:
                            events_df = pd.DataFrame(parsed_result)
                            events_df['url'] = best_url

                            events_df.to_csv('debug/events_df.csv', index=False)
                            
                            new_row = self.select_best_match(event_name, events_df)
                            if new_row is not None:
                                merged_row = self.merge_rows(event_row, new_row)
                                self.update_event_row(merged_row)
                                org_names = merged_row['org_name']
                                keywords = merged_row['dance_style']
                                self.db_handler.write_url_to_db(org_names, keywords, best_url, '', True, 1)
            else:
                logging.info(f"def process_events_without_url(): No URL found for event: {event_name}")

        await read_extract.close()
        logging.info("def process_events_without_url(): Completed processing events without URLs.")

    def google_search(self, event_name):
        """
        Finds the best URL for an event using Google search based on the event name.
        """
        location = self.config['location']['epicentre']
        query = f"{event_name} {location}"

        logging.info(f"Performing Google search for query: {query}")
        service = build("customsearch", "v1", developerKey=self.api_key)
        response = service.cse().list(
            q=query,
            cx=self.cse_id,
            num=self.config['search']['gs_num_results']
        ).execute()

        results = []
        if 'items' in response:
            for item in response['items']:
                title = item.get('title')
                url = item.get('link')
                results.append({'event_name': title, 'url': url})
            logging.info(f"Found {len(results)} results for query: {query}")
        else:
            logging.info(f"No results found for query: {query}")

        return pd.DataFrame(results)

    def find_best_url_for_event(self, original_event_name, results_df):
        best_match = None
        best_score = 0
        best_rank = 0

        for row in results_df.itertuples(index=False):
            score = fuzz.ratio(original_event_name, row.event_name)
            if score < 80:
                continue

            url = str(row.url).lower()
            if 'allevents' in url:
                rank = 3
            elif 'facebook' in url or 'instagram' in url:
                rank = 1
            else:
                rank = 2

            if (score > best_score) or (score == best_score and rank > best_rank):
                best_score = score
                best_rank = rank
                best_match = row

        return best_match.url if best_match is not None else ''

    def select_best_match(self, original_event_name, events_df):
        best_match = None
        best_score = 0
        for row in events_df.itertuples(index=False):
            score = fuzz.ratio(original_event_name, row.event_name)
            if score >= 80 and score > best_score:
                best_score = score
                best_match = row
        return best_match

    def merge_rows(self, original_row, new_row):
        merged = original_row._asdict()  # Convert namedtuple to dict
        for col in original_row._fields:
            orig_val = getattr(original_row, col)
            new_val = getattr(new_row, col, None) if hasattr(new_row, col) else None
            if pd.isna(orig_val) or orig_val == '':
                merged[col] = new_val
            elif pd.notna(new_val) and new_val != '':
                if len(str(new_val)) > len(str(orig_val)):
                    merged[col] = new_val
        return merged

    def update_event_row(self, merged_row):
        event_id = merged_row['event_id']
        update_columns = [col for col in merged_row if col != 'event_id']
        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        update_params = {col: merged_row[col] for col in update_columns}
        update_params['event_id'] = event_id
        self.db_handler.execute_query(update_query, update_params)


if __name__ == "__main__":

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    start_time = datetime.now()
    logging.info(f"\n__main__: Starting the crawler process at {start_time}")

    clean_up_instance = CleanUp(config)
    asyncio.run(clean_up_instance.process_events_without_url())

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
