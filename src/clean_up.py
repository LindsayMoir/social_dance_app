"""
clean_up.py

This module defines the CleanUp class, which performs data cleanup on event records 
that do not have URLs. The process involves:

1. Retrieving events without URLs from the database.
2. For each such event, performing a Google search using the event name to find 
   a relevant URL based on fuzzy matching and predefined preferences, where URL 
   quality is ranked (best: contains 'allevents', medium: does not contain 
   'facebook' or 'instagram', worst: contains 'facebook' or 'instagram').
3. Using Playwright to extract text from the found URL.
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
    - FacebookEventScraper from fb.py: For extracting text from Facebook events.
    - GoogleSearch from gs.py: For performing Google searches.
    - LLMHandler from llm.py: For interacting with the Language Learning Model.
    - EventSpider from scraper.py: For extracting text using Playwright.
    - Other standard libraries: yaml, requests, etc.

The CleanUp class is designed to be instantiated with a configuration object. 
Its main method, `process_events_without_url`, orchestrates the cleanup process 
as outlined above, using fuzzy matching and URL ranking to select the best 
URL and update event records accordingly.
"""

from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import logging
import pandas as pd
import yaml

from db import DatabaseHandler
from fb import FacebookEventScraper
from gs import GoogleSearch
from llm import LLMHandler
from scraper import EventSpider


class CleanUp:
    def __init__(self, config):
        self.config = config
        self.db_handler = DatabaseHandler(config)
        self.fb_handler = FacebookEventScraper(config_path="config/config.yaml")
        self.gs_instance = GoogleSearch(config_path="config/config.yaml")
        self.llm_handler = LLMHandler(config_path="config/config.yaml")
        self.es_instance = EventSpider()

        # Establish database connection
        self.conn = self.db_handler.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        logging.info("def __init__(): Database connection established.")
    

    def process_events_without_url(self):
        """
        Process events without URLs by finding appropriate URLs, extracting text, 
        and updating event records.
        """
        query = "SELECT * FROM events WHERE url = '' OR url IS NULL"
        no_urls_df = pd.read_sql(query, self.db_handler.conn)
        logging.info(f"def process_events_without_url(): Found {no_urls_df.shape[0]} events without URLs.")

        if no_urls_df.shape[0] == 0:
            logging.info('def process_events_without_url(): All events have an url')
            return
        
        # Reduce the size of the dataframe for testing
        no_urls_df = no_urls_df.head(5)

        for event_row in no_urls_df.itertuples(index=False):
            event_name = event_row.event_name

            # Placeholder: Implement Google search to find best URL
            results_df = self.google_search(event_name)
            best_url = self.find_best_url_for_event(event_name, results_df)

            if best_url:
                # Choose extraction method based on url type
                # if 'facebook' in best_url or 'instagram' in best_url:
                #     extracted_text = self.fb_handler.extract_event_text(best_url)
                #     logging.info(f"def process_events_without_url(): Extracted text from Facebook event: {best_url}")
                #else:
                # We have a problem with playwright arguing with playwright facebook. Lets just hope we do not get any facebook events for now.
                extracted_text = self.es_instance.extract_text_with_playwright(best_url)
                logging.info(f"def process_events_without_url(): Extracted text from URL: {best_url}")

                org_name = event_row.org_name
                keywords_list = event_row.dance.split(',') if event_row.dance_style else []

                relevant = self.llm_handler.check_keywords_in_text(best_url, extracted_text, org_name, keywords_list)

                if relevant:
                    prompt = self.llm_handler.generate_prompt(best_url, extracted_text, prompt)
                    llm_response = self.llm_handler.query_llm(prompt, best_url)

                    if llm_response:
                        parsed_result = self.llm_handler.extract_and_parse_json(llm_response, best_url)
                        if parsed_result:
                            events_df = pd.DataFrame(parsed_result)
                            new_row = self.select_best_match(event_name, events_df)
                            if new_row is not None:
                                merged_row = self.merge_rows(event_row, new_row)
                                self.update_event_row(merged_row)

                                org_names = merged_row.org_name
                                keywords = merged_row.dance_style
                                self.db_handler.write_url_to_db(org_names, keywords, best_url, '', True, 1)

            else:
                logging.info(f"def process_events_without_url(): No URL found for event: {event_name}")


    def google_search(self, event_name):
        """
        Placeholder for finding the best URL for an event using Google search.
        """
        # Implement Google search logic and URL selection preferences here.
        # Retrieve and store API credentials once during initialization
        api_key, cse_id = self.gs_instance.get_keys('Google')

        # Do the search

        # Create query string
        location = self.config['location']['epicentre']
        query = f"{event_name} {location}"

        # Perform Google search
        logging.info(f"Performing Google search for query: {query}")
        service = build("customsearch", "v1", developerKey=api_key)
        response = service.cse().list(
            q=query,
            cx=cse_id,
            num=config['search']['gs_num_results']
        ).execute()

        # Pull out title (event_name) and url from search results
        results = []
        if 'items' in response:
            for item in response['items']:
                title = item.get('title')
                url = item.get('link')
                results.append({
                    'event_name': title,
                    'url': url
                })
            logging.info(f"Found {len(results)} results for query: {query}")
        else:
            logging.info(f"No results found for query: {query}")

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)

        return results_df


    def find_best_url_for_event(self, original_event_name, results_df):
        """
        From multiple rows in results_df, select one event that fuzzy matches original_event_name and return its URL.
        with score >= 80, prioritizing by URL type:
        - Best: URL contains 'allevents'
        - Medium: URL does not contain 'facebook' or 'instagram'
        - Worst: URL contains 'facebook' or 'instagram'
        """
        best_match = None
        best_score = 0
        best_rank = 0  # Higher rank is better

        for row in results_df.itertuples(index=False):
            # Calculate fuzzy match score
            score = fuzz.ratio(original_event_name, row.event_name)
            if score < 80:
                continue

            # Determine URL rank
            url = str(row.url).lower()  # Ensure URL is a string and lowercase for comparison
            if 'allevents' in url:
                rank = 3
            elif 'facebook' in url or 'instagram' in url:
                rank = 1
            else:
                rank = 2

            # Compare current row with best_match using fuzzy score and URL rank
            if (score > best_score) or (score == best_score and rank > best_rank):
                best_score = score
                best_rank = rank
                best_match = row

        # Get the best_url from the row
        best_url = best_match.url if best_match is not None else ''
        return best_url
    
    
    def select_best_match(self, original_event_name, events_df):
        """
        From multiple rows in events_df, select one row that fuzzy matches original_event_name 
        with a score >= 80.
        """
        best_match = None
        best_score = 0
        for row in events_df.itertuples(index=False):
            score = fuzz.ratio(original_event_name, row.event_name)
            if score >= 80 and score > best_score:
                best_score = score
                best_match = row

        return best_match


    def merge_rows(self, original_row, new_row):
        """
        Merge original_row with new_row based on the specified criteria:
        - For each column, if original is empty, use new_row.
        - If both have values, keep the one with more characters.
        """
        merged = original_row.copy()
        for col in original_row.index:
            orig_val = original_row[col]
            new_val = new_row[col] if col in new_row.index else None
            if pd.isna(orig_val) or orig_val == '':
                merged[col] = new_val
            elif pd.notna(new_val) and new_val != '':
                if len(str(new_val)) > len(str(orig_val)):
                    merged[col] = new_val
        return merged
    

    def update_event_row(self, merged_row):
        """
        Update the event row in the database with data from merged_row.
        """
        event_id = merged_row['event_id']
        update_columns = [col for col in merged_row.index if col != 'event_id']
        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        update_params = {col: merged_row[col] for col in update_columns}
        update_params['event_id'] = event_id
        self.db_handler.execute_query(update_query, update_params)


if __name__ == "__main__":

    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Configure logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Instantiate CleanUp and process events without URLs
    clean_up_instance = CleanUp(config)
    clean_up_instance.process_events_without_url()
