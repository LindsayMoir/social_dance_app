"""
gs.py: A module for performing Google searches using the Custom Search API,
processing results with a Language Learning Model (LLM), and storing results.

This module defines the `GoogleSearch` class, which:
- Loads configuration from a YAML file.
- Sets up logging based on the configuration.
- Retrieves Google API credentials and reads keywords from specified CSV files.
- Constructs search queries using keywords and location parameters.
- Performs Google searches and processes results using an LLM to extract organization names.
- Aggregates and returns search results in a Pandas DataFrame.
- Provides an entry point to run the search process and save results to a CSV file.

Dependencies:
    - googleapiclient.discovery for Google Custom Search API
    - pandas for data manipulation and CSV I/O
    - yaml for configuration file parsing
    - logging for logging events
    - LLMHandler for processing results with a language model
    - credentials.py for centralized credential retrieval
"""

from datetime import datetime
from googleapiclient.discovery import build
import logging
import os
import pandas as pd
from urllib.parse import urlparse
import yaml

# Import the DatabaseHandler and LLMHandler class
from db import DatabaseHandler
from llm import LLMHandler

from credentials import get_credentials

class GoogleSearch():
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize GoogleSearch by loading configuration from a YAML file,
        setting up logging based on config, and retrieving Google API credentials.

        Args:
            config_path (str): Path to the configuration YAML file.
                               Defaults to "config/config.yaml".
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Retrieve and store API credentials using credentials.py
        self.appid_uid, self.key_pw, self.cse_id = get_credentials('Google')

        # Instantiate LLMHandler
        self.llm_handler = LLMHandler(config_path=config_path)

        self.keywords_list = self.llm_handler.get_keywords()
    

    def build_query_string(self, keyword):
        """
        Construct a query string using keywords and location from configuration.

        Args:
            row (pd.Series): A row from a DataFrame containing a 'keywords' field.

        Returns:
            str: A query string combining the keywords and the location.
        """
        location = self.config['location']['epicentre']
        query = f"{keyword} {location}"
        logging.debug(f"def build_query_string(): Built query string: {query}")
        return query
        
    
    def relevant_dance_url(self, title, url, snippet):
        """
        Extract organization names from a title string.

        Args:
            title (str): The title string to extract organization names from.

        Returns:
            source (str): A likely organization name extracted from the title.
        """
        prompt_file_path = self.config['prompts']['relevant_dance_url']
        with open(prompt_file_path, 'r') as file:
            prompt = file.read()
        prompt = f"{prompt}\nTitle: {title}\nURL: {url}\nSnippet: {snippet}"
        
        relevant = self.llm_handler.query_llm(url, prompt)
        if relevant and relevant.lower() == 'True'.lower():
            logging.info(f"def relevant_dance_url(): LLM's opinion on the relevance of this URL: {url} is: {relevant}")
            return True
        else:
            logging.info(f"def relevant_dance_url(): LLM's opinion on the relevance of this URL: {url} is: {relevant}")
            return False
    

    def get_domain(self, url: str) -> str:
        """
        Extracts the domain from a given URL.

        Args:
            url: The full URL string.
        Returns:
            The domain (e.g. "example.com").
        """
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove 'www.' prefix, if you don't want it
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    

    def google_search(self, query, keyword, num_results=10):
        """
        Perform a Google search using the Custom Search API and retrieve title, URL, and snippet.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of search results to retrieve. Defaults to 10.

        Returns:
            list: A list of dictionaries containing 'source', 'keywords', and 'link' for each search result.
        """
        logging.info(f"Performing Google search for query: {query}")
        service = build("customsearch", "v1", developerKey=self.key_pw)
        response = service.cse().list(
            q=query,
            cx=self.cse_id,
            num=self.config['search']['gs_num_results']
        ).execute()

        results = []
        if 'items' in response:
            for item in response['items']:
                title = item.get('title', 'No Title')
                url = item.get('link', 'No URL')
                snippet = item.get('snippet', 'No Snippet')

                logging.info(f"google_search(): Title: {title}, URL: {url}")

                url_domain = self.get_domain(url)

                # Skip if title is untitled/No Title OR domain is blacklisted
                if title.lower() == 'untitled' or title == 'No Title' or db_handler.avoid_domains(url_domain):
                    logging.info(f"Skipped result (bad title or blacklisted domain): {title} for query: {query}")
                    continue

                # Otherwise, process it
                dance_url = self.relevant_dance_url(title, url, snippet)
                if dance_url:
                    results.append({
                        'source': title,
                        'keywords': keyword,
                        'link': url
                    })
                    logging.info(f"Added result: {title}")
                else:
                    logging.info(f"Skipped irrelevant result: {title}")
        else:
            logging.info(f"No items found in response for query: {query}")

        logging.info(f"Found {len(results)} results for query: {query}")
        return results


    def driver(self):
        """
        Reads keywords from a DataFrame, constructs search queries,
        performs Google searches for each query, and aggregates the results.

        Returns:
            pandas.DataFrame: A dataframe containing all search results.
        """
        all_results = []
        
        for keyword in self.keywords_list:

            query = self.build_query_string(keyword)
            logging.info(f"def driver: {query}")

            results = self.google_search(query, keyword)
            if results:
                all_results.extend(results)

        logging.info(f"Driver completed with total {len(all_results)} results.")
        results_df = pd.DataFrame(all_results)

        return results_df


if __name__ == "__main__":
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

     # Build log_file name
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging_file = f"logs/{script_name}_log.txt"
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")
    logging.info("gs.py starting...")

    # Initialize GoogleSearch
    gs_instance = GoogleSearch()

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before gs.py
    start_df = db_handler.count_events_urls_start(file_name)

    results_df = gs_instance.driver()

    output_path = gs_instance.config['input']['gs_urls']
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results written to {output_path}")

    # Count events and urls after gs.py
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
