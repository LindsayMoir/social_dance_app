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

Classes:
    GoogleSearch: Encapsulates methods for loading configurations, performing searches,
                  processing responses with an LLM, and writing results.

Usage Example:
    if __name__ == "__main__":
        # Initialize the GoogleSearch instance and run the driver
        gs_instance = GoogleSearch()
        results_df = gs_instance.driver()
        # Write results to a CSV file
        output_path = gs_instance.config['input']['data_urls']
        results_df.to_csv(output_path, index=False)
        ...

Dependencies:
    - googleapiclient.discovery for Google Custom Search API
    - pandas for data manipulation and CSV I/O
    - yaml for configuration file parsing
    - logging for logging events
    - LLMHandler for processing results with a language model
"""

from datetime import datetime
from googleapiclient.discovery import build
import logging
import pandas as pd
import os
import yaml

# Import the LLMHandler class
from llm import LLMHandler


class GoogleSearch():
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize GoogleSearch by loading configuration from a YAML file,
        setting up logging based on config, and retrieving Google API credentials.

        Args:
            config_path (str): Path to the configuration YAML file.
                               Defaults to "config/config.yaml".
        """
        # Get config
        with open('config/config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

        # Retrieve and store API credentials once during initialization
        self.api_key, self.cse_id = self.get_keys('Google')


    def get_keys(self, organization="Google"):
        """
        Retrieve the API key and CSE ID for a given organization from a CSV file.
        """
        keys_path = self.config['input']['keys']
        keys_df = pd.read_csv(keys_path)
        logging.debug(f"Loaded keys from {keys_path}")
        keys_df = keys_df[keys_df['organization'] == organization]

        if keys_df.empty:
            logging.error(f"No keys found for organization: {organization}")
            raise ValueError(f"No keys found for organization: {organization}")

        logging.info(f"Retrieved keys for organization: {organization}")
        return keys_df.iloc[0]['key_pw'], keys_df.iloc[0]['cse_id']

    def read_keywords(self):
        """
        Reads keywords from a CSV file specified in the configuration.

        Returns:
            pandas.DataFrame: A DataFrame containing the keywords read from the CSV file.
        """
        keywords_path = self.config['input']['data_keywords']
        df = pd.read_csv(keywords_path)
        logging.info(f"Read {len(df)} keywords from {keywords_path}")
        return df

    def build_query_string(self, row):
        """
        Construct a query string using keywords and location from configuration.

        Args:
            row (pd.Series): A row from a DataFrame containing a 'keywords' field.

        Returns:
            str: A query string combining the keywords and the location.
        """
        location = self.config['location']['epicentre']
        keywords = row['keywords']
        query = f"{keywords} {location}"
        logging.debug(f"def build_query_string(): Built query string: {query}")
        return query
    
    def title_to_org_name(self, title, url):
        """
        Extract organization names from a title string.

        Args:
            title (str): The title string to extract organization names from.

        Returns:
            org_name (str): A likely organization names extracted from the title.
        """
        # Get prompt
        prompt_file_path = self.config['prompts']['title_to_org_name']
        with open(prompt_file_path, 'r') as file:
            prompt = file.read()
        prompt = prompt + title

        # Get the organization name from the title
        org_name = llm_instance.query_llm(prompt, url)
        logging.info(f"def title_to_org_name(): Organization name returned by LLM is: {org_name}")

        # Remove unecessary characters from the org_name
        org_name = org_name.translate(str.maketrans("", "", "'\"<>"))

        return org_name

    def google_search(self, query, keywords, num_results=10):
        """
        Perform a Google search using the Custom Search API and retrieve title, URL, and snippet.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of search results to retrieve. Defaults to 10.

        Returns:
            list: A list of dictionaries containing 'org_name', 'title', 'keywords', 'url', and 'snippet' for each search result.

        Raises:
            googleapiclient.errors.HttpError: If an error occurs during the API request.
        """
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
                logging.info(f"def google_search(): Title: {title}")
                url = item.get('link')
                org_name = self.title_to_org_name(title, url)
                results.append({
                    'org_names': org_name,
                    'keywords': keywords,
                    'links': url
                })
            logging.info(f"Found {len(results)} results for query: {query}")
        else:
            logging.info(f"No results found for query: {query}")
        return results

    def driver(self):
        """
        This function reads keywords from a DataFrame, constructs search queries,
        performs Google searches for each query, and aggregates the results.

        Returns:
            pandas.DataFrame: A dataframe containing all search results.

        Logs:
            Logs the total number of results obtained after completing the searches.
        """
        all_results = []
        keywords_df = self.read_keywords()
        for _, row in keywords_df.iterrows():
            query = self.build_query_string(row)
            results = self.google_search(query, row['keywords'])
            all_results.extend(results)
        logging.info(f"Driver completed with total {len(all_results)} results.")

        # Create a dataframe from the results
        results_df = pd.DataFrame(all_results)
        return results_df

# Example usage:
if __name__ == "__main__":

    # Get config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set up logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',  # Changed to append mode to preserve logs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Get the start time
    start_time = datetime.now()
    logging.info(f"\n\n__main__: Starting the crawler process at {start_time}")

    gs_instance = GoogleSearch()
    llm_instance = LLMHandler()
    results_df = gs_instance.driver()

    # Write to a CSV file so it's readable in Excel
    output_path = gs_instance.config['input']['data_urls']
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results written to {output_path}")

    # Get the end time
    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")

    # Calculate the total time taken
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
