import os
import yaml
import logging
import pandas as pd
from googleapiclient.discovery import build

# Initial basic logging configuration for early logs (console output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GoogleSearch:
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize GoogleSearch by loading configuration from a YAML file,
        setting up logging based on config, and retrieving Google API credentials.

        Args:
            config_path (str): Path to the configuration YAML file.
                               Defaults to "config/config.yaml".
        """
        self.config = self.load_config(config_path)
        self.setup_logging()  # Set up file logging based on configuration
        logging.info("def __init__(): Configuration loaded successfully.")

        # Retrieve and store API credentials once during initialization
        self.api_key, self.cse_id = self.get_keys('Google')

    def setup_logging(self):
        """
        Set up logging file handler based on configuration.
        """
        log_file = self.config.get('logging', {}).get('file', 'gs_log.txt')
        # Create file handler if not already set
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        # Add file handler to the root logger
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

    def load_config(self, config_path):
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist at the given path.
        """
        if not os.path.exists(config_path):
            logging.error(f"Configuration file not found at {config_path}")
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logging.debug(f"Loaded config: {config}")
            return config

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

    def google_search(self, query, num_results=10):
        """
        Perform a Google search using the Custom Search API and retrieve title, URL, and snippet.

        Args:
            query (str): The search query string.
            num_results (int, optional): The number of search results to retrieve. Defaults to 10.

        Returns:
            list: A list of dictionaries containing 'title', 'url', and 'snippet' for each search result.

        Raises:
            googleapiclient.errors.HttpError: If an error occurs during the API request.
        """
        logging.info(f"Performing Google search for query: {query}")
        service = build("customsearch", "v1", developerKey=self.api_key)
        response = service.cse().list(
            q=query,
            cx=self.cse_id,
            num=num_results
        ).execute()

        results = []
        if 'items' in response:
            for item in response['items']:
                title = item.get('title')
                url = item.get('link')
                snippet = item.get('snippet')
                results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
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
            results = self.google_search(query)
            all_results.extend(results)
        logging.info(f"Driver completed with total {len(all_results)} results.")

        # Create a dataframe from the results
        results_df = pd.DataFrame(all_results)
        return results_df

# Example usage:
if __name__ == "__main__":
    gs_instance = GoogleSearch()
    results_df = gs_instance.driver()

    # Write to a CSV file so it's readable in Excel
    output_path = gs_instance.config['output']['gs_search_results']
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results written to {output_path}")
