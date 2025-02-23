# dedup_llm.py
"""
DeduplicationHandler is a class responsible for handling the deduplication of events in a database. It initializes with a configuration file, sets up logging, connects to the database, and interfaces with the llm API for deduplication tasks.
    Attributes:
        config (dict): Configuration settings loaded from a YAML file.
        db_handler (DatabaseHandler): Handler for database operations.
        db_conn_str (str): Database connection string.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
        api_key (str): API key for llm.
        model (str): Model name for llm API.
        client (llm): llm API client.
        prompt_template (str): Template for the deduplication prompt.
    Methods:
        _load_config(config_path): Loads the configuration from a YAML file.
        _setup_logging(): Configures logging settings.
        _setup_database(): Sets up the database connection.
        _setup_llm_api(): Initializes the llm API client.
        load_prompt(): Loads the deduplication prompt template.
        fetch_possible_duplicates(): Fetches possible duplicate events from the database.
        query_llm(df): Queries the llm API with event data to identify duplicates.
        process_duplicates(): Processes and deletes duplicate events based on llm API response.
        delete_duplicates(df): Deletes duplicate events from the database.
"""
from datetime import datetime
from dotenv import load_dotenv
import json
from io import StringIO
import logging
import os
import pandas as pd
import re
from sqlalchemy import create_engine, text
import yaml

from llm import LLMHandler
from db import DatabaseHandler


class DeduplicationHandler:
    def __init__(self, config_path='config/config.yaml'):
        """
        Initializes the DeduplicationHandler with configuration, database, and API connections.
        """
        self._load_config(config_path)
        self._setup_logging()
        self.db_handler = DatabaseHandler(self.config)
        self.llm_handler = LLMHandler(config_path)
        self._setup_database()
    
    def _load_config(self, config_path):
        """
        Loads configuration from a YAML file.

        Args:
            config_path (str): The path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the specified config_path does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _setup_logging(self):
        """
        Configures the logging settings for the application.

        This method sets up the logging configuration using the parameters specified
        in the `self.config` dictionary. It configures the log file, log level, log 
        format, and date format. Once configured, it logs an informational message 
        indicating that logging has been set up and the run has started.

        The logging configuration includes:
        - Log file path: specified by `self.config['logging']['log_file']`
        - File mode: 'a' (append mode)
        - Log level: INFO
        - Log format: "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        - Date format: '%Y-%m-%d %H:%M:%S'

        Logs an informational message with the current date and time when logging is configured.
        """
        logging.basicConfig(
            filename=self.config['logging']['log_file'],
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("dedup.py starting...")
        logging.info(f"def _setup_logging(): Logging configured and run started at time: {datetime.now()}")


    def _setup_database(self):
        """
        Sets up the database connection for the application.

        This method loads the environment variables, retrieves the database connection string,
        and initializes the SQLAlchemy engine with the connection string.

        Raises:
            KeyError: If the 'DATABASE_CONNECTION_STRING' environment variable is not set.
        """
        load_dotenv()
        self.db_conn_str = os.getenv("DATABASE_CONNECTION_STRING")
        self.engine = create_engine(self.db_conn_str)

    
    def load_prompt(self, chunk):
        """
        Loads the prompt template from a file specified in the configuration.

        This method reads the content of the file path provided in the configuration
        under the 'prompts' section with the key 'dedup' and assigns it to the 
        `prompt_template` attribute.

        Raises:
            FileNotFoundError: If the file specified in the configuration does not exist.
            KeyError: If the configuration does not contain the required keys.
        """
        with open(self.config['prompts']['dedup'], "r") as file:
            prompt_template = file.read()

        prompt = f"{prompt_template}\n{chunk}"
        logging.info(f"def load_prompt(): prompt is: \n{prompt}")

        return prompt
    

    def fetch_possible_duplicates(self):
        """Fetches possible duplicate events from the database based on start and end times.

        This method uses a SQL query to identify events that have similar start and end times,
        grouping them into potential duplicates. The query performs the following steps:
        1. Identifies events with start times and end times grouped into 15-minute intervals.
        2. Filters out groups that have more than one event.
        3. Assigns a group ID to each event and ranks them within their group.

        Returns:
            pd.DataFrame: A DataFrame containing the unique events identified as potential duplicates,
                          with columns for group ID, event details, and other relevant information.
        """
        sql = """
        WITH DuplicateStartTimes AS (
            SELECT start_date,
                DATE_TRUNC('minute', start_time) - 
                (EXTRACT(minute FROM start_time) % 15) * INTERVAL '1 minute' AS start_time_group,
                DATE_TRUNC('minute', end_time) - 
                (EXTRACT(minute FROM end_time) % 15) * INTERVAL '1 minute' AS end_time_group
            FROM events
            WHERE start_date > '2025-02-13'
        ),
        FilteredDuplicates AS (
            SELECT start_date, start_time_group, end_time_group,
                COUNT(*) OVER (PARTITION BY start_date, start_time_group, end_time_group) AS event_count
            FROM DuplicateStartTimes
        ),
        FinalFiltered AS (
            SELECT start_date, start_time_group, end_time_group
            FROM FilteredDuplicates
            WHERE event_count > 1
        ),
        GroupedEvents AS (
            SELECT e.*, 
                DENSE_RANK() OVER (ORDER BY e.start_date, f.start_time_group, f.end_time_group) AS Group_ID,
                ROW_NUMBER() OVER (PARTITION BY e.event_id ORDER BY e.start_date, e.start_time) AS rn
            FROM events e
            JOIN FinalFiltered f
            ON e.start_date = f.start_date
            AND DATE_TRUNC('minute', e.start_time) = f.start_time_group
            AND DATE_TRUNC('minute', e.end_time) = f.end_time_group
        )
        SELECT Group_ID, event_id, event_name, event_type, source, dance_style, url, price, location, address_id, description, time_stamp
        FROM GroupedEvents
        WHERE rn = 1
        ORDER BY Group_ID, start_date, start_time;
        """
        df = pd.read_sql(text(sql), self.engine)
        logging.info(f"def fetch_possible_duplicates(): Read {len(df)} rows from the database")
        return df


    def process_duplicates(self):
        """
        Processes duplicate entries in chunks by:
        1. Fetching duplicate entries.
        2. Filtering groups with fewer than 2 rows.
        3. Processing in chunks of 50 rows.
        4. Querying LLM for deduplication.
        5. Merging responses with the dataset.
        6. Saving results and deleting duplicates.
        """
        df = self.fetch_possible_duplicates()
        df = self.filter_valid_duplicates(df)

        if df.empty:
            logging.warning("def process_duplicates(): No valid duplicates found. Exiting.")
            return

        response_dfs = self.process_in_chunks(df, chunk_size=50)

        self.merge_and_save_results(df, response_dfs)


    def filter_valid_duplicates(self, df):
        """
        Filters out groups that have fewer than 2 rows.
        
        Args:
            df (DataFrame): DataFrame containing possible duplicate events.

        Returns:
            DataFrame: Filtered DataFrame containing only valid duplicate groups.
        """
        if df.empty:
            logging.warning("def filter_valid_duplicates(): No duplicates found.")
            return df

        df = df.groupby('group_id').filter(lambda x: len(x) > 1)

        if df.empty:
            logging.warning("def filter_valid_duplicates(): No groups have more than 1 row.")
        
        logging.info(f"def filter_valid_duplicates(): Number of rows after filtering: {len(df)}")
        return df


    def process_in_chunks(self, df, chunk_size=50):
        """
        Processes the dataset in chunks and queries the LLM.

        Args:
            df (DataFrame): The filtered dataset containing duplicate groups.
            chunk_size (int): Number of rows to process per batch.

        Returns:
            list: A list of DataFrames containing processed LLM responses.
        """
        response_dfs = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]

            if chunk.empty:
                logging.warning(f"def process_in_chunks(): Skipping empty chunk {i}.")
                continue

            response_df = self.process_chunk_with_llm(chunk, i)
            if response_df is not None:
                response_dfs.append(response_df)

        return response_dfs


    def process_chunk_with_llm(self, chunk, chunk_index):
        """
        Queries the LLM with a batch of duplicate records and returns a DataFrame.

        Args:
            chunk (DataFrame): A batch of duplicate events to process.
            chunk_index (int): The index of the current chunk.

        Returns:
            DataFrame or None: A DataFrame with the LLM response, or None if the response is invalid.
        """
        try:
            prompt = self.load_prompt(chunk.to_markdown())
            logging.info(f"def process_chunk_with_llm(): Prompt for chunk {chunk_index}:\n{prompt}")

            response_chunk = self.llm_handler.query_llm(prompt)

            if not response_chunk:
                logging.warning(f"def process_chunk_with_llm(): Received empty response for chunk {chunk_index}.")
                return None

            # Convert response to DataFrame
            return self.parse_llm_response(response_chunk)

        except Exception as e:
            logging.error(f"def process_chunk_with_llm(): Error processing chunk {chunk_index}: {e}")
            return None


    def parse_llm_response(self, response_chunk):
        """
        Extracts the structured JSON from the LLM response and converts it into a DataFrame.

        Args:
            response_chunk (str): The raw response from the LLM.

        Returns:
            pd.DataFrame: Cleaned DataFrame with extracted structured JSON data.
        """
        try:
            # Find the JSON-like block within the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_chunk, re.DOTALL)

            if not json_match:
                logging.error("def clean_response(): No valid JSON structure found in response.")
                return pd.DataFrame()  # Return empty DataFrame if no match is found

            # Extract the JSON string
            json_str = json_match.group()

            # Load JSON into a DataFrame
            df = pd.read_json(StringIO(json_str))

            # Ensure DataFrame has expected columns
            required_columns = {"group_id", "event_id", "Label"}
            if not required_columns.issubset(df.columns):
                logging.error(f"def clean_response(): Extracted JSON is missing required columns: {df.columns}")
                return pd.DataFrame()

            logging.info(f"def clean_response(): Successfully extracted {len(df)} rows from response.")
            return df

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"def clean_response(): Error parsing LLM response to JSON: {e}")
            return pd.DataFrame()  # Return empty DataFrame if parsing fails


    def merge_and_save_results(self, df, response_dfs):
        """
        Merges the LLM responses with the dataset, saves the results, and deletes duplicates.

        Args:
            df (DataFrame): The original dataset with duplicate records.
            response_dfs (list): A list of DataFrames containing LLM responses.

        Returns:
            None
        """
        if not response_dfs:
            logging.warning("def merge_and_save_results(): No valid responses from LLM. Skipping deduplication.")
            return

        response_df = pd.concat(response_dfs, ignore_index=True)
        df = df.merge(response_df, on="event_id", how="left")

        df.to_csv(self.config['output']['dedup'], index=False)
        self.delete_duplicates(df)


    def delete_duplicates(self, df):
        """
        Deletes duplicate events from the database based on the provided DataFrame.

        This method identifies events marked with a "Label" of 1 in the given DataFrame
        and deletes them from the database in batches.

        Args:
            df (pandas.DataFrame): A DataFrame containing event data, where the "Label" column
                       indicates duplicate events with a value of 1, and the "event_id"
                       column contains the unique identifiers for the events.

        Returns:
            None
        """
        to_be_deleted_event_list = df.loc[df["Label"] == 1, "event_id"].tolist()
        logging.info(f"def delete_duplicates(): Number of event_id(s) to be deleted is: {len(to_be_deleted_event_list)}")  
        if to_be_deleted_event_list:
            if self.db_handler.delete_multiple_events(to_be_deleted_event_list):
                logging.info("def delete_duplicates(): All selected events successfully deleted.")
            else:
                logging.error("def delete_duplicates(): Some selected events could not be deleted.")
        else:
            logging.info("def delete_duplicates(): No events marked for deletion.")


if __name__ == "__main__":
    deduper = DeduplicationHandler()
    deduper.process_duplicates()
