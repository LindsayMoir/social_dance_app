# dedup_llm.py
"""
DeduplicationHandler is a class responsible for handling the deduplication of events in a database. It initializes with a configuration file, sets up logging, connects to the database, and interfaces with the Mistral API for deduplication tasks.
    Attributes:
        config (dict): Configuration settings loaded from a YAML file.
        db_handler (DatabaseHandler): Handler for database operations.
        db_conn_str (str): Database connection string.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
        api_key (str): API key for Mistral.
        model (str): Model name for Mistral API.
        client (Mistral): Mistral API client.
        prompt_template (str): Template for the deduplication prompt.
    Methods:
        _load_config(config_path): Loads the configuration from a YAML file.
        _setup_logging(): Configures logging settings.
        _setup_database(): Sets up the database connection.
        _setup_mistral_api(): Initializes the Mistral API client.
        _load_prompt(): Loads the deduplication prompt template.
        fetch_possible_duplicates(): Fetches possible duplicate events from the database.
        query_mistral(df): Queries the Mistral API with event data to identify duplicates.
        process_duplicates(): Processes and deletes duplicate events based on Mistral API response.
        delete_duplicates(df): Deletes duplicate events from the database.
"""

import json
import logging
import os
import pandas as pd
import yaml
from datetime import datetime
from sqlalchemy import create_engine, text
from mistralai import Mistral
from dotenv import load_dotenv
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
        self._setup_database()
        self._setup_mistral_api()
        self._load_prompt()
    
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


    def _setup_mistral_api(self):
        """
        Sets up the Mistral API client.

        This method initializes the Mistral API client by retrieving the API key from
        the environment variables and setting up the client with the specified model.

        Attributes:
            api_key (str): The API key for authenticating with the Mistral API.
            model (str): The model name to be used with the Mistral API.
            client (Mistral): The Mistral API client instance.

        Logs:
            Info: Logs the creation of the Mistral client.
        """
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.model = "mistral-large-latest"
        self.client = Mistral(api_key=self.api_key)
        logging.info("def _setup_mistral_api(): Mistral client created")
    
    def _load_prompt(self):
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
            self.prompt_template = file.read()
    

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
    
    def query_mistral(self, df):
        """
        Queries the Mistral model with a given DataFrame and returns the response as a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be included in the prompt.

        Returns:
            pd.DataFrame: The DataFrame constructed from the JSON response of the Mistral model.

        Raises:
            ValueError: If the response from the Mistral model does not contain valid JSON.

        Logs:
            The method logs the generated prompt, the extracted JSON response, and the number of rows read from the response.
        """
        prompt = f"{self.prompt_template}\n\n{df.to_markdown()}"
        logging.info(f"def query_mistral(): Prompt: \n{prompt}")

        chat_response = self.client.chat.complete(model=self.model, messages=[{"role": "user", "content": prompt}])
        response_text = chat_response.choices[0].message.content

        start, end = response_text.find("["), response_text.rfind("]") + 1
        response_json = response_text[start:end]
        logging.info(f"def query_mistral(): Extracted JSON: {response_json}")

        response_df = pd.DataFrame(json.loads(response_json))
        logging.info(f"def query_mistral(): Read {len(response_df)} rows from the response")

        return response_df
    
    
    def process_duplicates(self):
        """
        Processes duplicate entries in the dataset in chunks.

        This method performs the following steps:
        1. Fetches possible duplicate entries from the dataset.
        2. Filters out groups with fewer than 2 rows.
        3. Splits the dataset into chunks of 50 rows.
        4. Queries the Mistral service in batches.
        5. Merges the responses with the original dataset.
        6. Saves the merged dataset to a CSV file.
        7. Deletes the identified duplicates.

        Returns:
            None
        """
        df = self.fetch_possible_duplicates()

        # Make sure that there are at least 2 rows in every group_id.
        # If not, remove the group_id from the dataframe and the rows for that group_id from the dataframe.
        df = df.groupby('group_id').filter(lambda x: len(x) > 1)
        logging.info(f"def process_duplicates(): Number of rows after filtering: {len(df)}")

        chunk_size = 50
        response_dfs = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            response_chunk = self.query_mistral(chunk)
            response_dfs.append(response_chunk)

        # Merge all responses
        response_df = pd.concat(response_dfs, ignore_index=True)
        df = df.merge(response_df, on="event_id", how="left")
        
        # Save results and delete duplicates
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


# Example usage
if __name__ == "__main__":
    deduper = DeduplicationHandler()
    deduper.process_duplicates()
