# irrelevant_rows_llm.py
"""
IrrelevantRowsHandler is a class responsible for handling the deleting of irrelevant events in the database. 
It initializes with a configuration file, sets up logging, connects to the database, and interfaces with the llm APIs.
    Attributes:
        config (dict): Configuration settings loaded from a YAML file.
        db_handler (DatabaseHandler): Handler for database operations.
        db_conn_str (str): Database connection string.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
        api_key (str): API key for llm.
        model (str): Model name for llm API.
        client (llm): llm API client.
        prompt_template (str): Template for the irrelevant rows prompt.
    Methods:
        _load_config(config_path): Loads the configuration from a YAML file.
        _setup_logging(): Configures logging settings.
        load_prompt(): Loads the prompt template.
        query_llm(df): Queries the llm API with event data to identify irrelevants.
        process_rows(): Processes and deletes irrelevant events based on llm API response.
        delete_irrelevant_rows(df): Deletes irrelevant events from the database.
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


class IrrelevantRowsHandler:
    def __init__(self, config_path='config/config.yaml'):
        """
        Initializes the IrrelevantRowsHandler with configuration, database, and API connections.
        """
        self._load_config(config_path)
        self._setup_logging()
        self.db_handler = DatabaseHandler(self.config)
        self.llm_handler = LLMHandler(config_path)

        # Get the file name of the code that is running
        self.file_name = os.path.basename(__file__)

        # Count events and urls before irrelevant_rows.py starts
        self.start_df = self.db_handler.count_events_urls_start(self.file_name)

    
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
        - Log file path: specified by `self.config['logging']['log_file_p2']`
        - File mode: 'w' (write mode)
        - Log level: INFO
        - Log format: "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        - Date format: '%Y-%m-%d %H:%M:%S'

        Logs an informational message with the current date and time when logging is configured.
        """
         # Build log_file name
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        logging_file = f"logs/{script_name}_log.txt" 
        logging.basicConfig(
            filename=logging_file,
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        logging.info("\n\nirrelevant_rows.py starting...")
        logging.info(f"def _setup_logging(): Logging configured and run started at time: {datetime.now()}")

    
    def load_prompt(self, chunk):
        """
        Loads the prompt template from a file specified in the configuration.

        This method reads the content of the file path provided in the configuration
        under the 'prompts' section with the key 'irrelevant_rows' and assigns it to the 
        `prompt_template` attribute.

        Raises:
            FileNotFoundError: If the file specified in the configuration does not exist.
            KeyError: If the configuration does not contain the required keys.
        """
        prompt_config = self.config['prompts']['irrelevant_rows']
        if isinstance(prompt_config, dict):
            prompt_file = prompt_config['file']
        else:
            # Backward compatibility with old string format
            prompt_file = prompt_config
        
        with open(prompt_file, "r") as file:
            prompt_template = file.read()

        prompt = f"{prompt_template}\n{chunk}"
        # logging.info(f"def load_prompt(): prompt is: \n{prompt}")

        return prompt
    

    def get_events(self):
        """Fetches possible irrelevant events from the database based on start and end times.

        This method uses a SQL query to identify events that have similar start and end times,
        grouping them into potential irrelevants. The query performs the following steps:
        1. Identifies events with start times and end times grouped into 15-minute intervals.
        2. Filters out groups that have more than one event.
        3. Assigns a group ID to each event and ranks them within their group.

        Returns:
            pd.DataFrame: A DataFrame containing the unique events identified as potential irrelevants,
                          with columns for group ID, event details, and other relevant information.
        """
        sql = """
            SELECT * FROM events;
            """
        df = pd.read_sql(text(sql), self.db_handler.get_db_connection())
        logging.info(f"def get_events(): Read {len(df)} rows from the database")
        return df


    def process_rows(self):
        """
        Processes irrelevant entries in chunks by:
        1. Fetching irrelevant entries.
        2. Filtering groups with fewer than 2 rows.
        3. Processing in chunks of 50 rows.
        4. Querying LLM for irrelevant rows.
        5. Merging responses with the dataset.
        6. Saving results and deleting irrelevants.
        """
        df = self.get_events()

        if df.empty:
            logging.warning("def process_rows(): No valid irrelevants found. Exiting.")
            return

        response_dfs = self.process_in_chunks(df, chunk_size=50)

        self.merge_and_save_results(df, response_dfs)


    def process_in_chunks(self, df, chunk_size=50):
        """
        Processes the dataset in chunks and queries the LLM.

        Args:
            df (DataFrame): The filtered dataset containing irrelevant groups.
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
        Queries the LLM with a batch of irrelevant records and returns a DataFrame.

        Args:
            chunk (DataFrame): A batch of irrelevant events to process.
            chunk_index (int): The index of the current chunk.

        Returns:
            DataFrame or None: A DataFrame with the LLM response, or None if the response is invalid.
        """
        try:
            prompt = self.load_prompt(chunk.to_json(orient="records"))
            # logging.info(f"def process_chunk_with_llm(): Prompt for chunk {chunk_index}:\n{prompt}")

            response_chunk = self.llm_handler.query_llm('', prompt)

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
            required_columns = {"event_id", "Label", "event_type_new"}
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
        Merges the LLM responses with the dataset, saves the results, and deletes irrelevants.

        Args:
            df (DataFrame): The original dataset with irrelevant records.
            response_dfs (list): A list of DataFrames containing LLM responses.

        Returns:
            None
        """
        if not response_dfs:
            logging.warning("def merge_and_save_results(): No valid responses from LLM. Skipping classifying irrelevant rows.")
            return

        response_df = pd.concat(response_dfs, ignore_index=True)
        df = df.merge(response_df, on="event_id", how="left")

        df.to_csv(self.config['output']['irrelevant_rows'], index=False)
        logging.info(f"def merge_and_save_results(): Saved {len(df)} rows to {self.config['output']['irrelevant_rows']}.")
        self.delete_irrelevant_rows(df)
        self.update_dance_styles(df)


    def update_dance_styles(self, df):
        """
        Updates the dance styles of the relevant events in the
        database based on the LLM API response.

        Args:
            df (pandas.DataFrame): A DataFrame containing event data, where the "event_id" column
                       contains the unique identifiers for the events, and the "event_type_new"
                       column contains the predicted dance styles.

        Returns:
            None
        """
        logging.info(f"update_dance_styles(): Updating dance styles for {df.shape[0]} rows in the database.")

        # Drop existing event_type column
        df.drop(columns=["event_type"], inplace=True)

        # Rename column event_type_new to event_type
        df.rename(columns={"event_type_new": "event_type"}, inplace=True)

        # Remove NaN values from the event_type column
        before_dropna = df.shape[0]
        df = df.dropna(subset=["event_type"])
        after_dropna = df.shape[0]
        if before_dropna != after_dropna:
            logging.warning(f"update_dance_styles(): before_dropna {before_dropna} rows after_dropna {after_dropna}.")

        # Prepare data for updating rows in the database
        for idx, row in df.iterrows():
            update_query = """
            UPDATE events
            SET event_type = :event_type
            WHERE event_id = :event_id
            """
            params = {
            "event_type": row["event_type"],
            "event_id": row["event_id"]
            }
            self.db_handler.execute_query(update_query, params)
        logging.info(f"update_dance_styles(): Updated {idx} rows (dance_style) in the database.")


    def delete_irrelevant_rows(self, df):
        """
        Deletes irrelevant events from the database based on the provided DataFrame.

        This method identifies events marked with a "Label" of 1 in the given DataFrame
        and deletes them from the database in batches.

        Args:
            df (pandas.DataFrame): A DataFrame containing event data, where the "Label" column
                       indicates irrelevant events with a value of 1, and the "event_id"
                       column contains the unique identifiers for the events.

        Returns:
            None
        """
        to_be_deleted_event_list = df.loc[df["Label"] == 1, "event_id"].tolist()
        logging.info(f"def delete_irrelevant_rows(): Number of event_id(s) to be deleted is: {len(to_be_deleted_event_list)}")  
        if to_be_deleted_event_list:
            if self.db_handler.delete_multiple_events(to_be_deleted_event_list):
                logging.info("def delete_irrelevant_rows(): All selected events successfully deleted.")
            else:
                logging.error("def delete_irrelevant_rows(): Some selected events could not be deleted.")
        else:
            logging.info("def delete_irrelevant_rows(): No events marked for deletion.")

        # Count events and urls after irrelevant_rows.py
        self.db_handler.count_events_urls_end(self.start_df, self.file_name)
        return None


if __name__ == "__main__":

    start_time = datetime.now()

    # Instantiate class libraries
    irrelevant = IrrelevantRowsHandler('config/config.yaml')

    # Process irrelevant rows
    irrelevant.process_rows()

    end_time = datetime.now()
    logging.info(f"__main__: Finished the crawler process at {end_time}")
    total_time = end_time - start_time
    logging.info(f"__main__: Total time taken: {total_time}\n\n")
