# dedup_llm.py
"""
DeduplicationHandler is a class responsible for handling the deduplication of events in a database.
It initializes with a configuration file, sets up logging, connects to the database, and interfaces
with the llm API for deduplication tasks.
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
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def _setup_logging(self):
        """
        Configures the logging settings for the application.
        """
         # Build log_file name
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        logging_file = f"/logs{script_name}_log" 
        logging.basicConfig(
            filename=logging_file,
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info("\n\ndedup_llm.py starting...")
        logging.info(f"def _setup_logging(): Logging configured and run started at time: {datetime.now()}")

    def _setup_database(self):
        """
        Sets up the database connection for the application.
        """
        load_dotenv()
        self.db_conn_str = os.getenv("DATABASE_CONNECTION_STRING")
        self.engine = create_engine(self.db_conn_str)

    def load_prompt(self, chunk):
        """
        Loads the prompt template from a file specified in the configuration.
        """
        with open(self.config['prompts']['dedup'], "r") as file:
            prompt_template = file.read()

        prompt = f"{prompt_template}\n{chunk}"
        # logging.info(f"def load_prompt(): prompt loaded.")

        return prompt
    
    def fetch_possible_duplicates(self):
        """
        Fetches possible duplicate events from the database based on start and end times.
        Returns:
            pd.DataFrame: DataFrame with potential duplicates.
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
                DENSE_RANK() OVER (ORDER BY e.start_date, f.start_time_group, f.end_time_group) AS group_id,
                ROW_NUMBER() OVER (PARTITION BY e.event_id ORDER BY e.start_date, e.start_time) AS rn
            FROM events e
            JOIN FinalFiltered f
            ON e.start_date = f.start_date
            AND DATE_TRUNC('minute', e.start_time) = f.start_time_group
            AND DATE_TRUNC('minute', e.end_time) = f.end_time_group
        )
        SELECT group_id, event_id, event_name, event_type, source, dance_style, url, price, location, address_id, description, time_stamp
        FROM GroupedEvents
        WHERE rn = 1
        ORDER BY group_id, start_date, start_time;
        """
        df = pd.read_sql(text(sql), self.engine)
        logging.info(f"def fetch_possible_duplicates(): Read {len(df)} rows from the database")
        return df

    def process_duplicates(self):
        """
        Processes duplicate entries in chunks and deletes duplicates.
        Returns:
            int: The number of events deleted during this run.
        """
        df = self.fetch_possible_duplicates()
        df = self.filter_valid_duplicates(df)

        if df.empty:
            logging.warning("def process_duplicates(): No valid duplicates found. Exiting.")
            return 0

        response_dfs = self.process_in_chunks(df, chunk_size=50)

        if response_dfs:
            return self.merge_and_save_results(df, response_dfs)
        else:
            logging.warning("def process_duplicates(): No valid responses from LLM. Skipping deduplication.")
            return 0

    def filter_valid_duplicates(self, df):
        """
        Filters out groups that have fewer than 2 rows.
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
        """
        try:
            prompt = self.load_prompt(chunk.to_markdown())
            logging.info(f"def process_chunk_with_llm(): Prompt for chunk {chunk_index}.")

            response_chunk = self.llm_handler.query_llm('', prompt)

            if not response_chunk:
                logging.warning(f"def process_chunk_with_llm(): Received empty response for chunk {chunk_index}.")
                return None

            return self.parse_llm_response(response_chunk)

        except Exception as e:
            logging.error(f"def process_chunk_with_llm(): Error processing chunk {chunk_index}: {e}")
            return None

    def parse_llm_response(self, response_chunk):
        """
        Extracts the structured JSON from the LLM response and converts it into a DataFrame.
        """
        try:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_chunk, re.DOTALL)

            if not json_match:
                logging.error("def clean_response(): No valid JSON structure found in response.")
                return pd.DataFrame()

            json_str = json_match.group()
            df = pd.read_json(StringIO(json_str))

            required_columns = {"group_id", "event_id", "Label"}
            if not required_columns.issubset(df.columns):
                logging.error(f"def clean_response(): Extracted JSON is missing required columns: {df.columns}")
                return pd.DataFrame()

            logging.info(f"def clean_response(): Successfully extracted {len(df)} rows from response.")
            return df

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"def clean_response(): Error parsing LLM response to JSON: {e}")
            return pd.DataFrame()

    def merge_and_save_results(self, df, response_dfs):
        """
        Merges the LLM responses with the dataset, saves the results, and deletes duplicates.
        Returns:
            int: The number of events deleted.
        """
        response_df = pd.concat(response_dfs, ignore_index=True)
        df = df.merge(response_df, on="event_id", how="left")

        df.to_csv(self.config['output']['dedup'], index=False)
        deleted_count = self.delete_duplicates(df)
        return deleted_count

    def delete_duplicates(self, df):
        """
        Deletes duplicate events from the database based on the provided DataFrame.
        Returns:
            int: The number of events that were marked for deletion.
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
        return len(to_be_deleted_event_list)
    

    def parse_address(self):
        """
        Parse the full_address column in batches and update the corresponding columns in the address table.
        For each batch, the prompt is built by including the entire row (all fields) for every record,
        using itertuples(). The LLM is then queried once per batch, returning a JSON containing multiple address rows.
        """
        sql = "SELECT * FROM address"
        df = pd.read_sql(text(sql), self.engine)
        logging.info(f"parse_address: Read {len(df)} rows from the database")

        batch_size = 100
        num_rows = len(df)

        # Read in prompt template from file.
        with open(self.config['prompts']['address_fix'], "r") as file:
            prompt_template = file.read()

        for i in range(0, num_rows, batch_size):
            batch_df = df.iloc[i: i + batch_size]

            # Build the prompt by converting each row (using itertuples) into key: value lines.
            prompt_lines = []
            for row in batch_df.itertuples(index=False):
                row_str = ", ".join([f"{field}: {getattr(row, field)}" for field in row._fields])
                prompt_lines.append(row_str)
            prompt = f"{prompt_template}\n" + "\n".join(prompt_lines)
            logging.info(f"parse_address: Processing batch {i // batch_size + 1} with prompt:\n{prompt}")

            # Query the LLM once per batch.
            response = self.llm_handler.query_llm('', prompt)
            if not response:
                logging.error(f"parse_address: No response received for batch {i // batch_size + 1}")
                continue

            # Expect a JSON response containing multiple address rows.
            parsed_addresses = self.llm_handler.extract_and_parse_json(response, '')
            if not parsed_addresses:
                logging.error(f"parse_address: Parsing failed for batch {i // batch_size + 1}")
                continue

            # Ensure that parsed_addresses is a list.
            if not isinstance(parsed_addresses, list):
                parsed_addresses = [parsed_addresses]

            logging.info(f"parse_address: Successfully parsed {len(parsed_addresses)} addresses in batch {i // batch_size + 1}")
            self._flush_batch(parsed_addresses)


    def _flush_batch(self, batch_data):
        """
        Flush the batch data to CSV for debugging and insert the records into the database.
        """
        batch_df = pd.DataFrame(batch_data)
        batch_df.to_csv(self.config['debug']['address_fix'], mode='a', index=False)
        values = batch_df.to_dict(orient='records')
        self.db_handler.multiple_db_inserts('address', values)
        logging.info(f"_flush_batch: Successfully inserted {len(values)} records into the database.")


    def driver(self):
        """
        Main driver function for the deduplication process.
        """
        # Loop until no duplicates are flagged for deletion (i.e. total_deleted == 0)
        while True:
            total_deleted = self.process_duplicates()
            logging.info(f"Main loop: Number of events deleted in this pass: {total_deleted}")
            if total_deleted == 0:
                logging.info("No duplicates found. Exiting deduplication loop.")
                break
        
        # Parse the address data
        self.parse_address()
        logging.info("dedup.py finished.")


if __name__ == "__main__":

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the class libraries
    deduper = DeduplicationHandler()
    db_handler = DatabaseHandler(config)

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    deduper.driver()
    
    db_handler.count_events_urls_end(start_df, file_name)

