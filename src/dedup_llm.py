# dedup_llm.py

"""
This script provides the DeduplicationHandler class and related logic for identifying and removing duplicate event records from a database. It supports multiple deduplication strategies, including LLM-based and embedding-based clustering, and manages the full deduplication workflow: configuration, logging, database access, LLM prompting, result parsing, and database updates.

Key Features:
- Loads configuration and logging settings from YAML.
- Connects to a database using SQLAlchemy.
- Fetches potential duplicate events using SQL queries.
- Uses LLMs to classify duplicates and embedding models (DBSCAN) for clustering.
- Supports address parsing and correction via LLM.
- Merges, saves, and deletes duplicates based on LLM or clustering results.
- Tracks deduplication statistics and supports manual evaluation of results.

Classes:
    DeduplicationHandler: Orchestrates deduplication, LLM interaction, clustering, and database updates.

Typical Usage:
    python dedup_llm.py

Dependencies:
    - pandas, SQLAlchemy, sentence-transformers, scikit-learn, dotenv, yaml, etc.
"""
from datetime import datetime
from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz
import json
from io import StringIO
import logging
import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sqlalchemy import create_engine, text
import subprocess
from datetime import datetime, timedelta
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
        self.llm_handler = LLMHandler(config_path)
        self.db_handler = self.llm_handler.db_handler  # Use connected handler
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
        logging_file = f"logs/{script_name}_log.txt" 
        logging.basicConfig(
            filename=logging_file,
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
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
        prompt_config = self.config['prompts']['dedup']
        if isinstance(prompt_config, dict):
            prompt_file = prompt_config['file']
        else:
            # Backward compatibility with old string format
            prompt_file = prompt_config
        
        with open(prompt_file, "r") as file:
            prompt_template = file.read()

        prompt = f"{prompt_template}\n{chunk}"
        # logging.info(f"def load_prompt(): prompt loaded.")

        return prompt
    
    def fetch_possible_duplicates(self):
        """
        Identify and retrieve potential duplicate events from the database based on similar start and end times.

        This method queries the events table to find groups of events that occur on the same date and have start and end times
        rounded to the nearest 15-minute interval. Only groups containing more than one event are considered possible duplicates.
        The method returns a pandas DataFrame with one representative event per duplicate group, including relevant event details.

        Parameters:
            None

        Returns:
            pd.DataFrame: A DataFrame containing possible duplicate events with the following columns:
                - group_id (int): Identifier for each group of potential duplicates.
                - event_id (int): Unique identifier for the event.
                - event_name (str): Name of the event.
                - event_type (str): Type or category of the event.
                - source (str): Source from which the event was obtained.
                - dance_style (str): Dance style associated with the event.
                - url (str): URL with more information about the event.
                - price (float or str): Price of the event.
                - location (str): Location of the event.
                - address_id (int): Address identifier.
                - description (str): Description of the event.
                - time_stamp (datetime): Timestamp of the event record.

        Notes:
            - Only events with start_date after '2025-02-13' are considered.
            - Events are grouped by date and 15-minute rounded start/end times.
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
        Processes duplicate entries in the dataset by fetching, filtering, and handling them in manageable chunks.
        Deletes duplicates based on responses from an LLM and saves the results.

        Inputs:
            None (operates on the instance's internal state and methods).

            int: The number of events deleted during this run. Returns 0 if no valid duplicates are found or if no valid responses are received from the LLM.

        Raises:
            None directly, but logs warnings if no duplicates are found or if no valid responses are received from the LLM.
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
        Filters the input DataFrame to retain only groups (based on 'group_id') that contain more than one row.
        Parameters:
            df (pandas.DataFrame): The input DataFrame. Must contain a 'group_id' column.
        Returns:
            pandas.DataFrame: A DataFrame containing only the rows from groups with more than one member.
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
        Processes the input DataFrame in chunks and queries the LLM for each chunk.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed in chunks.
            chunk_size (int, optional): The number of rows per chunk. Defaults to 50.

        Returns:
            list of pd.DataFrame: A list of DataFrames containing the responses from the LLM for each processed chunk.
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
        Processes a batch (chunk) of duplicate records using a Large Language Model (LLM).

        Args:
            chunk (pd.DataFrame): A DataFrame containing a batch of duplicate records to be processed.
            chunk_index (int): The index of the current chunk, used for logging and tracking.

        Returns:
            pd.DataFrame or None: A DataFrame containing the processed results from the LLM if successful,
            or None if an error occurs or the LLM response is empty.

        Raises:
            None: All exceptions are caught and logged internally.
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
        Parses a response chunk from an LLM to extract a JSON array of objects and convert it into a pandas DataFrame.
        The method searches for a JSON array structure within the input string, attempts to parse it,
        and validates that the resulting DataFrame contains the required columns: 'group_id', 'event_id', and 'Label'.
        If parsing fails or required columns are missing, an empty DataFrame is returned.
        Args:
            response_chunk (str): The raw string response from the LLM, expected to contain a JSON array.
        Returns:
            pd.DataFrame: A DataFrame containing the extracted data if successful, otherwise an empty DataFrame.
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
        Merges the provided DataFrame with a list of LLM response DataFrames on the 'event_id' column,
        saves the merged results to a CSV file, and deletes duplicate events.

        Args:
            df (pd.DataFrame): The original DataFrame containing event data.
            response_dfs (List[pd.DataFrame]): A list of DataFrames containing LLM responses to be merged.

            int: The number of duplicate events deleted after merging and saving the results.
        """
        response_df = pd.concat(response_dfs, ignore_index=True)
        df = df.merge(response_df, on="event_id", how="left")

        df.to_csv(self.config['output']['dedup'], index=False)
        deleted_count = self.delete_duplicates(df)
        return deleted_count

    def delete_duplicates(self, df):
        """
        Deletes duplicate events from the database based on the provided DataFrame.
        Args:
            df (pandas.DataFrame): A DataFrame containing event data. Duplicate events are identified by rows where the "Label" column equals 1. The "event_id" column is used to specify which events to delete.

            int: The number of events that were marked for deletion (i.e., the number of event IDs found with "Label" == 1).

        Logs:
            - The number of events marked for deletion.
            - Success or failure of the deletion operation.
            - If no events are marked for deletion.

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
        prompt_config = self.config['prompts']['address_fix']
        if isinstance(prompt_config, dict):
            prompt_file = prompt_config['file']
        else:
            # Backward compatibility with old string format
            prompt_file = prompt_config
        
        with open(prompt_file, "r") as file:
            prompt_template = file.read()

        for i in range(0, num_rows, batch_size):
            batch_df = df.iloc[i: i + batch_size]

            # Build the prompt by converting each row (using itertuples) into key: value lines.
            prompt_lines = []
            for row in batch_df.itertuples(index=False):
                row_str = ", ".join([f"{field}: {getattr(row, field)}" for field in row._fields])
                prompt_lines.append(row_str)
            prompt = f"{prompt_template}\n" + "\n".join(prompt_lines)
            logging.info(f"parse_address: Processing batch {i // batch_size + 1}")

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


    def fix_problem_events(self, dry_run=False):
        """
        Identifies and fixes events with missing or problematic location/address information.
        Groups events by (event_name, description, location) to avoid redundant processing.
        Uses LLM and regex to repair address data and update both address and event tables.
        """
        logging.info("fix_problem_events: Retrieving problematic events...")
        sql = """
            SELECT * FROM events
            WHERE 
                address_id IS NULL 
                OR address_id = 0
                OR location IS NULL 
                OR LENGTH(TRIM(location)) < 20
                OR LENGTH(COALESCE(event_name, '')) > 100
        """
        df = pd.read_sql(text(sql), self.engine)

        df.to_csv("output/test/problematic_events_raw.csv", index=False)

        if df.empty:
            logging.info("fix_problem_events: No problematic events found.")
            return

        fix_events = []
        fix_addresses = []
        processed_event_ids = set()

        os.makedirs("output/test", exist_ok=True)

        groups = df.groupby(['event_name', 'description', 'location'], dropna=False)

        grouped_df = pd.concat([group for _, group in groups])
        grouped_df.to_csv("output/test/groups.csv", index=False)

        grouped_df_ids = grouped_df['event_id'].tolist()
        missing_rows = df[~df['event_id'].isin(grouped_df_ids)]
        if not missing_rows.empty:
            logging.warning(f"{len(missing_rows)} rows missing from groupby result. Saved to output/test/missing_from_groups.csv")
            missing_rows.to_csv("output/test/missing_from_groups.csv", index=False)

        for i, ((event_name, description, location), group) in enumerate(groups):
            logging.info(f"--- Processing event group {i+1} ---")
            row = group.iloc[0]
            event_ids = [int(eid) for eid in group['event_id'].tolist()]
            processed_event_ids.update(event_ids)

            if event_name is not None and pd.notna(event_name) and len(str(event_name)) > 100:
                logging.info(f"Group {i+1}: Event name exceeds 100 characters. Calling handle_long_event_name")
                short_name = " ".join(str(event_name).split()[:5])
                self.handle_long_event_name(short_name, event_name, description, group, dry_run, fix_events, fix_addresses)
                continue

            if location and isinstance(location, str) and location.strip() != '':
                logging.info(f"Group {i+1}: Attempting fuzzy match for location '{location}'")
                matched = self.match_location_to_building(location, group, dry_run, fix_events)
                if matched:
                    logging.info(f"Group {i+1}: Matched location to known building. Skipping further processing.")
                    continue
                if self.handle_existing_location(location, group, dry_run, fix_events):
                    logging.info(f"Group {i+1}: Found existing location match. Skipping further processing.")
                    continue
                self.handle_llm_address_fix(location, group, dry_run, fix_events, fix_addresses)
                continue

            if (pd.isna(location) or (isinstance(location, str) and location.strip() == '')) and len(event_name) > 100:
                logging.info(f"Group {i+1}: Location missing and long event name. Using fallback.")
                self.handle_fallback_llm(event_name, description, group, dry_run, fix_events, fix_addresses)
                continue

        remaining_df = df[~df['event_id'].isin(processed_event_ids)].copy()
        if remaining_df.empty:
            logging.info("No remaining rows to process in match_location_to_building().")
        else:
            logging.info(f"Processing {len(remaining_df)} unmatched rows using match_location_to_building()...")
            for _, row in remaining_df.iterrows():
                location = row.get("location", "")
                if location and isinstance(location, str) and location.strip():
                    group = pd.DataFrame([row])
                    self.match_location_to_building(location, group, dry_run, fix_events)

        if dry_run:
            pd.DataFrame(fix_events).to_csv("output/test/fix_events.csv", index=False)
            pd.DataFrame(fix_addresses).to_csv("output/test/fix_address.csv", index=False)
        else:
            logging.info("fix_problem_events: Updates committed to database.")


    def match_location_to_building(self, location, group, dry_run, fix_events):
        """
        Attempts to match a given location string to a building name in the address database using fuzzy string matching.
        If a sufficiently close match is found, updates the associated events with the matched address information.
        Args:
            location (str): The location string to match against building names.
            group (pandas.DataFrame): DataFrame containing event rows to potentially update.
            dry_run (bool): If True, do not perform updates but append proposed changes to fix_events.
            fix_events (list): List to collect proposed event updates when dry_run is True.
        Returns:
            bool: True if a suitable match was found and events were updated (or proposed for update), False otherwise.
        Side Effects:
            - May update events in the database if dry_run is False.
            - Appends proposed updates to fix_events if dry_run is True.
            - Logs information about the matching process.
        """
        building_query = "SELECT address_id, building_name, full_address FROM address WHERE building_name IS NOT NULL"
        building_df = pd.read_sql(text(building_query), self.engine)
        building_list = building_df.to_dict('records')

        best_score = 0
        best_record = None
        for record in building_list:
            score = fuzz.partial_ratio(location.lower(), record['building_name'].lower())
            if score > best_score:
                best_score = score
                best_record = record

        if best_score >= 85 and best_record:
            address_id = int(best_record['address_id'])
            full_address = best_record['full_address']
            logging.info(f"match_location_to_building: Fuzzy match found for location '{location}' "
                        f"→ building_name '{best_record['building_name']}' (score {best_score}). "
                        f"Updating events with address_id {address_id} and location '{full_address}'")
            for _, row in group.iterrows():
                new_data = {"address_id": address_id, "location": full_address}
                if dry_run:
                    fix_events.append({**row.to_dict(), **new_data})
                else:
                    self.update_event_with_sql(row, new_data)
            return True
        else:
            logging.info(f"match_location_to_building: No suitable fuzzy match found for location '{location}'. Best score: {best_score}.")
        return False


    def handle_long_event_name(self, short_name, event_name, description, group, dry_run, fix_events, fix_addresses):
        """
        Handles events with long names by truncating the event name, extracting and fixing address information using an LLM,
        and updating event records accordingly.
        This method performs the following steps:
            1. Truncates the event name to the first five words.
            2. Uses an LLM to generate and fix address information based on the event description.
            3. Parses the LLM response to extract address details.
            4. Retrieves or creates an address ID from the database.
            5. Prepares updated event data with the truncated name and fixed address.
            6. Depending on the dry_run flag, either appends the changes to the provided lists or updates the database.
        Args:
            event_name (str): The original name of the event.
            description (str): The event description, used for extracting address information.
            group (pandas.DataFrame): A DataFrame containing event records to update.
            dry_run (bool): If True, changes are collected but not committed to the database.
            fix_events (list): A list to collect updated event data in dry run mode.
            fix_addresses (list): A list to collect fixed address data in dry run mode.
        Returns:
            None
        """
        logging.info(f"handle_long_event_name: Using short_name: {short_name}")
        
        prompt, schema_type = self.llm_handler.generate_prompt(
            url="address",
            extracted_text=description,
            prompt_type="address_fix"
        )
        response = self.llm_handler.query_llm("address", prompt, schema_type)
        parsed = self.llm_handler.extract_and_parse_json(response, "address", schema_type)
        
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not parsed:
            logging.warning("handle_long_event_name: Parsed result is empty or None")
            return
        
        address_id = self.db_handler.resolve_or_insert_address(parsed)
        if address_id is None:
            logging.warning("handle_long_event_name: Failed to resolve or insert address, setting address_id = 0")
            address_id = 0
        else:
            address_id = int(address_id)
        logging.info(f"handle_long_event_name: Generated address_id = {address_id}")
        
        for _, row in group.iterrows():
            new_data = {
                "event_name": short_name,
                "address_id": address_id,
                "location": parsed.get("full_address")
            }
            logging.info(f"handle_long_event_name: Preparing update for event_id {row['event_id']} with {new_data}")
            if dry_run:
                fix_events.append({**row.to_dict(), **new_data})
                fix_addresses.append(parsed)
            else:
                self.update_event_with_sql(row, new_data)


    def handle_existing_location(self, location, group, dry_run, fix_events):
        """
        Handles events associated with an existing location by updating their address IDs.
        This method checks if the given location exists and retrieves its corresponding address ID.
        For each event in the provided group, it prepares to update the event's address ID.
        If `dry_run` is True, the updates are appended to the `fix_events` list for review without making changes to the database.
        Otherwise, the events are updated in the database using SQL.
        Args:
            location (str): The location name to check for an existing address ID.
            group (pandas.DataFrame): A DataFrame containing event rows to be updated.
            dry_run (bool): If True, no database updates are performed; changes are collected in `fix_events`.
            fix_events (list): A list to collect event updates when in dry run mode.
        Returns:
            bool: True if the location exists and events were processed; False otherwise.
        """

        address_id = self.find_address_id_by_location(location)
        logging.info(f"handle_existing_location: Found address_id {address_id} for location '{location}'")
        
        if address_id:
            for _, row in group.iterrows():
                new_data = {"address_id": int(address_id)}
                logging.info(f"handle_existing_location: Preparing update for event_id {row['event_id']} with {new_data}")
                if dry_run:
                    fix_events.append({**row.to_dict(), **new_data})
                else:
                    self.update_event_with_sql(row, new_data)
            return True
        return False
    

    def handle_llm_address_fix(self, location, group, dry_run, fix_events, fix_addresses):
        """
        Processes and fixes address information for a group of events using an LLM (Large Language Model).
        This method generates a prompt for the LLM to correct or standardize the given location string,
        parses the LLM's response, and updates the corresponding events in the database with the fixed address.
        Supports a dry run mode for previewing changes without applying them.
        Args:
            location (str): The raw location string to be fixed.
            group (pandas.DataFrame): DataFrame containing event rows to update.
            dry_run (bool): If True, changes are collected in fix_events and fix_addresses instead of being applied.
            fix_events (list): List to collect event updates during dry run.
            fix_addresses (list): List to collect address updates during dry run.
        Returns:
            None
        Logs:
            - Warnings if the LLM response is empty or cannot be parsed.
            - Info about generated address IDs and prepared updates.
        """

        prompt, schema_type = self.llm_handler.generate_prompt(
            url="address",
            extracted_text=location,
            prompt_type="address_fix"
        )
        response = self.llm_handler.query_llm("address", prompt, schema_type)
        parsed = self.llm_handler.extract_and_parse_json(response, "address", schema_type)
        
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not parsed:
            logging.warning("handle_llm_address_fix: Parsed result is empty or None")
            return
        
        address_id = self.db_handler.resolve_or_insert_address(parsed)
        if address_id is None:
            logging.warning("handle_llm_address_fix: Failed to resolve or insert address, setting address_id = 0")
            address_id = 0
        else:
            address_id = int(address_id)
        logging.info(f"handle_llm_address_fix: Generated address_id = {address_id}")
        
        for _, row in group.iterrows():
            new_data = {
                "address_id": address_id,
                "location": parsed.get("full_address")
            }
            logging.info(f"handle_llm_address_fix: Preparing update for event_id {row['event_id']} with {new_data}")
            if dry_run:
                fix_events.append({**row.to_dict(), **new_data})
                fix_addresses.append(parsed)
            else:
                self.update_event_with_sql(row, new_data)


    def handle_fallback_llm(self, event_name, description, group, dry_run, fix_events, fix_addresses):
        """
        Handles fallback logic for resolving and updating event addresses using an LLM (Large Language Model).
        This method generates a prompt based on the event description, queries the LLM for address correction,
        parses the response, and updates the event records with the resolved address information. If `dry_run` is True,
        the updates are collected in the provided lists instead of being committed to the database.
        Args:
            event_name (str): The name of the event being processed.
            description (str): The textual description of the event, used for address extraction.
            group (pandas.DataFrame): A DataFrame containing event records to be updated.
            dry_run (bool): If True, do not perform database updates; collect changes in `fix_events` and `fix_addresses`.
            fix_events (list): A list to collect event update data when `dry_run` is True.
            fix_addresses (list): A list to collect address data when `dry_run` is True.
        Returns:
            None
        Logs:
            - Warnings if the LLM response cannot be parsed or is empty.
            - Information about generated address IDs and prepared updates.
        """
        prompt, schema_type = self.llm_handler.generate_prompt(
            url="address",
            extracted_text=description,
            prompt_type="address_fix"
        )
        response = self.llm_handler.query_llm("address", prompt, schema_type)
        parsed = self.llm_handler.extract_and_parse_json(response, "address", schema_type)
        
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if not parsed:
            logging.warning("handle_fallback_llm: Parsed result is empty or None")
            return
        
        address_id = self.db_handler.resolve_or_insert_address(parsed)
        if address_id is None:
            logging.warning("handle_fallback_llm: Failed to resolve or insert address, setting address_id = 0")
            address_id = 0
        else:
            address_id = int(address_id)
        logging.info(f"handle_fallback_llm: Generated address_id = {address_id}")
        
        for _, row in group.iterrows():
            new_data = {
                "address_id": address_id,
                "location": parsed.get("full_address")
            }
            logging.info(f"handle_fallback_llm: Preparing update for event_id {row['event_id']} with {new_data}")
            if dry_run:
                fix_events.append({**row.to_dict(), **new_data})
                fix_addresses.append(parsed)
            else:
                self.update_event_with_sql(row, new_data)


    def update_event_with_sql(self, row, new_data):
        """
        Updates an event record in the database using the provided new data.
        Args:
            row (dict): A dictionary containing at least the "event_id" key, representing the event to update.
            new_data (dict): A dictionary of column-value pairs to update in the event record.
        Returns:
            None
        Side Effects:
            Executes an SQL UPDATE statement on the "events" table to modify the specified event.
        Logs:
            Logs the update operation with the event ID and new data.
        Raises:
            Any exceptions raised by the underlying database handler's execute_query method.
        """
        event_id = int(row["event_id"])
        logging.info(f"update_event_with_sql: Updating event_id {event_id} with {new_data}")
        
        update_cols = [f"{key} = :{key}" for key in new_data.keys()]
        query = f"UPDATE events SET {', '.join(update_cols)} WHERE event_id = :event_id"
        params = {**new_data, "event_id": event_id}
        
        self.db_handler.execute_query(query, params)


    def find_address_id_by_location(self, location):
        """
        Retrieves the address ID corresponding to a given location.
        Args:
            location (str): The full address to search for in the database.
        Returns:
            int or None: The address ID if found, otherwise None.
        Raises:
            Exception: If there is an error executing the database query.
        """
        sql = "SELECT address_id FROM address WHERE full_address = :location"
        result = self.db_handler.execute_query(sql, {"location": location})
        return int(result[0][0]) if result else None
    

    def delete_event_if_completely_empty(self, row, dry_run, fix_events):
        """
        Deletes an event from the database if it is determined to be completely empty.
        If dry_run is True, the event is not actually deleted; instead, the event data is appended
        to the fix_events list with a "delete" flag for review. If dry_run is False, the event is
        deleted from the database.
        Args:
            row (pandas.Series): A row containing event data, including the "event_id".
            dry_run (bool): If True, do not perform deletion, only log the action.
            fix_events (list): A list to which event data is appended if dry_run is True.
        Returns:
            None
        Side Effects:
            - Appends to fix_events if dry_run is True.
            - Deletes the event from the database if dry_run is False.
            - Logs the deletion action.
        """
        event_id = int(row["event_id"])
        
        if dry_run:
            fix_events.append({**row.to_dict(), "delete": True})
        else:
            delete_sql = "DELETE FROM events WHERE event_id = :event_id"
            self.db_handler.execute_query(delete_sql, {"event_id": event_id})
            logging.info("delete_event_if_completely_empty: Deleted event_id %s", event_id)


    def deduplicate_with_embeddings(self, eps=0.3, min_samples=2):
        """
        Perform deduplication of event records using sentence embeddings and DBSCAN clustering.
        This method retrieves event data from the database, generates text embeddings for each event,
        and clusters similar events based on their semantic similarity using the DBSCAN algorithm.
        Events are grouped by start date and clustered within a 30-minute time window and, if available,
        by address ID. For each cluster of similar events, a canonical event is selected.
        The deduplication results and statistics are saved to CSV files in the "output" directory.

        Args:
            eps (float, optional): The maximum distance between two samples for them to be considered
                as in the same neighborhood (DBSCAN epsilon parameter). Defaults to 0.3.
            min_samples (int, optional): The number of samples in a neighborhood for a point to be
                considered as a core point (DBSCAN min_samples parameter). Defaults to 2.

        Outputs:
            - "output/dups_trans_db_scan.csv": CSV file containing deduplicated event clusters.
            - "output/stats_dedup.csv": CSV file containing deduplication run statistics.

        Logs:
            - Information about the deduplication process, including progress and output file locations.

        Raises:
            Any exceptions raised by database access, file I/O, or model inference are propagated.

        Dependencies:
            - Requires the 'SentenceTransformer' model 'all-MiniLM-L6-v2'.
            - Requires pandas, scikit-learn, and other standard libraries.
        """
        logging.info("Starting embedding-based deduplication...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query = """
            SELECT event_id, event_name, dance_style, description, day_of_week,
                   start_date, end_date, start_time, end_time, source, location,
                   price, url, event_type, address_id
            FROM events
            ORDER BY start_date DESC
        """
        df = pd.read_sql(query, self.engine)

        if df.empty:
            logging.info("No events found for deduplication.")
            return

        string_cols = df.select_dtypes(include='object').columns
        df[string_cols] = df[string_cols].fillna('')
        df['start_datetime'] = pd.to_datetime(df['start_date'].astype(str) + ' ' + df['start_time'].astype(str))

        address_query = "SELECT address_id, postal_code FROM address"
        address_df = pd.read_sql(address_query, self.engine)

        os.makedirs("output", exist_ok=True)
        results = []
        cluster_id_counter = 0

        for start_date, group in df.groupby('start_date'):
            group = group.copy()
            group.sort_values('start_time', inplace=True)
            processed_indices = set()

            for idx, row in group.iterrows():
                if idx in processed_indices:
                    continue

                base_time = row['start_datetime']
                base_address_id = row['address_id']

                time_window = group[(group['start_datetime'] >= base_time - timedelta(minutes=30)) &
                                    (group['start_datetime'] <= base_time + timedelta(minutes=30))]
                time_window = time_window[~time_window.index.isin(processed_indices)]

                # Address ID restriction
                if pd.notnull(base_address_id):
                    time_window = time_window[(time_window['address_id'].isnull()) |
                                              (time_window['address_id'] == base_address_id)]

                if len(time_window) < 2:
                    processed_indices.update(time_window.index)
                    continue

                time_window['text'] = time_window.apply(lambda r: ' | '.join([
                    str(r['event_name']), str(r['dance_style']), str(r['description']),
                    str(r['day_of_week']), str(r['start_date']), str(r['end_date']),
                    str(r['start_time']), str(r['end_time']), str(r['source']),
                    str(r['location']), str(r['price']), str(r['url']), str(r['event_type'])
                ]), axis=1)

                embeddings = model.encode(time_window['text'].tolist(), convert_to_numpy=True)
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
                time_window['cluster'] = clustering.labels_

                for cluster_id, subgroup in time_window[time_window['cluster'] != -1].groupby('cluster'):
                    deduped_cluster = self.find_canonical_event(subgroup, address_df)
                    deduped_cluster['cluster_id'] = cluster_id_counter
                    deduped_cluster['score_correct'] = None
                    results.append(deduped_cluster)
                    cluster_id_counter += 1

                processed_indices.update(time_window.index)

        if results:
            output_df = pd.concat(results, ignore_index=True)
            output_df.to_csv("output/dups_trans_db_scan.csv", index=False)
            logging.info("Saved deduplication results to output/dups_trans_db_scan.csv")
            
            # Create a clean summary of ALL cluster members for review (canonical + duplicates)
            cluster_review = output_df[
                ['event_id', 'event_name', 'start_date', 'start_time', 'location', 'source', 'cluster_id', 'is_canonical']
            ].copy()
            cluster_review['status'] = cluster_review['is_canonical'].map({True: 'CANONICAL', False: 'PROPOSED_DUPLICATE'})
            cluster_review = cluster_review.drop('is_canonical', axis=1)
            cluster_review['reason'] = 'semantic_similarity'
            
            # Sort by cluster_id so canonical and duplicates are grouped together
            cluster_review = cluster_review.sort_values(['cluster_id', 'status'])
            cluster_review.to_csv("output/duplicates.csv", index=False)
            
            num_duplicates = len(cluster_review[cluster_review['status'] == 'PROPOSED_DUPLICATE'])
            num_clusters = cluster_review['cluster_id'].nunique()
            logging.info(f"Saved {num_clusters} clusters with {num_duplicates} proposed duplicates to output/duplicates.csv for review")
            
            # Delete the proposed duplicate events from the database
            duplicate_event_ids = cluster_review[cluster_review['status'] == 'PROPOSED_DUPLICATE']['event_id'].tolist()
            if duplicate_event_ids:
                logging.info(f"Deleting {len(duplicate_event_ids)} duplicate events from database...")
                if self.db_handler.delete_multiple_events(duplicate_event_ids):
                    logging.info(f"Successfully deleted {len(duplicate_event_ids)} duplicate events from database")
                else:
                    logging.error(f"Failed to delete some duplicate events from database")
            else:
                logging.info("No duplicate events to delete")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_rows = len(df)
            clustered_rows = len(output_df)
            canonical_rows = output_df['is_canonical'].sum()
            duplicate_rows = clustered_rows - canonical_rows
            num_clusters = output_df['cluster_id'].nunique()
            git_commit = self.get_git_version()

            stats_row = pd.DataFrame([{
                'timestamp': timestamp,
                'total_rows': total_rows,
                'clustered_rows': clustered_rows,
                'num_clusters': num_clusters,
                'canonical_rows': canonical_rows,
                'duplicate_rows': duplicate_rows,
                'eps': eps,
                'min_samples': min_samples,
                'model': 'all-MiniLM-L6-v2',
                'method': 'DBSCAN',
                'git_commit': git_commit
            }])

            stats_file = "output/stats_dedup.csv"
            if os.path.exists(stats_file):
                stats_row.to_csv(stats_file, mode='a', header=False, index=False)
            else:
                stats_row.to_csv(stats_file, mode='w', header=True, index=False)

            logging.info("Appended deduplication stats to output/stats_dedup.csv")
        else:
            logging.info("No clusters found to output.")


    def evaluate_scored_clusters(self, input_file="output/dups_trans_db_scan.csv", stats_file="output/stats_dedup.csv"):
        """
        Loads manually scored results and logs evaluation statistics.
        Appends evaluation details to the same stats CSV used during deduplication.
        """
        if not os.path.exists(input_file):
            logging.warning(f"evaluate_scored_clusters: File not found: {input_file}")
            return

        df = pd.read_csv(input_file)
        if 'score_correct' not in df.columns:
            logging.warning("evaluate_scored_clusters: 'score_correct' column missing.")
            return

        if df['score_correct'].isnull().all():
            logging.warning("evaluate_scored_clusters: No manual scores entered yet.")
            return

        evaluated = df[df['score_correct'].isin([0, 1])]
        if evaluated.empty:
            logging.warning("evaluate_scored_clusters: No usable score entries found.")
            return

        total = len(evaluated)
        correct = evaluated['score_correct'].sum()
        accuracy = round(correct / total, 4)

        logging.info(f"Evaluation accuracy: {accuracy} ({correct}/{total} correct)")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        git_commit = self.get_git_version()

        eval_row = pd.DataFrame([{
            'timestamp': timestamp,
            'eval_total_scored': total,
            'eval_correct': correct,
            'eval_accuracy': accuracy,
            'git_commit': git_commit
        }])

        if os.path.exists(stats_file):
            eval_row.to_csv(stats_file, mode='a', header=False, index=False)
        else:
            eval_row.to_csv(stats_file, mode='w', header=True, index=False)

        logging.info("Appended evaluation stats to output/stats_dedup.csv")


    def get_git_version(self):
        """
        Retrieves the current Git commit hash for the repository.
        Returns:
            str: The current Git commit hash as a string if available, otherwise "unknown".
        Exceptions:
            Returns "unknown" if the Git command fails or if an exception occurs.
        """
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            return "unknown"


    def score_event_row(self, row, address_df):
        """
        Calculates a score for an event row based on the presence and quality of its fields.
        The scoring algorithm rewards rows with non-null and non-empty values in key fields such as
        'location', 'description', and 'event_name'. Additional points are given if the row has a valid
        'address_id' and the corresponding address has a postal code. Optional fields like 'source',
        'dance_style', 'event_type', 'start_time', and 'url' contribute fractional points if present.
        The length of the 'description' field also slightly increases the score.
        Args:
            row (pd.Series): A pandas Series representing a single event row with expected fields.
            address_df (pd.DataFrame): A DataFrame containing address information, indexed by 'address_id'.
        Returns:
            float: The computed score for the event row. Returns negative infinity if 'location' is missing.
        """
        if not row['location']:
            return -float('inf')

        score = 0

        if pd.notnull(row['address_id']):
            score += 1
            matching_address = address_df[address_df['address_id'] == row['address_id']]
            if not matching_address.empty and pd.notnull(matching_address.iloc[0].get('postal_code')):
                score += 1

        for field in ['location', 'description', 'event_name']:
            if not row[field]:
                score -= 1

        for field in ['source', 'dance_style', 'event_type', 'start_time', 'url']:
            if row[field]:
                score += 0.5

        score += 0.01 * len(row['description']) if row['description'] else 0
        return score


    def find_canonical_event(self, cluster_df, address_df):
        """
        Identifies the canonical event within a cluster of potentially duplicate events and marks duplicates.
        This method scores each event in the provided cluster DataFrame using the `score_event_row` method,
        selects the event with the highest score as the canonical event, and marks all other events as duplicates.
        The canonical event is flagged with `is_canonical = True`, while duplicates are flagged with `is_canonical = False`.
        Args:
            cluster_df (pd.DataFrame): DataFrame containing events that are considered potential duplicates.
            address_df (pd.DataFrame): DataFrame containing address information used for scoring events.
        Returns:
            pd.DataFrame: DataFrame with the canonical event and its duplicates, each row annotated with an
            `is_canonical` boolean column indicating whether the event is canonical.
        """
        cluster_df = cluster_df.copy()
        cluster_df['score'] = cluster_df.apply(lambda row: self.score_event_row(row, address_df), axis=1)
        canonical = cluster_df.sort_values('score', ascending=False).iloc[0]
        duplicates = cluster_df[cluster_df['event_id'] != canonical['event_id']].copy()
        canonical['is_canonical'] = True
        duplicates['is_canonical'] = False
        return pd.concat([canonical.to_frame().T, duplicates])


    def driver(self):
        """
        Main driver function for the deduplication process.
        """
        while True:
            total_deleted = self.process_duplicates()
            logging.info(f"Main loop: Number of events deleted in this pass: {total_deleted}")
            if total_deleted == 0:
                logging.info("No duplicates found. Exiting deduplication loop.")
                break

        # Fix null locations and addresses
        logging.info("Starting fix_problem_events()...")
        self.fix_problem_events(dry_run=False)

        # This calls a LLM and parses the address.full_address into the appropriate columns.
        # It does not seem like much of an issue anymore. Only run it when it does become an issue.
        # You may have to change the sql. Right now it gets EVERY row in the address table.
        # self.parse_address()

        # This uses transformers and DBSCAN to find duplicates based on embeddings.
        self.deduplicate_with_embeddings()

        # In order to improve deduplication we really need to evaluate the clusters that were scored.
        if self.config.get('score', {}).get('dup_trans_db_scan', False):
            self.evaluate_scored_clusters()
        else:
            logging.info("Skipping evaluation scoring due to config setting.")

        logging.info("dedup.py finished.")


if __name__ == "__main__":

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize the class libraries
    deduper = DeduplicationHandler()
    db_handler = deduper.db_handler  # Use the connected handler

    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    deduper.driver()
    
    db_handler.count_events_urls_end(start_df, file_name)
    