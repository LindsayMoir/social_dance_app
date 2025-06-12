"""
db.py

This module provides the DatabaseHandler class, which encapsulates methods 
for connecting to, interacting with, and managing a PostgreSQL database 
using SQLAlchemy. It includes functionality to create tables, execute queries, 
update records, clean and process event data, deduplicate tables, and delete old 
or invalid entries. The module also sets up logging to record database operations 
and errors.

Classes:
    DatabaseHandler: 
        - Initializes a connection to a PostgreSQL database using configuration 
          parameters.
        - Provides methods to create necessary tables ('urls', 'events', 'address', 
          'organizations') if they do not exist, with options to drop tables 
          based on configuration.
        - Contains methods to execute SQL queries safely with error handling.
        - Supports updating URLs, writing new URLs, cleaning event data, writing 
          events to the database, deduplicating records, and deleting outdated 
          or null events.
        - Handles address parsing and normalization, inserting addresses into 
          the 'address' table, and associating events with address IDs.
        - Manages database connection lifecycle, including closing connections.

Usage Example:
    if __name__ == "__main__":
        # Load configuration and set up logging
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        logging.basicConfig(
            filename=config['logging']['log_file'],
            filemode='a',
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        # Initialize DatabaseHandler and manage database operations
        db_handler = DatabaseHandler(config)
        db_handler.create_tables()
        db_handler.dedup()
        db_handler.delete_old_events()
        db_handler.delete_events_with_nulls()
        db_handler.conn.dispose()

Dependencies:
    - logging: For logging events, errors, and informational messages.
    - os, datetime, re: Standard libraries for file operations, time, and regex.
    - pandas (pd): For data manipulation and DataFrame operations.
    - sqlalchemy: For database connection, query execution, and ORM support.
    - yaml: For loading configuration from YAML files.

Note:
    The module assumes a valid YAML configuration file containing database 
    credentials, logging configurations, and application-specific settings. 
    Logging is configured in the main block to record operations and any errors 
    encountered during database interactions.
"""

from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
from fuzzywuzzy import fuzz
import logging
import numpy as np
import os
import pandas as pd
import re  # Added missing import
import requests
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
import sys
import yaml
import warnings


class DatabaseHandler():
    def __init__(self, config):
        """
        Initializes the DatabaseHandler with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters for the database connection.
        """
        self.config = config
        self.load_blacklist_domains()

        if os.getenv("RENDER"):
            logging.info("def __init__(): Running on Render.")
            connection_string = os.getenv('RENDER_EXTERNAL_DB_URL')
            self.conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
            logging.info("def __init__(): Database connection established for Render social_dance_db.")
        else:
            # Running locally
            logging.info("def __init__(): Running locally.")
            self.conn = self.get_db_connection()
            logging.info("def __init__(): Database connection established for social_dance_db.")

            # Create address_db_engine
            self.address_db_engine = create_engine(os.getenv("ADDRESS_DB_CONNECTION_STRING"), 
                                                   isolation_level="AUTOCOMMIT")
            logging.info("def __init__(): Database connection established for address_db.")

        if self.conn is None:
                raise ConnectionError("def __init__(): DatabaseHandler: Failed to establish a database connection.")

        self.metadata = MetaData()
        # Reflect the existing database schema into metadata
        self.metadata.reflect(bind=self.conn)

        # Get google api key
        self.google_api_key = os.getenv("GOOGLE_KEY_PW")

        # Create df from urls table.
        self.urls_df = self.create_urls_df()
        logging.info("__init__(): URLs DataFrame created with %d rows.", len(self.urls_df))

        def _compute_hit_ratio(x):
            true_count = x.sum()
            false_count = (~x).sum()

            # Case 1: at least one True and one False
            if true_count > 0 and false_count > 0:
                return true_count / false_count

            # Case 2: 0 Trues and all False
            if true_count == 0:
                return 0.0

            # Case 3: all Trues and no False
            return 1.0

        # Create a groupby that gives a hit_ratio and a sum of crawl_try for how useful the URL is
        self.urls_gb = (
            self.urls_df
            .groupby('link')
            .agg(
                hit_ratio=('relevant', _compute_hit_ratio),
                crawl_try=('crawl_try', 'sum')
            )
            .reset_index()
        )
        logging.info(f"__init__(): urls_gb has {len(self.urls_gb)} rows and {len(self.urls_gb.columns)} columns.")
            

    def load_blacklist_domains(self):
        """ Load blacklisted domains from CSV once at initialization. """
        csv_path = self.config['constants']['black_list_domains']
        df = pd.read_csv(csv_path)
        self.blacklisted_domains = set(df['Domain'].str.lower().str.strip())
        logging.info(f"Loaded {len(self.blacklisted_domains)} blacklisted domains.")

    def avoid_domains(self, url):
        """ Check if URL contains any blacklisted domain. """
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.blacklisted_domains)
    

    def get_db_connection(self):
        """
        Establishes and returns a SQLAlchemy engine for connecting to the PostgreSQL database.

        Returns:
            sqlalchemy.engine.Engine: A SQLAlchemy engine for the PostgreSQL database connection.
            None: If the connection fails.
        """
        try:
            # Read the database connection parameters from the config
            connection_string = (
                f"postgresql://{os.getenv('DATABASE_USER')}:" 
                f"{os.getenv('DATABASE_PASSWORD')}@"
                f"{os.getenv('DATABASE_HOST')}/"
                f"{os.getenv('DATABASE_NAME')}"
            )

            # Create and return the SQLAlchemy engine
            engine = create_engine(connection_string, isolation_level="AUTOCOMMIT")
            return engine

        except Exception as e:
            logging.error("DatabaseHandler: Database connection failed: %s", e)
            return None
        

    def create_tables(self):
        """
        Creates the 'urls', 'events', 'address', and 'organization tables in the database if they do not already exist.
        If config['testing']['drop_tables'] is True, it will drop existing tables before creation.
        """
        
        # Check if we need to drop tables as per configuration
        if self.config['testing']['drop_tables'] == True:
            drop_queries = [
                "DROP TABLE IF EXISTS events CASCADE;"
            ]
            # Copy events to events_history
            sql = 'SELECT * FROM events;'
            events_df = pd.read_sql(sql, self.conn) 
            events_df.to_sql('events_history', self.conn, if_exists='append', index=False)

        else:
            # Don't drop any tables
            drop_queries = []

        if drop_queries:
            for query in drop_queries:
                self.execute_query(query)
                logging.info(
                    f"create_tables: Existing tables dropped as per configuration value of "
                    f"'{self.config['testing']['drop_tables']}'."
                )
        else:
            pass

        # Create the 'urls' table
        urls_table_query = """
            CREATE TABLE IF NOT EXISTS urls (
                link_id SERIAL PRIMARY KEY
                link TEXT,
                parent_url TEXT,
                source TEXT,
                keywords TEXT,
                relevant BOOLEAN,
                crawl_try INTEGER,
                time_stamp TIMESTAMP
            )
        """
        self.execute_query(urls_table_query)
        logging.info("create_tables: 'urls' table created or already exists.")

        # Create the 'events' table
        events_table_query = """
            CREATE TABLE IF NOT EXISTS events (
                event_id SERIAL PRIMARY KEY,
                event_name TEXT,
                dance_style TEXT,
                description TEXT,
                day_of_week TEXT,
                start_date DATE,
                end_date DATE,
                start_time TIME,
                end_time TIME,
                source TEXT,
                location TEXT,
                price TEXT,
                url TEXT,
                event_type TEXT,
                address_id INTEGER,
                time_stamp TIMESTAMP
            )
        """
        self.execute_query(events_table_query)
        logging.info("create_tables: 'events' table created or already exists.")

        # Create the 'address' table
        address_table_query = """
            CREATE TABLE IF NOT EXISTS address (
                address_id SERIAL PRIMARY KEY,
                full_address TEXT UNIQUE,
                building_name TEXT,
                street_number TEXT,
                street_name TEXT,
                street_type TEXT,
                direction TEXT,
                city TEXT,
                met_area TEXT,
                province_or_state TEXT,
                postal_code TEXT,
                country_id TEXT,
                time_stamp TIMESTAMP
            )
        """
        self.execute_query(address_table_query)
        logging.info("create_tables: 'address' table created or already exists.")

        # Create the 'runs' table
        runs_table_query = """
            CREATE TABLE IF NOT EXISTS runs (
                run_id SERIAL PRIMARY KEY,
                run_name TEXT UNIQUE,
                run_description TEXT,
                start_time TEXT,
                end_time TEXT,
                elapsed_time TEXT,
                python_file_name TEXT,
                unique_urls_count INTEGER,
                total_url_attempts INTEGER,
                urls_with_extracted_text INTEGER,
                urls_with_found_keywords INTEGER,
                events_written_to_db INTEGER,
                time_stamp TIMESTAMP
            )
        """
        self.execute_query(runs_table_query)
        logging.info("create_tables: 'address' table created or already exists.")

        # See if this worked.
        query = """
                SELECT table_schema,
                    table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """
        rows = self.execute_query(query)
        if rows:
            for schema, table in rows:
                logging.info("Schema: %s, Table: %s", schema, table)
        else:
            logging.info("No tables found or query failed.")


    def create_urls_df(self):
        """
        Creates a DataFrame from the 'urls' table in the database.

        Returns:
            pandas.DataFrame: A DataFrame containing all rows from the 'urls' table.
        """
        query = "SELECT * FROM urls;"
        try:
            urls_df = pd.read_sql(query, self.conn)
            logging.info("create_urls_df: Successfully created DataFrame from 'urls' table.")
            if urls_df.empty:
                logging.warning("create_urls_df: 'urls' table is empty.")
            else:
                logging.info("create_urls_df: 'urls' table contains %d rows.", len(urls_df))
            return urls_df
        except SQLAlchemyError as e:
            logging.error("create_urls_df: Failed to create DataFrame from 'urls' table: %s", e)
            return pd.DataFrame()
        

    def execute_query(self, query, params=None):
        """
        Executes a given SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (dict): Optional dictionary of parameters for parameterized queries.

        Returns:
            list of dict or int or None: A list of rows as dictionaries if the query returns rows,
                                        the number of rows affected for update operations,
                                        else None.
        """
        if self.conn is None:
            logging.error("execute_query: No database connection available.")
            return None

        # Handle NaN values in params (such as address_id)
        if params:
            for key, value in params.items():
                if isinstance(value, (list, np.ndarray, pd.Series)):
                    # Check if any element in the array-like object is NaN
                    if pd.isna(value).any():
                        params[key] = None  # Set to None (NULL in SQL)
                else:
                    # For scalar values, check directly
                    if pd.isna(value):
                        params[key] = None  # Set to None (NULL in SQL)

        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(query), params or {})
                if result.returns_rows:
                    rows = result.fetchall()
                    return rows
                else:
                    connection.commit()
                    return result.rowcount  # Return the number of rows affected
                    
        except SQLAlchemyError as e:
            logging.error("def execute_query(): %s\nQuery execution failed: %s", query, e)
            return None

    
    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.conn:
            try:
                self.conn.dispose()
                logging.info("close_connection: Database connection closed successfully.")
            except Exception as e:
                logging.error("close_connection: Failed to close database connection: %s", e)
        else:
            logging.warning("close_connection: No database connection to close.")


    def write_url_to_db(self, url_row):
        """
        Logs a URL activity by appending a new row to the 'urls' table via pandas.

        Args:
            url_row (tuple): (link, parent_url, source, keywords, crawl_try, time_stamp)
        """
        # 1) Unpack
        link, parent_url, source, keywords, relevant, crawl_try, time_stamp = url_row

        # 2) Normalize keywords into a simple comma-separated string
        if not isinstance(keywords, str):
            if isinstance(keywords, (list, tuple, set)):
                keywords = ','.join(map(str, keywords))
            else:
                keywords = str(keywords)

        # 3) Strip out braces/brackets/quotes and trim each term
        cleaned = re.sub(r'[\{\}\[\]\"]', '', keywords)
        parts = [p.strip() for p in cleaned.split(',') if p.strip()]
        keywords = ', '.join(parts)

        # 4) Build a one-row DataFrame
        df = pd.DataFrame([{
            'link':           link,
            'parent_url':     parent_url,
            'source':         source,
            'keywords':       keywords,
            'relevant':       relevant,
            'crawl_try':      crawl_try,
            'time_stamp':     time_stamp
        }])

        # 5) Append to the table
        try:
            df.to_sql('urls', con=self.conn, if_exists='append', index=False)
            logging.info("write_url_to_db(): appended URL '%s'", link)
        except Exception as e:
            logging.error("write_url_to_db(): failed to append URL '%s': %s", link, e)
    

    def write_events_to_db(self, df, url, parent_url, source, keywords):
        """
        Writes event data to the 'events' table in the database.

        Args:
            df (pandas.DataFrame): DataFrame containing events data.
            url (str): URL from which the events data was sourced.

        Notes:
            - The 'event_id' column is auto-generated and should not be included in the DataFrame.
            - Ensures that only relevant columns are written to the database.
        """
        # — ensure url and parent_url are safe to search —
        url = '' if pd.isna(url) else str(url)
        parent_url = '' if pd.isna(parent_url) else str(parent_url)

        # Need to check if it is from google calendar or from the LLM.
        if 'calendar' in url or 'calendar' in parent_url:
            
            # Rename columns to match the database schema
            df = df.rename(columns={
                'URL': 'url',
                'Type_of_Event': 'event_type',
                'Name_of_the_Event': 'event_name',
                'Day_of_Week': 'day_of_week',
                'Start_Date': 'start_date',
                'End_Date': 'end_date',
                'Start_Time': 'start_time',
                'End_Time': 'end_time',
                'Price': 'price',
                'Location': 'location',
                'Description': 'description'
            })
            
            # Check data type and assign dance_style
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            df['dance_style'] = keywords
        
        # Fill a blank source
        if source == '':
            source = url.split('.')[-2]
        if source in df.columns:
            # If df['source'] is null or '', use source from the function argument
            df['source'] = df['source'].fillna('').replace('', source)
        else:
            df['source'] = source
        
        # fill missing or empty strings with the current URL
        df['url'] = df['url'].fillna('').replace('', url)

        # Suppress warnings for date parsing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Ensure 'start_date' and 'start_date' are in datetime.date format
            for col in ['start_date', 'end_date']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

            # Ensure 'start_time' and 'end_time' are in datetime.time format
            for col in ['start_time', 'end_time']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

            # Turn warnings back on
            warnings.resetwarnings()

        # No need to clean and convert the 'price' column to numeric format as it should remain as text
        if 'price' not in df.columns:
            logging.warning("write_events_to_db: 'price' column is missing. Filling with empty string.")
            df['price'] = ''

        # Add a 'time_stamp' column with the current timestamp
        df['time_stamp'] = datetime.now()

        # Clean up the 'location' column and update address_ids
        cleaned_df = self.clean_up_address(df)

        # Delete the rows if the majority of important columns are empty
        important_columns = ['start_date', 'end_date', 'start_time', 'end_time', 'location', 'description']

        # Replace empty strings (and strings containing only whitespace) with NA in the entire DataFrame for just the important columns
        cleaned_df[important_columns] = cleaned_df[important_columns].replace(r'^\s*$', pd.NA, regex=True)

        # Now drop rows where all important columns are NA
        cleaned_df = cleaned_df.dropna(subset=important_columns, how='all')

        # Drop rows older than “today minus N days”
        # Ensure 'end_date' is datetime64[ns]
        cleaned_df['end_date'] = pd.to_datetime(cleaned_df['end_date'], errors='coerce')

        # Drop rows older than “today minus N days”
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.config['clean_up']['old_events'])
        cleaned_df = cleaned_df.loc[cleaned_df['end_date'] >= cutoff].reset_index(drop=True)

        # If there are no events left, bail out early
        if cleaned_df.empty:
            logging.info("write_events_to_db: No events remain after checking for end_date, skipping write.")
            relevant, crawl_try, time_stamp = False, 1, datetime.now()
            url_row = [url, parent_url, source, keywords, relevant, crawl_try, time_stamp]
            self.write_url_to_db(url_row)
            return None

        # Save the cleaned events data to a CSV file for debugging purposes
        cleaned_df.to_csv('output/cleaned_events.csv', index=False)

        # Log the number of events to be written
        logging.info(f"write_events_to_db: Number of events to write: {len(df)}")

        cleaned_df.to_csv('output/cleaned_events.csv', index=False)

        # Write the cleaned events data to the 'events' table
        cleaned_df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        relevant, crawl_try, time_stamp = True, 1, datetime.now()
        url_row = [url, parent_url, source, keywords, relevant, crawl_try, time_stamp]
        self.write_url_to_db(url_row)
        logging.info("write_events_to_db: Events data written to the 'events' table.")
    
    
    def update_event(self, event_identifier, new_data, best_url):
        """
        Updates an event row in the database based on event_identifier criteria by overlaying new_data.
        If new_data has information for a column, it replaces existing data. The best_url is also set.
        
        Args:
            event_identifier (dict): Criteria to locate the event row (e.g. {'event_name': ..., 'start_date': ..., 'start_time': ...}).
            new_data (dict): Data to overlay onto the existing event record.
            best_url (str): The best URL to update for the event.
        """
        select_query = """
        SELECT * FROM events
        WHERE event_name = :event_name
        AND start_date = :start_date
        AND start_time = :start_time
        """
        result = self.execute_query(select_query, event_identifier)
        existing_row = result[0] if result else None
        if not existing_row:
            logging.error("update_event: No matching event found for identifier: %s", event_identifier)
            return False
        
        # Overlay new data onto existing row
        updated_data = dict(existing_row)
        for col, new_val in new_data.items():
            if new_val not in [None, '', pd.NaT]:
                updated_data[col] = new_val
        
        # Update URL
        updated_data['url'] = best_url

        update_cols = [col for col in updated_data.keys() if col != 'event_id']
        set_clause = ", ".join([f"{col} = :{col}" for col in update_cols])
        update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        updated_data['event_id'] = existing_row['event_id']

        self.execute_query(update_query, updated_data)
        logging.info("update_event: Updated event %s", updated_data)
        return True


    def get_address_id(self, address_dict):
        """
        Inserts a new address or updates the existing address using INSERT ... ON CONFLICT.
        A time_stamp column is added and set to the current datetime.

        Args:
            address_dict (dict): Dictionary containing address details.

        Returns:
            int or None: The address_id if successful, else None.
        """
        # Add the current datetime to the address dictionary for the time_stamp column.
        address_dict['time_stamp'] = datetime.now()

        try:
            # Define the INSERT query with ON CONFLICT DO UPDATE and RETURNING address_id,
            # including the time_stamp column.
            insert_query = """
                INSERT INTO address (
                    full_address, street_number, street_name, street_type, 
                    city, province_or_state, postal_code, country_id, time_stamp
                ) VALUES (
                    :full_address, :street_number, :street_name, :street_type,
                    :city, :province_or_state, :postal_code, :country_id, :time_stamp
                )
                ON CONFLICT (full_address) DO UPDATE SET
                    street_number = EXCLUDED.street_number,
                    street_name = EXCLUDED.street_name,
                    street_type = EXCLUDED.street_type,
                    city = EXCLUDED.city,
                    province_or_state = EXCLUDED.province_or_state,
                    postal_code = EXCLUDED.postal_code,
                    country_id = EXCLUDED.country_id,
                    time_stamp = EXCLUDED.time_stamp
                RETURNING address_id;
            """

            # Execute the INSERT query
            insert_result = self.execute_query(insert_query, address_dict)

            if insert_result:
                # Insert succeeded or update occurred; retrieve the address_id
                row = insert_result[0]
                address_id = row[0]  # address_id is the first column in the result

                # Convert to Python int
                if isinstance(address_id, np.int64):
                    address_id = int(address_id)

                logging.info(f"get_address_id: Inserted or updated address_id {address_id} for address '{address_dict['full_address']}'.")
                return address_id

            logging.error(f"get_address_id: Failed to insert or update address_id for '{address_dict['full_address']}'.")
            return None

        except SQLAlchemyError as e:
            logging.error(f"get_address_id: Database error: {e}")
            return None

    
    def get_postal_code(self, address, api_key):
        """
        Given an address string, query the Google Geocoding API and extract the postal code.

        Args:
            address (str): The address string to geocode.
            api_key (str): Your Google Maps Geocoding API key.

        Returns:
            str or None: The postal code if found, otherwise None.
        """
        endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address,
            "key": api_key
        }
        
        response = requests.get(endpoint, params=params)
        if response.status_code != 200:
            logging.warning(f"Geocoding API request failed with status code {response.status_code}")
            return None
        
        data = response.json()
        if data.get("status") != "OK":
            logging.warning(f"def get_postal_code(): Geocoding API rejection: {data.get('status')}")
            return None
        
        # Look for the postal_code component in the results.
        for result in data.get("results", []):
            for component in result.get("address_components", []):
                if "postal_code" in component.get("types", []):
                    return component.get("long_name")
    

    def get_municipality(self, address, api_key):
        """
        Given an address string, query the Google Geocoding API and extract the municipality.

        Args:
            address (str): The address string to geocode.
            api_key (str): Your Google Maps Geocoding API key.

        Returns:
            str or None: The municipality (locality) if found, otherwise None.
        """
        endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address,
            "key": api_key
        }
        
        response = requests.get(endpoint, params=params)
        if response.status_code != 200:
            logging.warning(f"Geocoding API request failed with status code {response.status_code}"
                            f" for address '{address}'")
            return None
        
        data = response.json()
        if data.get("status") != "OK":
            logging.warning(f"Geocoding API rejection: {data.get('status')}"
                            f" for address '{address}'")
            return None
        
        # Iterate through the results to find the "locality" component (which is typically the municipality)
        for result in data.get("results", []):
            for component in result.get("address_components", []):
                if "locality" in component.get("types", []):
                    return component.get("long_name")
                else:
                    return None


    def clean_up_address(self, events_df):
        """
        Cleans and standardizes address data from the 'events' table by:
        1) Extracting or retrieving a Canadian postal code (regex or Google).
        2) Attempting to fill in the address from the address DB.
        3) Falling back to partial updates if no DB match.
        4) If no postal code, uses Google municipality.
        """
        logging.info("def clean_up_address(): Starting with events_df shape: %s", events_df.shape)
        
        # Load address table once from the database.
        address_df = pd.read_sql("SELECT * FROM address", self.conn)

        for index, row in events_df.iterrows():
            raw_location = row.get('location')
            event_id = row.get('event_id')
            if not raw_location or pd.isna(raw_location):
                continue

            location = str(raw_location).strip()
            logging.info(f"def clean_up_address(): Processing location '{location}' (event_id: {event_id})")
            
            # INSERT HERE: Try using the existing address update method.
            update_list = self.get_address_update_for_event(event_id, location, address_df)
            if update_list:
                # If there's a match, update the address_id and skip further processing.
                new_address_id = update_list[0]['address_id']
                events_df.loc[index, 'address_id'] = new_address_id
                logging.info("def clean_up_address(): Updated event %s with address_id %s using DB match", event_id, new_address_id)
                continue  # Skip remaining logic for this row.
            
            # 1) Try extracting a postal code via regex.
            postal_code = self.extract_canadian_postal_code(location)

            # 2) If none found, try Google. ***TEMP
            # if not postal_code:
            #     google_pc = self.get_postal_code(location, self.google_api_key)
            #     if google_pc and self.is_canadian_postal_code(google_pc):
            #         postal_code = google_pc
            #         logging.info("Got Canadian postal code '%s' from Google for '%s'", postal_code, location)

            # 3) If postal code is found, query DB to get full address.
            if postal_code:
                updated_location, address_id = self.populate_from_db_or_fallback(location, postal_code)
                events_df.loc[index, 'location'] = updated_location
                events_df.loc[index, 'address_id'] = address_id
            # else: ***TEMP
            #     # 4) If still no postal code, fallback to municipality from Google.
            #     updated_location, address_id = self.fallback_with_municipality(location)
            #     if updated_location:
            #         events_df.loc[index, 'location'] = updated_location
            #     if address_id:
            #         events_df.loc[index, 'address_id'] = address_id

        return events_df

    # ─────────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────────

    def extract_canadian_postal_code(self, location_str):
        """
        Extracts a Canadian postal code from a location string using regex.
        Returns the postal code with spaces removed, or None if not found or invalid.
        """
        match = re.search(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d', location_str)
        if match:
            possible_pc = match.group().replace(' ', '')
            if self.is_canadian_postal_code(possible_pc):
                return possible_pc
        return None

    def create_address_dict(self, full_address, street_number, street_name, street_type, postal_box, city, province_or_state, postal_code, country_id):
        """
        Creates an address dictionary with the given parameters.

        Args:
            full_address (str): The full address.
            street_number (str): The street number.
            street_name (str): The street name.
            street_type (str): The street type.
            postal_box (str): The postal box.
            city (str): The city.
            province_or_state (str): The province or state.
            postal_code (str): The postal code.
            country_id (str): The country ID.

        Returns:
            dict: The address dictionary.
        """
        return {
            'full_address': full_address,
            'street_number': street_number,
            'street_name': street_name,
            'street_type': street_type,
            'postal_box': postal_box,
            'city': city,
            'province_or_state': province_or_state,
            'postal_code': postal_code,
            'country_id': country_id
        }

    def populate_from_db_or_fallback(self, location_str, postal_code):
        """
        Given a valid Canadian postal_code, tries to populate address from the DB.
        If no match is found, returns a fallback partial location.

        Returns:
            (updated_location, address_id)
        """
        numbers = re.findall(r'\d+', location_str)
        query = """
            SELECT
                civic_no,
                civic_no_suffix,
                official_street_name,
                official_street_type,
                official_street_dir,
                mail_mun_name,
                mail_prov_abvn,
                mail_postal_code
            FROM locations
            WHERE mail_postal_code = %s;
        """
        df = pd.read_sql(query, self.address_db_engine, params=(postal_code,))
        # if df.empty: ***TEMP
        #     # Fallback if no match in DB
        #     municipality = self.get_municipality(location_str, self.google_api_key) 
        #     updated_location = f"{location_str}, {municipality}, BC, {postal_code}, CA"
        #     updated_location = updated_location.replace('None,', '').strip()
        #     logging.info(f"updated_location is: {updated_location}")

        #     address_dict = self.create_address_dict(
        #         updated_location, None, None, None, None, municipality, 'BC', postal_code, 'CA'
        #     )
        #     logging.info(f"address_dict is: {address_dict}")

        #     address_id = self.get_address_id(address_dict)
        #     logging.info("No DB match for postal code '%s'. Using fallback: '%s'", postal_code, updated_location)
        #     return updated_location, address_id

        # Single or multiple rows
        if df.empty:
            return None, None

        if df.shape[0] == 1:
            row = df.iloc[0]
        else:
            match_index = self.match_civic_number(df, numbers)
            if match_index is None or match_index not in df.index:
                logging.warning("populate_from_db_or_fallback(): No valid match found.")
                return None, None
            row = df.loc[match_index]

        updated_location = self.format_address_from_db_row(row)

        address_dict = self.create_address_dict(
            updated_location, str(row.civic_no) if row.civic_no else None, row.official_street_name,
            row.official_street_type, None, row.mail_mun_name, row.mail_prov_abvn, row.mail_postal_code, 'CA'
        )
        address_id = self.get_address_id(address_dict)
        logging.info("Populated from DB for postal code '%s': '%s'", postal_code, updated_location)
        return updated_location, address_id

    def fallback_with_municipality(self, location_str):
        """
        If no postal code is found, tries Google for the municipality.
        If it's in BC, returns an updated location and partial address_id.
        Otherwise returns (None, None).

        Returns:
            (updated_location, address_id)
        """
        municipality = self.get_municipality(location_str, self.google_api_key)
        if not municipality:
            return None, None

        with open(self.config['input']['municipalities'], 'r', encoding='utf-8') as f:
            muni_list = [line.strip() for line in f if line.strip()]
        if municipality in muni_list:
            updated_location = f"{location_str}, {municipality}, BC, CA"
            updated_location = updated_location.replace('None', '').replace(',,', ',').strip()
            address_dict = self.create_address_dict(
                updated_location, None, None, None, None, municipality, 'BC', None, 'CA'
            )
            address_id = self.get_address_id(address_dict)
            logging.info("Fallback with municipality: '%s'", updated_location)
            return updated_location, address_id
        return None, None
    

    def match_civic_number(self, df, numbers):
        """
        Given a DataFrame of addresses and numeric strings from the location,
        attempts to match the first number to a civic_no. Returns the best row index,
        or the first row if no match is found.
        """
        if df.empty:
            logging.warning("match_civic_number(): Received empty DataFrame.")
            return None
        
        if not numbers:
            return df.index[0]
        for i, addr_row in df.iterrows():
            if addr_row.civic_no is not None:
                try:
                    if int(numbers[0]) == int(addr_row.civic_no):
                        return i
                except ValueError:
                    continue

        return df.index[0]
    

    def format_address_from_db_row(self, db_row):
        """
        Constructs a formatted address string from a single DB row,
        explicitly including the city (mail_mun_name).
        """
        # Build the street portion
        parts = [
            str(db_row.civic_no) if db_row.civic_no else "",
            str(db_row.civic_no_suffix) if db_row.civic_no_suffix else "",
            db_row.official_street_name or "",
            db_row.official_street_type or "",
            db_row.official_street_dir or ""
        ]
        street_address = " ".join(part for part in parts if part).strip()

        # Insert the city if available
        city = db_row.mail_mun_name or ""

        # Construct final location string
        formatted = (
            f"{street_address}, "
            f"{city}, "                      # <─ Include municipality
            f"{db_row.mail_prov_abvn or ''}, "
            f"{db_row.mail_postal_code or ''}, CA"
        )
        # Clean up spacing
        formatted = re.sub(r'\s+,', ',', formatted)
        formatted = re.sub(r',\s+,', ',', formatted)
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        return formatted

    def is_canadian_postal_code(self, postal_code):
        """
        Checks if a postal code matches the Canadian format: A1A 1A1 (with optional space).
        """
        pattern = r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$'
        return bool(re.match(pattern, postal_code.strip()))
    

    def dedup(self):
        """
        Removes duplicates from the 'events' table in the database.

        For the 'events' table, duplicates are identified based on the combination of
        'Name_of_the_Event' and 'Start_Date'. Only the latest entry is kept.

        For the 'urls' table, duplicates are identified based on the 'link' column.
        Only the latest entry is kept.
        """
        try:
            # Deduplicate 'events' table based on 'Name_of_the_Event' and 'Start_Date'
            dedup_events_query = """
                DELETE FROM events e1
                USING events e2
                WHERE e1.event_id < e2.event_id
                    AND e1.address_id = e2.address_id
                    AND e1.start_date = e2.start_date
                    AND e1.end_date = e2.end_date
                    AND ABS(EXTRACT(EPOCH FROM (e1.start_time - e2.start_time))) <= 900
                    AND ABS(EXTRACT(EPOCH FROM (e1.end_time - e2.end_time))) <= 900;
            """
            deleted_count = self.execute_query(dedup_events_query)
            logging.info("def dedup(): Deduplicated events table successfully. Rows deleted: %d", deleted_count)

        except Exception as e:
            logging.error("def dedup(): Failed to deduplicate tables: %s", e)


    def fetch_events_dataframe(self):
        """
        Fetch all events from the database and return as a sorted DataFrame.
        """
        query = "SELECT * FROM events"
        df = pd.read_sql(query, self.conn)
        df.sort_values(by=['start_date', 'start_time'], inplace=True)
        return df

    def decide_preferred_row(self, row1, row2):
        """
        Decide which of the two rows to keep based on the specified criteria:
        a. Prefer the one with a non-empty URL.
        b. If neither has a URL, prefer the one with more filled columns.
        c. If tied, keep the most recent based on time_stamp.
        
        Returns:
            tuple: (preferred_row, other_row)
        """
        # Prefer row with URL
        if row1['url'] and not row2['url']:
            return row1, row2
        if row2['url'] and not row1['url']:
            return row2, row1

        # Count filled columns (excluding event_id)
        filled_columns = lambda row: row.drop(labels='event_id').count()
        count1 = filled_columns(row1)
        count2 = filled_columns(row2)
        
        if count1 > count2:
            return row1, row2
        elif count2 > count1:
            return row2, row1
        else:
            # If still tied, choose the most recent based on time_stamp
            if row1['time_stamp'] >= row2['time_stamp']:
                return row1, row2
            else:
                return row2, row1

    def update_preferred_row_from_other(self, preferred, other, columns):
        """
        Update missing columns in the preferred row using values from the other row.
        """
        for col in columns:
            if pd.isna(preferred[col]) or preferred[col] == '':
                if not (pd.isna(other[col]) or other[col] == ''):
                    preferred[col] = other[col]
        return preferred


    def fuzzy_duplicates(self):
        """
        Identify and remove fuzzy duplicates from the events table.
        
        Steps:
        1. Sort events by start_date, start_time.
        2. For groups with same start_date and start_time:
             a. Fuzzy match event_name for events in group.
             b. If fuzzy score > 80, decide which row to keep:
                i. Prefer row with URL.
                ii. If neither has URL, keep row with more filled columns.
                iii. If tied, keep most recent based on time_stamp.
             c. Update kept row with missing data from duplicate.
             d. Update the database: update kept row, delete duplicate row.
        """
        # Fetch and sort events using self.conn
        events_df = self.fetch_events_dataframe()
        grouped = events_df.groupby(['start_date', 'start_time'])

        for _, group in grouped:
            if len(group) < 2:
                continue  # Skip if no duplicates possible

            # Check for duplicates within the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row_i = group.iloc[i].copy()
                    row_j = group.iloc[j].copy()
                    score = fuzz.ratio(row_i['event_name'], row_j['event_name'])
                    if score > 80:
                        # Decide which row to keep
                        preferred, other = self.decide_preferred_row(row_i, row_j)

                        # Update preferred row with missing columns from other row
                        preferred = self.update_preferred_row_from_other(preferred, other, events_df.columns)

                        # Prepare update query parameters
                        update_columns = [col for col in events_df.columns if col != 'event_id']
                        set_clause = ", ".join([f"{col} = :{col}" for col in update_columns])
                        update_query = f"""
                            UPDATE events
                            SET {set_clause}
                            WHERE event_id = :event_id
                        """

                        update_params = {}
                        for col in update_columns:
                            value = preferred[col]
                            if isinstance(value, (np.generic, np.ndarray)):
                                try:
                                    value = value.item() if hasattr(value, 'item') else value.tolist()
                                except Exception:
                                    pass
                            update_params[col] = value
                        update_params['event_id'] = int(preferred['event_id'])

                        # Execute update query using self.execute_query
                        self.execute_query(update_query, update_params)
                        logging.info("fuzzy_duplicates: Kept row with event_id %s", preferred['event_id'])

                        # Delete duplicate row from database
                        delete_query = "DELETE FROM events WHERE event_id = :event_id"
                        self.execute_query(delete_query, {'event_id': int(other['event_id'])})
                        logging.info("fuzzy_duplicates: Deleted duplicate row with event_id %s", other['event_id'])

        logging.info("fuzzy_duplicates: Fuzzy duplicate removal completed successfully.")


    def delete_old_events(self):
        """
        Deletes events older than a specified number of days from the 'events' table.

        The number of days is specified in the configuration under 'clean_up' -> 'old_events'.
        """
        try:
            days = int(self.config['clean_up']['old_events'])
            delete_query = """
                DELETE FROM events
                WHERE End_Date < CURRENT_DATE - INTERVAL '%s days';
            """ % days
            self.execute_query(delete_query)
            deleted_count = self.execute_query(delete_query)
            logging.info(f"delete_old_events: Deleted {deleted_count} events older than {days} days.")
        except Exception as e:
            logging.error("delete_old_events: Failed to delete old events: %s", e)


    def delete_likely_dud_events(self):
        """
        1. If the event source, dance_style, and url == '', delete the event, UNLESS it has an address_id then keep it.
        2. Drop events outside of British Columbia (BC).
            a. However, not all events will have an address_id (Primary Key in address table, Foreign Key in events table).
            b. Also, not all rows in the address table will have a province_or_state.
            c. If they do have a province_or_state and it is not 'BC' then delete the event.
        3. Drop events that are not in Canada.
            a. However, NOT all events will have an address_id (Primary Key in address table, Foreign Key in events table). 
            b. Also, not all rows in the address table will have a country_id. 
            c. If they do have a country_id and it is not 'CA' then delete the event. 
        4. Delete rows in events where dance_style and url are == '' AND event_type == 'other' AND location IS NULL and description IS NULL
        """

        # 1. Delete events where source, dance_style, and url are empty, unless they have an address_id
        delete_query_1 = """
        DELETE FROM events
        WHERE source = :source
        AND dance_style = :dance_style
        AND url = :url
        AND address_id IS NULL
        RETURNING event_id;
        """
        params = {
            'source': '',
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }

        deleted_events = self.execute_query(delete_query_1, params)
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events with empty source, dance_style, and url, and no address_id.", deleted_count)

        # 2. Delete events outside of British Columbia (BC)
        delete_query_2 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE province_or_state IS NOT NULL
            AND province_or_state != :province_or_state
        )
        RETURNING event_id;
        """
        params = {
            'province_or_state': 'BC'
            }
        
        deleted_events = self.execute_query(delete_query_2, params)
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events outside of British Columbia (BC).", deleted_count)

        # 3. Delete events that are not in Canada
        delete_query_3 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE country_id IS NOT NULL
            AND country_id != :country_id
        )
        RETURNING event_id;
        """
        params = {
            'country_id': 'CA'
            }
        
        deleted_events = self.execute_query(delete_query_3, params)
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info("delete_likely_dud_events: Deleted %d events that are not in Canada (CA).", deleted_count)

        # 4. Delete rows in events where dance_style and url are == '' AND event_type == 'other' AND location IS NULL and description IS NULL
        delete_query_4 = """
        DELETE FROM events
        WHERE dance_style = :dance_style
            AND url = :url
            AND event_type = :event_type
            AND location IS NULL
            AND description IS NULL
        RETURNING event_id;
        """
        params = {
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }
        
        deleted_events = self.execute_query(delete_query_4, params)
        deleted_count = len(deleted_events) if deleted_events else 0
        logging.info(
            "def delete_likely_dud_events(): Deleted %d events with empty "
            "dance_style, url, event_type 'other', and null location "
            "and description.",
            deleted_count
        )
        

    def delete_event(self, url, event_name, start_date):
        """
        Deletes an event from the 'events' table and the corresponding URL from the 'urls' table.

        Args:
            url (str): The URL of the event to be deleted.
            event_name (str): The name of the event to be deleted.
            start_date (str): The start date of the event to be deleted.
        """
        try:
            logging.info("delete_event: Deleting event with URL: %s, Event Name: %s, Start Date: %s", url, event_name, start_date)

            # Delete the event from 'events' table
            delete_event_query = """
                DELETE FROM events
                WHERE Name_of_the_Event = :event_name
                  AND Start_Date = :start_date;
            """
            params = {'event_name': event_name, 'start_date': start_date}
            self.execute_query(delete_event_query, params)
            logging.info("delete_event: Deleted event from 'events' table.")

        except Exception as e:
            logging.error("delete_event: Failed to delete event: %s", e)


    def delete_events_with_nulls(self):
        """
        Deletes events with start_date and start_time being null in the 'events' table.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE (start_date IS NULL AND start_time IS NULL) OR 
            (start_time IS NULL AND end_time IS NULL);
            """
            self.execute_query(delete_query)
            deleted_count = self.execute_query(delete_query) or 0
            logging.info("def delete_events_with_nulls(): Deleted %d events where (start_date and start_time are null).", deleted_count)
        except Exception as e:
            logging.error("def delete_events_with_nulls(): Failed to delete events with nulls: %s", e)


    def delete_event_with_event_id(self, event_id):
        """
        Deletes an event from the 'events' table based on the event_id.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE event_id = :event_id;
            """
            params = {'event_id': event_id}
            self.execute_query(delete_query, params)
            logging.info("delete_event_with_event_id: Deleted event with event_id %d successfully.", event_id)
        except Exception as e:
            logging.error("delete_event_with_event_id: Failed to delete event with event_id %d: %s", event_id, e)

    
    def delete_multiple_events(self, event_ids):
        """
        Deletes multiple events from the 'events' table based on a list of event_ids.

        Args:
            event_ids (list): List of event_id(s) to delete.

        Returns:
            bool: True if all deletions were successful, False otherwise.
        """
        if not event_ids:
            logging.warning("delete_multiple_events: No event_ids provided for deletion.")
            return False

        success_count = 0
        for event_id in event_ids:
            try:
                self.delete_event_with_event_id(event_id)
                success_count += 1
            except Exception as e:
                logging.error("delete_multiple_events: Error deleting event_id %d: %s", event_id, e)

        logging.info("delete_multiple_events: Deleted %d out of %d events.", success_count, len(event_ids))
        
        return success_count == len(event_ids)  # Return True if all deletions were successful


    def multiple_db_inserts(self, table_name, values):
        """
        Inserts or updates multiple records in the specified table.

        Parameters:
            table_name (str): The name of the table to insert/update.
            values (list of dict): List of dictionaries where each dict represents a row to insert/update.
        """
        if not values:
            logging.info("multiple_db_inserts(): No values to insert or update.")
            return

        try:
            table = Table(table_name, self.metadata, autoload_with=self.conn)
            with self.conn.begin() as conn:
                for row in values:
                    stmt = insert(table).values(row)
                    if table_name == "address":
                        pk = "address_id"
                    elif table_name == "events":
                        pk = "event_id"
                    else:
                        raise ValueError(f"Unsupported table: {table_name}")
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[pk],
                        set_={col: stmt.excluded[col] for col in row.keys() if col != pk}
                    )
                    conn.execute(stmt)
            logging.info(f"multiple_db_inserts(): Successfully inserted/updated {len(values)} rows in {table_name} table.")
        except Exception as e:
            logging.error(f"multiple_db_inserts(): Error inserting/updating records in {table_name} table - {e}")


    def delete_address_with_address_id(self, address_id):
        """
        Deletes an address from the 'address' table based on the address_id.
        """
        try:
            # Convert address_id to native Python int
            address_id = int(address_id)
            delete_query = """
                DELETE FROM address
                WHERE address_id = :address_id;
            """
            params = {'address_id': address_id}
            self.execute_query(delete_query, params)
            logging.info("delete_address_with_address_id: Deleted address with address_id %d successfully.", address_id)
        except Exception as e:
            logging.error("delete_address_with_address_id: Failed to delete address with address_id %d: %s", address_id, e)


    def is_foreign(self):
        """
        Determines which events are likely not in British Columbia (BC) by comparing the event location
        against a list of known municipalities and street names, returning all event columns for context.
        
        The method:
            1. Loads all events from the "events" table into a DataFrame.
            2. Loads street names from the "address" table into a DataFrame.
            3. Reads a list of municipalities from a text file.
            4. Filters the events DataFrame to include only those rows whose 'location' field does not 
            contain any municipality name (from muni_list) or street name (from street_list), case-insensitively.
            5. Deletes the identified events from the "events" table.
        
        Returns:
            pd.DataFrame: A DataFrame containing all columns from the events table for events that are
                        likely not located in BC.
        """
        # 1. Load events from the database.
        events_sql = "SELECT * FROM events"
        events_df = pd.read_sql(events_sql, self.conn)
        logging.info("is_foreign(): Loaded %d records from events.", len(events_df))

        # 2. Load address street names.
        address_sql = "SELECT street_name FROM address"
        address_df = pd.read_sql(address_sql, self.conn)
        street_list = address_df['street_name'].tolist()
        logging.info("is_foreign(): Loaded %d street names from address.", len(street_list))

        # 3. Read municipalities from file.
        with open(self.config['input']['municipalities'], 'r', encoding='utf-8') as f:
            muni_list = [line.strip() for line in f if line.strip()]
        logging.info("is_foreign(): Loaded %d municipalities from file.", len(muni_list))

        # Read countries from .csv file
        countries_df = pd.read_csv(self.config['input']['countries'])
        countries_list = countries_df['country_names'].tolist()

        # 4. Filtering logic: if the location or description contains a foreign country, mark it as foreign.
        def is_foreign_location(row):
            location = row['location'] if row['location'] else ''
            description = row['description'] if row['description'] else ''
            source = row['source'] if row['source'] else ''
            combined_text = f"{location} {description} {source}".lower()
            
            # Check if any known foreign country appears in the text.
            country_found = any(country.lower() in combined_text for country in countries_list)
            # Check if any BC municipality appears.
            muni_found = any(muni.lower() in combined_text for muni in muni_list if muni)
            
            # If a foreign country is found, consider it foreign.
            if country_found:
                return True
            # If a BC municipality is found, it's likely not foreign.
            if muni_found:
                return False
            # Default to False if neither is found.
            return False

        # Create a boolean mask for events that are likely foreign.
        mask = events_df.apply(is_foreign_location, axis=1)
        foreign_events_df = events_df[mask].copy()
        logging.info("is_foreign(): Found %d events likely in foreign countries or municipalities.", len(foreign_events_df))

        # 5. Delete the identified events from the database.
        if not foreign_events_df.empty:
            event_ids = foreign_events_df['event_id'].tolist()
            self.delete_multiple_events(event_ids)
            logging.info("is_foreign(): Deleted %d events from the database.", len(event_ids))

        return foreign_events_df
    

    def groupby_source(self):
        """
        Runs a GROUP BY query against the events table to get event counts per source.
        """
        query = "SELECT source, COUNT(*) AS counted FROM events GROUP BY source ORDER BY counted DESC"
        groupby_df = pd.read_sql_query(query, self.conn)
        logging.info(f"def groupby_source(): Retrieved groupby results from events table.")
        return groupby_df


    def count_events_urls_start(self, file_name):
        """
        Counts the number of events and URLs in the database at the start time and returns a DataFrame with the results.

        Args:
            file_name (str): The name of the .py file initiating the count.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
            - file_name (str): The name of the .py file.
            - start_time (datetime): The timestamp when the count was initiated.
            - events_count_start (int): The number of events in the database at the start time.
            - urls_count_start (int): The number of URLs in the database at the start time.
        """

        # Add a df for the name of the .py file
        file_name_df = pd.DataFrame([[file_name]], columns=["file_name"])

        # Get start_time
        start_time = datetime.now()
        start_time_df = pd.DataFrame([[start_time]], columns=["start_time"])

        # Count events in db at start
        sql = "SELECT COUNT(*) as events_count_start FROM events"
        events_count_start_df = pd.read_sql(sql, self.conn)

        # Count events in db at start
        sql = "SELECT COUNT(*) as urls_count_start FROM urls"
        urls_count_start_df = pd.read_sql(sql, self.conn)

        # Concatenate the above dataframes into a new dataframe called start_df
        start_df = pd.concat([file_name_df, start_time_df, events_count_start_df, urls_count_start_df], axis=1)
        start_df.columns = ['file_name', 'start_time_df', 'events_count_start', 'urls_count_start']

        return start_df
    

    def count_events_urls_end(self, start_df, file_name):
        """
        Counts the number of events and URLs in the database at the end of a process, compares them with the counts at the start, 
        and writes the results to a CSV file.

        Parameters:
        start_df (pd.DataFrame): A DataFrame containing the initial counts of events and URLs, as well as the file name and start time.

        Returns:
        None

        The function performs the following steps:
        1. Executes SQL queries to count the number of events and URLs in the database at the end of the process.
        2. Concatenates the initial counts with the new counts into a single DataFrame.
        3. Calculates the number of new events and URLs added to the database.
        4. Adds a timestamp and calculates the elapsed time since the start.
        5. Writes the resulting DataFrame to a CSV file, appending if the file already exists.
        6. Logs the file name where the statistics were written.
        """

        # Count events in db at end
        sql = "SELECT COUNT(*) as events_count_end FROM events"
        events_count_end_df = pd.read_sql(sql, self.conn)

        # Count events in db at end
        sql = "SELECT COUNT(*) as urls_count_end FROM urls"
        urls_count_end_df = pd.read_sql(sql, self.conn)

        # Create the dataframe
        results_df = pd.concat([start_df, events_count_end_df, urls_count_end_df], axis=1)
        results_df.columns = ['file_name', 'start_time_df', 'events_count_start', 'urls_count_start', 'events_count_end', 'urls_count_end']
        results_df['new_events_in_db'] = results_df['events_count_end'] - results_df['events_count_start']
        results_df['new_urls_in_db'] = results_df['urls_count_end'] - results_df['urls_count_start']
        results_df['time_stamp'] = datetime.now()
        results_df['elapsed_time'] = results_df['time_stamp'] - results_df['start_time_df']

        # Write the df to a csv file
        output_file = self.config['output']['events_urls_diff']
        if not os.path.isfile(output_file):
            results_df.to_csv(output_file, index=False)
        else:
            results_df.to_csv(output_file, mode='a', header=False, index=False)

        logging.info(f"def count_events_urls_end(): Wrote events and urls statistics to: {file_name}")


    def dedup_address(self):
        """
        Deduplicates the 'address' table based on the street_number and street_name columns.
        """
        # Create sql statement to select all rows where street_number and street_name are not null or empty
        sql = """
                SELECT *
                FROM address
                WHERE street_number IS NOT NULL 
                AND street_number <> ''
                AND street_name IS NOT NULL 
                AND street_name <> '';
            """
        df = pd.read_sql(sql, self.conn)

        # Get duplicates
        duplicates = df[df.duplicated(subset=['street_number', 'street_name'], keep='last')]
        if duplicates.empty:
            logging.info("dedup_address(): No duplicates found in the 'address' table.")

        else:
            # Get the address_ids of the duplicates
            address_ids = duplicates['address_id'].tolist()

            # Delete the duplicates
            for address_id in address_ids:
                self.delete_address_with_address_id(address_id)
            logging.info("dedup_address(): Deleted %d duplicate addresses.", len(address_ids))


    def get_address_update_for_event(self, event_id, location, address_df):
        """
        Given an event's ID, its location, and the address DataFrame, 
        extract the street number from the location using regex and 
        then look for an address row in address_df that has a matching 
        street_number and whose street_name appears in the location.
        
        Returns a list with a single update dictionary (if a match is found)
        or an empty list if no match is found.
        """
        updates = []
        if location is None:
            return updates
        match = re.search(r'\d+', location)
        if match:
            extracted_number = match.group()
            matching_addresses = address_df[address_df['street_number'] == extracted_number]
            for _, addr_row in matching_addresses.iterrows():
                if pd.notnull(addr_row['street_name']) and addr_row['street_name'] in location:
                    new_address_id = addr_row['address_id']
                    updates.append({
                        "event_id": event_id,
                        "address_id": new_address_id
                    })
                    break  # Stop after the first match is found.
        return updates

    def fix_address_id_in_events(self):
        """
        Goes through the events table and ensures that the address_id in the events table 
        matches the correct address_id from the address table.
        For each event:
        - Extracts the street number from the event's location using regex.
        - If a match is found, looks for a matching row in the address table based on street_number.
        - If an address row is found and its street_name is present in the location,
        the event's address_id is updated.
        - The event_id and updated address_id are collected in a list for a bulk update.
        """
        # Read the events table into a DataFrame.
        sql = "SELECT * FROM events"
        events_df = pd.read_sql(sql, self.conn)
        
        # Read the address table into a DataFrame.
        sql = "SELECT * FROM address ORDER BY street_number, street_name"
        address_df = pd.read_sql(sql, self.conn)
        
        # List to hold updates for the events table.
        events_address_ids_to_be_updated_list = []
        
        # Process each event row.
        for row in events_df.itertuples(index=False):
            event_id = row.event_id
            location = row.location
            
            updates = self.get_address_update_for_event(event_id, location, address_df)
            events_address_ids_to_be_updated_list.extend(updates)
        
        logging.info("fix_address_id_in_events: events_address_ids_to_be_updated_list: %s", len(events_address_ids_to_be_updated_list))
        
        # Bulk update the events table using the provided multiple_db_inserts method.
        self.multiple_db_inserts("events", events_address_ids_to_be_updated_list)


    def stale_date(self, url):
        """
        Check whether this URL's most recent event is “stale” (older than our allowed threshold).

        Steps:
        1. Query the `events` table for all rows where `url = :url`, ordered by `start_date` descending, limit 1.
        2. If there are no events for this URL, return True (i.e. it's “stale” because nothing exists yet).
        3. Otherwise, take that single most-recent `start_date`.
            a. Convert it into a Python date object.
        4. Compute `cutoff_date = (today's date) - (config['clean_up']['old_events'] days)`.
            a. If `latest_start_date < cutoff_date`, then the URL is older than our threshold ⇒ return True.
            b. Otherwise, return False (it's still fresh).
        """
        try:
            # 1. Fetch the most recent start_date for this URL
            query = """
                SELECT start_date
                FROM events_history
                WHERE url = :url
                ORDER BY start_date DESC
                LIMIT 1;
            """
            params = {'url': url}
            result = self.execute_query(query, params)

            # 2. If no rows returned, nothing has been recorded for this URL ⇒ treat as “stale”
            if not result:
                return True

            latest_start_date = result[0][0]
            # 3. If, for some reason, start_date is NULL in the DB, treat as “stale”
            if latest_start_date is None:
                return True

            # 3a. Convert whatever was returned into a Python date
            #     (pd.to_datetime handles string, datetime, or pandas.Timestamp)
            latest_date = pd.to_datetime(latest_start_date).date()

            # 4. Compute cutoff_date = today – N days
            days_threshold = int(self.config['clean_up']['old_events'])
            cutoff_date = datetime.now().date() - timedelta(days=days_threshold)

            # 4a. If the event’s date is older than cutoff_date, it’s stale → return True
            return latest_date < cutoff_date

        except Exception as e:
            logging.error(f"stale_date: Error checking stale date for url {url}: {e}")
            # In case of any error, default to True (safer to re‐process)
            return True


    def should_process_url(self, url):
        """
        Decide whether to process the given URL based on:
        1. The most recent 'relevant' value in self.urls_df.
        2. The hit_ratio in self.urls_gb if the last 'relevant' was False.

        Returns True if:
        - The URL has never been seen before (no rows in self.urls_df), or
        - The last time we saw it, 'relevant' was True, or
        - The last time was False but hit_ratio > 0.1.
        Otherwise returns False.
        """
        # 1. Filter all rows for this URL
        df_url = self.urls_df[self.urls_df['link'] == url]
        # If we've never recorded this URL, process it
        if df_url.empty:
            logging.info(f"should_process_url: URL {url} has never been seen before, processing it.")
            return True

        # 2. Look at the most recent "relevant" value
        last_relevant = df_url.iloc[-1]['relevant']
        if last_relevant and self.stale_date:
            logging.info(f"should_process_url: URL {url} was last seen as relevant, processing it.")
            return True

        # 3. Last was False → check hit_ratio in self.urls_gb
        hit_row = self.urls_gb[self.urls_gb['link'] == url]

        if not hit_row.empty:
            # Extract scalars from the grouped DataFrame
            hit_ratio = hit_row.iloc[0]['hit_ratio']
            crawl_trys = hit_row.iloc[0]['crawl_try']

            if hit_ratio > 0.1 or crawl_trys <= 3:
                logging.info(
                    "should_process_url: URL %s was last seen as not relevant "
                    "but hit_ratio (%.2f) > 0.1 or crawl_try (%d) ≤ 3, processing it.",
                    url, hit_ratio, crawl_trys
                )
                return True

        # 4. Otherwise, do not process this URL
        logging.info(f"should_process_url: URL {url} does not meet criteria for processing, skipping it.")
        return False
    

    def avoid_domains(self, url):
        pass


    def update_dow_date(self, event_id: int, corrected_date) -> bool:
        """
        Updates both start_date and end_date of the event identified by event_id
        to corrected_date. Returns True if the UPDATE succeeded.
        """
        update_query = """
            UPDATE events
               SET start_date = :corrected_date,
                   end_date   = :corrected_date
             WHERE event_id  = :event_id
        """
        params = {
            'corrected_date': corrected_date,
            'event_id':        event_id
        }
        self.execute_query(update_query, params)
        return True


    def check_dow_date_consistent(self) -> None:
        """
        1. SELECT event_id, start_date, day_of_week, end_date FROM events.
        2. For each tuple (event_id, start_date, day_of_week), check if
           start_date.weekday() matches the stored day_of_week. If not, shift
           ±k days (k in [-3..+3]) to hit the correct weekday.
        3. Call update_dow_date(...) whenever a shift is needed, which now
           updates both start_date and end_date.
        4. Log every adjustment.
        """
        select_query = """
            SELECT event_id, start_date, day_of_week
              FROM events
        """
        rows = self.execute_query(select_query)
        # rows is a list of tuples: (event_id, start_date, day_of_week)

        name_to_wd = {
            'monday':    0,
            'tuesday':   1,
            'wednesday': 2,
            'thursday':  3,
            'friday':    4,
            'saturday':  5,
            'sunday':    6
        }

        for row in rows:
            event_id  = row[0]
            orig_date = row[1]  # DATE
            dow_text  = row[2]  # TEXT

            if orig_date is None or not dow_text:
                continue

            key = dow_text.strip().lower()
            if key not in name_to_wd:
                logging.warning(
                    "check_dow_date_consistent: event_id %s has unrecognized day_of_week '%s'; skipping.",
                    event_id, dow_text
                )
                continue

            target_wd  = name_to_wd[key]
            current_wd = orig_date.weekday()

            if current_wd == target_wd:
                continue  # no change needed

            # Compute minimal shift in [-3..+3] so that (orig_date + shift).weekday() == target_wd
            diff_mod_7 = (target_wd - current_wd + 7) % 7
            if diff_mod_7 <= 3:
                shift = diff_mod_7
            else:
                shift = diff_mod_7 - 7

            corrected_date = orig_date + timedelta(days=shift)

            if corrected_date != orig_date:
                success = self.update_dow_date(event_id, corrected_date)
                if success:
                    logging.info(
                        "check_dow_date_consistent: event_id %d: "
                        "start_date (and end_date) changed from %s to %s "
                        "(day_of_week was '%s').",
                        event_id,
                        orig_date.isoformat(),
                        corrected_date.isoformat(),
                        dow_text
                    )
                else:
                    logging.error(
                        "check_dow_date_consistent: failed to update event_id %d", event_id
                    )


    def driver(self):
        """
        Main driver function to perform database operations.
        """
        if config['testing']['drop_tables'] == True:
            self.create_tables()
        else:
            self.dedup()
            self.delete_old_events()
            self.delete_events_with_nulls()
            self.delete_likely_dud_events()
            self.fuzzy_duplicates()
            self.is_foreign()
            self.dedup_address()
            self.fix_address_id_in_events()
            #self.check_dow_date_consistent()

        # Close the database connection
        self.conn.dispose()  # Using dispose() for SQLAlchemy Engine
        logging.info("driver(): Database operations completed successfully.")
        

if __name__ == "__main__":

    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set up logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='a',  # Changed to append mode to preserve logs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logging.info("\n\ndb.py starting...")

    start_time = datetime.now()
    logging.info("\n\nMain: Started the process at %s", start_time)

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)
    
    # Get the file name of the code that is running
    file_name = os.path.basename(__file__)

    # Count events and urls before cleanup
    start_df = db_handler.count_events_urls_start(file_name)

    # Perform deduplication and delete old events
    db_handler.driver()

    # Count events and urls after cleanup
    db_handler.count_events_urls_end(start_df, file_name)

    end_time = datetime.now()
    logging.info("Main: Finished the process at %s", end_time)
    logging.info("Main: Total time taken: %s", end_time - start_time)
