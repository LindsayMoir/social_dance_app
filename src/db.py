"""
db.py
This module provides a DatabaseHandler class for managing database connections and operations.
It includes methods for creating tables, writing URLs and events to the database, and handling address deduplication.
It also includes methods for loading blacklisted domains and checking URLs against them.
It uses SQLAlchemy for database interactions and pandas for data manipulation.
It supports both local and Render environments, with configurations loaded from environment variables.
It also includes methods for fuzzy matching addresses and deduplicating the address table.
                    WHERE address_id = :old_id
"""

from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
from rapidfuzz import fuzz
import json
import logging
import numpy as np
import os
import pandas as pd
from rapidfuzz.fuzz import ratio
import re  # Added missing import
import requests
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, List, Dict
import sys
import yaml
import warnings


class DatabaseHandler():
    def __init__(self, config):
        """
        Initializes the DatabaseHandler instance with the provided configuration.

        This constructor sets up the database connections based on the environment (Render or local),
        loads the blacklist domains, initializes SQLAlchemy metadata, retrieves the Google API key,
        and prepares DataFrames for URL analysis.

            config (dict): Configuration parameters for the database connection.

        Raises:
            ConnectionError: If the database connection could not be established.

        Side Effects:
            - Loads blacklist domains.
            - Establishes connections to the main and address databases.
            - Reflects the existing database schema into SQLAlchemy metadata.
            - Retrieves the Google API key from environment variables.
            - Creates a DataFrame from the URLs table and computes grouped statistics for URL usefulness.
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


    def set_llm_handler(self, llm_handler):
        """
        Inject an instance of LLMHandler after both classes are constructed.
        """
        self.llm_handler = llm_handler
        

    def load_blacklist_domains(self):
        """
        Loads a set of blacklisted domains from a CSV file specified in the configuration.

        The CSV file path is retrieved from self.config['constants']['black_list_domains'].
        The CSV is expected to have a column named 'Domain'. All domain names are converted
        to lowercase and stripped of whitespace before being added to the blacklist set.

        The resulting set is stored in self.blacklisted_domains.

        Logs the number of loaded blacklisted domains at the INFO level.
        """
        csv_path = self.config['constants']['black_list_domains']
        df = pd.read_csv(csv_path)
        self.blacklisted_domains = set(df['Domain'].str.lower().str.strip())
        logging.info(f"Loaded {len(self.blacklisted_domains)} blacklisted domains.")

    def avoid_domains(self, url):
        """
        Check if the given URL contains any blacklisted domain.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL contains any domain from the blacklist, False otherwise.

        Note:
            The check is case-insensitive.
        """
        """ Check if URL contains any blacklisted domain. """
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.blacklisted_domains)
    

    def get_db_connection(self):
        """
        Establish and return a SQLAlchemy engine for the PostgreSQL database.

        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine instance if connection is successful.
            None: If the connection could not be established.
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
        Creates the required tables in the database if they do not already exist.

        Tables created:
            - urls
            - events
            - address
            - runs

        If config['testing']['drop_tables'] is True, existing tables are dropped before creation,
        and the current 'events' table is backed up to 'events_history'.

        This method ensures all necessary tables for the application are present and logs the process.
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
        Creates and returns a pandas DataFrame from the 'urls' table in the database.

        Returns:
            pandas.DataFrame: DataFrame containing all rows from the 'urls' table.
        Notes:
            - If the table is empty or an error occurs, returns an empty DataFrame.
            - Logs the number of rows loaded or any errors encountered.
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
            params (dict, optional): Dictionary of parameters for parameterized queries.

        Returns:
            list: List of rows (as tuples) if the query returns rows.
            int: Number of rows affected for non-select queries.
            None: If the query fails or there is no database connection.
        """
        if self.conn is None:
            logging.error("execute_query: No database connection available.")
            return None

        # Handle NaN values in params (such as address_id)
        if params:
            for key, value in params.items():
                if isinstance(value, (list, np.ndarray, pd.Series)):
                    if pd.isna(value).any():
                        params[key] = None
                else:
                    if pd.isna(value):
                        params[key] = None

        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(query), params or {})

                if result.returns_rows:
                    rows = result.fetchall()
                    logging.info(
                        "execute_query(): query returned %d rows", 
                        len(rows)
                    )
                    return rows
                else:
                    affected = result.rowcount
                    connection.commit()
                    logging.info(
                        "execute_query(): non-select query affected %d rows", 
                        affected
                    )
                    return affected

        except SQLAlchemyError as e:
            logging.error(
                "execute_query(): Query execution failed (%s)\nQuery was: %s", 
                e, query
            )
            return None

    
    def close_connection(self):
        """
        Closes the database connection if it exists.

        This method attempts to properly dispose of the current database connection.
        If the connection is successfully closed, an informational log message is recorded.
        If an error occurs during the closing process, the exception is logged as an error.
        If there is no active connection, a warning is logged indicating that there is no connection to close.
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
        Appends a new URL activity record to the 'urls' table in the database.

        This method processes and normalizes the provided URL data, especially the 'keywords' field,
        ensuring it is stored as a clean, comma-separated string. The data is then inserted as a new row
        into the 'urls' table using pandas' DataFrame and SQL interface.

            url_row (tuple): A tuple containing the following fields in order:
                - link (str): The URL to be logged.
                - parent_url (str): The parent URL from which this link was found.
                - source (str): The source or context of the URL.
                - keywords (str | list | tuple | set): Associated keywords, which can be a string or an iterable.
                - relevant (bool | int): Indicator of relevance.
                - crawl_try (int): Number of crawl attempts.
                - time_stamp (str | datetime): Timestamp of the activity.

        Raises:
            Exception: Logs an error if the database insertion fails.

        Side Effects:
            - Appends a new row to the 'urls' table in the connected database.
            - Logs success or failure of the operation.
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
    

    def clean_up_address_basic(self, events_df):
        """
        Cleans events using only local DB and regex methods (no Foursquare).
        """
        logging.info("clean_up_address_basic(): Starting with shape %s", events_df.shape)

        address_df = pd.read_sql("SELECT * FROM address", self.conn)

        for index, row in events_df.iterrows():
            event_id = row.get('event_id')
            location = row.get('location')

            address_id, new_location = self.try_resolve_address(
                event_id, location, address_df=address_df, use_foursquare=False
            )

            if address_id:
                events_df.at[index, 'address_id'] = address_id
                events_df.at[index, 'location'] = new_location

        return events_df
    

    def try_resolve_address(self, event_id, location, address_df=None, use_foursquare=False):
        """
        Tries to resolve an address_id for a given location.
        Can optionally use Foursquare fallbacks.

        Args:
            event_id: ID of the event being processed.
            location: String location field from the event.
            address_df: Optional DataFrame of known addresses.
            use_foursquare: Whether to attempt external lookups via Foursquare.

        Returns:
            (address_id, new_location): Tuple containing the resolved address_id (or None)
                                        and the updated location string (or None).
        """
        if not location or pd.isna(location):
            return None, None
        
        if location is None:
            pass  # Keep it as None
        elif isinstance(location, str):
            location = location.strip()

        # 1. Try DB match using street number/name
        if address_df is not None:
            update_list = self.get_address_update_for_event(event_id, location, address_df)
            if update_list:
                return update_list[0]['address_id'], location

        # 2. Try local regex postal code extraction
        postal_code = self.extract_canadian_postal_code(location)
        if postal_code:
            updated_location, address_id = self.populate_from_db_or_fallback(location, postal_code)
            if address_id:
                return address_id, updated_location

        if use_foursquare:
            # 3. Try Foursquare postal code
            postal_code = self.get_postal_code_foursquare(location)
            if postal_code and self.is_canadian_postal_code(postal_code):
                updated_location, address_id = self.populate_from_db_or_fallback(location, postal_code)
                if address_id:
                    return address_id, updated_location

            # 4. Try Foursquare municipality fallback
            updated_location, address_id = self.fallback_with_municipality(location)
            if address_id:
                return address_id, updated_location

        # Final fallback: ensure consistent return type
        return None, None
            

    def get_address_update_for_event(self, event_id, location, address_df):
        """
        Given an event's ID, its location string, and a DataFrame of addresses, 
        this method attempts to extract the street number from the location using a regular expression. 
        It then searches the address DataFrame for a row where the 'street_number' matches the extracted number 
        and the 'street_name' is present in the location string.
        Parameters:
            event_id (Any): The unique identifier for the event.
            location (str): The location string containing address information.
            address_df (pd.DataFrame): A DataFrame containing address data with at least 
                'street_number', 'street_name', and 'address_id' columns.
        Returns:
            List[dict]: A list containing a single dictionary with 'event_id' and 'address_id' keys 
            if a matching address is found; otherwise, an empty list.
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
    

    def extract_canadian_postal_code(self, location_str):
        """
        Extracts a valid Canadian postal code from a given location string.

        This method uses a regular expression to search for a Canadian postal code pattern
        within the provided string. If a match is found, it removes any spaces from the
        postal code and validates it using the `is_canadian_postal_code` method. If the
        postal code is valid, it returns the cleaned postal code; otherwise, it returns None.

        Args:
            location_str (str): The input string potentially containing a Canadian postal code.

        Returns:
            str or None: The valid Canadian postal code without spaces if found and valid, otherwise None.
        """
        match = re.search(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d', location_str)
        if match:
            possible_pc = match.group().replace(' ', '')
            if self.is_canadian_postal_code(possible_pc):
                return possible_pc
        return None
    

    def is_canadian_postal_code(self, postal_code):
        """
        Determines whether the provided postal code matches the Canadian postal code format.

        A valid Canadian postal code follows the pattern: A1A 1A1, where 'A' is a letter and '1' is a digit.
        The space between the third and fourth characters is optional.

        Args:
            postal_code (str): The postal code string to validate.

        Returns:
            bool: True if the postal code matches the Canadian format, False otherwise.
        """
        pattern = r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$'
        return bool(re.match(pattern, postal_code.strip()))
    

    def populate_from_db_or_fallback(self, location_str, postal_code):
        """
        Attempts to populate a formatted address using a Canadian postal code by querying the local address database.
        If no match is found in the database, returns a fallback value (currently returns None, None).

        Args:
            location_str (str): The raw location string, typically containing civic number and street information.
            postal_code (str): The Canadian postal code to look up in the database.

            tuple:
                updated_location (str or None): The formatted address string if found, otherwise None.
                address_id (Any or None): The unique identifier for the address if found, otherwise None.

        Notes:
            - If multiple database rows match the postal code, attempts to match the civic number from the location string.
            - If no valid match is found, returns (None, None).
            - Logging is used to provide information and warnings about the lookup process.
            - Fallback logic for missing database entries.
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
        address_id = self.resolve_or_insert_address(address_dict)
        logging.info("Populated from DB for postal code '%s': '%s'", postal_code, updated_location)
        return updated_location, address_id
    

    def resolve_or_insert_address(self, parsed_address: dict) -> Optional[int]:
        """
        Resolves an address by checking for an exact match on street_number and street_name,
        followed by fuzzy matching on building_name. Inserts the address if no match is found.

        Args:
            parsed_address (dict): Dictionary of parsed address fields.

        Returns:
            int or None: The address_id of the matched or newly inserted address.
        """
        if not parsed_address:
            logging.info("resolve_or_insert_address: No parsed address provided.")
            return None

        building_name = (parsed_address.get("building_name") or "").strip()
        street_number = (parsed_address.get("street_number") or "").strip()
        street_name = (parsed_address.get("street_name") or "").strip()

        # Step 1: Only try street_number + street_name match if both are present
        if street_number and street_name:
            select_query = """
                SELECT address_id, building_name, street_number, street_name
                FROM address
                WHERE LOWER(street_number) = LOWER(:street_number)
                AND LOWER(street_name) = LOWER(:street_name)
            """
            params = {
                "street_number": street_number,
                "street_name": street_name,
            }
            street_matches = self.execute_query(select_query, params)

            for addr_id, b_name, s_num, s_name in street_matches or []:
                # Confirm fields actually match (avoid nulls acting as wildcard)
                if s_num.lower() == street_number.lower() and s_name.lower() == street_name.lower():
                    if building_name:
                        sim_score = ratio(building_name, b_name or "")
                        if sim_score >= 85:
                            logging.info(f"Street+fuzzy building_name match (score={sim_score}) → address_id={addr_id}")
                            return addr_id
                    else:
                        logging.info(f"Exact match on street_number and street_name (no building_name) → address_id={addr_id}")
                        return addr_id
        else:
            logging.info("resolve_or_insert_address: Missing street_number or street_name; skipping street match")

        # Step 2: Try fuzzy match on building_name across all known addresses
        if building_name:
            logging.info(f"resolve_or_insert_address: No street match; trying fuzzy match on building_name='{building_name}'")

            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            candidates = self.execute_query(query)

            for addr_id, existing_name in candidates or []:
                if existing_name and ratio(building_name, existing_name) >= 85:
                    logging.info(f"Fuzzy match on building_name → '{existing_name}' (score ≥ 85) → address_id={addr_id}")
                    return addr_id

        # Step 3: Insert the new address
        insert_query = """
            INSERT INTO address (
                building_name, street_number, street_name, city,
                province_or_state, postal_code, country, full_address
            ) VALUES (
                :building_name, :street_number, :street_name, :city,
                :province_or_state, :postal_code, :country, :full_address
            )
            RETURNING address_id;
        """

        result = self.execute_query(insert_query, parsed_address)
        if result:
            address_id = result[0][0]
            logging.info(f"Inserted new address with address_id: {address_id}")
            return address_id
        else:
            logging.error("resolve_or_insert_address: Failed to insert new address")
            return None
    

    def get_full_address_from_id(self, address_id: int) -> Optional[str]:
        """
        Returns the full_address from the address table for the given address_id.
        """
        query = "SELECT full_address FROM address WHERE address_id = :address_id"
        result = self.execute_query(query, {"address_id": address_id})
        return result[0][0] if result else None


    def format_address_from_db_row(self, db_row):
        """
        Constructs a formatted address string from a database row.

        This method takes a database row object containing address components and constructs
        a single formatted address string. The formatted address includes the street address,
        city (municipality), province abbreviation, postal code, and country ("CA").
        Missing components are omitted gracefully.

        Args:
            db_row: An object representing a database row with address fields. Expected attributes are:
                - civic_no
                - civic_no_suffix
                - official_street_name
                - official_street_type
                - official_street_dir
                - mail_mun_name (city/municipality)
                - mail_prov_abvn (province abbreviation)
                - mail_postal_code

        Returns:
            str: A formatted address string in the form:
                "<street address>, <city>, <province abbreviation>, <postal code>, CA"
            Any missing components are omitted from the output.
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
    

    def write_events_to_db(self, df, url, parent_url, source, keywords):
        """
        Processes and writes event data to the 'events' table in the database.
        This method performs several data cleaning and transformation steps on the input DataFrame,
        including renaming columns, handling missing values, formatting dates and times, and removing
        outdated or incomplete events. It also logs relevant information and writes a record of the
        processed URL to a separate table.
            df (pandas.DataFrame): DataFrame containing raw event data to be processed and stored.
            url (str): The URL from which the events data was sourced.
            parent_url (str): The parent URL, if applicable, for hierarchical event sources.
            source (str): The source identifier for the event data. If empty, it will be inferred from the URL.
            keywords (str or list): Keywords or dance styles associated with the events. If a list, it will be joined into a string.
            - The method automatically renames columns to match the database schema if the data is from a Google Calendar source.
            - Missing or empty 'source' and 'url' fields are filled with appropriate values.
            - Dates and times are coerced into standard formats; warnings during parsing are suppressed.
            - The 'price' column is ensured to exist and is treated as text.
            - A 'time_stamp' column is added to record the time of data insertion.
            - The method cleans up the 'location' field and updates address IDs via a helper method.
            - Rows with all important fields missing are dropped.
            - Events older than a configurable number of days are excluded.
            - If no valid events remain after cleaning, the method logs this and records the URL as not relevant.
            - Cleaned data is saved to a CSV file for debugging and then written to the 'events' table in the database.
            - The method logs key actions and outcomes for traceability.
        Returns:
            None
        """
        url = '' if pd.isna(url) else str(url)
        parent_url = '' if pd.isna(parent_url) else str(parent_url)

        if 'calendar' in url or 'calendar' in parent_url:
            df = self._rename_google_calendar_columns(df)
            df['dance_style'] = ', '.join(keywords) if isinstance(keywords, list) else keywords

        source = source if source else url.split('.')[-2]
        df['source'] = df.get('source', pd.Series([''] * len(df))).replace('', source).fillna(source)
        df['url'] = df.get('url', pd.Series([''] * len(df))).replace('', url).fillna(url)

        self._convert_datetime_fields(df)

        if 'price' not in df.columns:
            logging.warning("write_events_to_db: 'price' column is missing. Filling with empty string.")
            df['price'] = ''

        df['time_stamp'] = datetime.now()

        # Basic location cleanup
        df = self.clean_up_address_basic(df)

        # Resolve structured addresses using LLM + match/insert logic
        updated_rows = []
        for i, row in df.iterrows():
            event_dict = row.to_dict()
            event_dict = self.normalize_nulls(event_dict)
            updated_event = self.process_event_address(event_dict)
            for key in ["address_id", "location"]:
                if key in updated_event:
                    df.at[i, key] = updated_event[key]
            updated_rows.append(updated_event)

        logging.info(f"write_events_to_db: Address processing complete for {len(updated_rows)} events.")

        # Remove old or incomplete events
        df = self._filter_events(df)

        if df.empty:
            logging.info("write_events_to_db: No events remain after filtering, skipping write.")
            self.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now()])
            return

        df.to_csv('output/cleaned_events.csv', index=False)
        logging.info(f"write_events_to_db: Number of events to write: {len(df)}")

        df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        self.write_url_to_db([url, parent_url, source, keywords, True, 1, datetime.now()])
        logging.info("write_events_to_db: Events data written to the 'events' table.")


    def _rename_google_calendar_columns(self, df):
        """
        Renames columns of a DataFrame containing Google Calendar event data to standardized column names.
        Parameters:
            df (pandas.DataFrame): The input DataFrame with original Google Calendar column names.
        Returns:
            pandas.DataFrame: A DataFrame with columns renamed to standardized names:
                - 'URL' -> 'url'
                - 'Type_of_Event' -> 'event_type'
                - 'Name_of_the_Event' -> 'event_name'
                - 'Day_of_Week' -> 'day_of_week'
                - 'Start_Date' -> 'start_date'
                - 'End_Date' -> 'end_date'
                - 'Start_Time' -> 'start_time'
                - 'End_Time' -> 'end_time'
                - 'Price' -> 'price'
                - 'Location' -> 'location'
                - 'Description' -> 'description'
        """
        return df.rename(columns={
            'URL': 'url', 'Type_of_Event': 'event_type', 'Name_of_the_Event': 'event_name',
            'Day_of_Week': 'day_of_week', 'Start_Date': 'start_date', 'End_Date': 'end_date',
            'Start_Time': 'start_time', 'End_Time': 'end_time', 'Price': 'price',
            'Location': 'location', 'Description': 'description'
        })

    def _convert_datetime_fields(self, df):
        """
        Converts specific datetime-related columns in a pandas DataFrame to appropriate date and time types.
        This method processes the following columns:
            - 'start_date' and 'end_date': Converts to `datetime.date` objects.
            - 'start_time' and 'end_time': Converts to `datetime.time` objects.
        Any parsing errors are coerced to NaT/NaN. UserWarnings during conversion are suppressed.
        Args:
            df (pandas.DataFrame): The DataFrame containing the columns to convert.
        Returns:
            None: The DataFrame is modified in place.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for col in ['start_date', 'end_date']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            for col in ['start_time', 'end_time']:
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
            warnings.resetwarnings()

    def _filter_events(self, df):
        """
        Filters a DataFrame of events by removing rows with all important columns empty and excluding old events.
        This method performs the following steps:
        1. Replaces empty or whitespace-only strings in important columns with pandas NA values.
        2. Drops rows where all important columns ('start_date', 'end_date', 'start_time', 'end_time', 'location', 'description') are missing.
        3. Converts the 'end_date' column to datetime, coercing errors to NaT.
        4. Removes events whose 'end_date' is older than the cutoff date, defined as the current time minus the number of days specified in the configuration under 'clean_up' -> 'old_events'.
        Args:
            df (pd.DataFrame): DataFrame containing event data.
        Returns:
            pd.DataFrame: Filtered DataFrame with only relevant and recent events.
        """
        important_cols = ['start_date', 'end_date', 'start_time', 'end_time', 'location', 'description']
        df[important_cols] = df[important_cols].replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(subset=important_cols, how='all')

        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.config['clean_up']['old_events'])
        return df[df['end_date'] >= cutoff].reset_index(drop=True)
    
    
    def update_event(self, event_identifier, new_data, best_url):
        """
        Update an existing event in the database by overlaying new data and setting the best URL.
        This method locates an event row in the 'events' table using the provided event_identifier criteria.
        It overlays the values from new_data onto the existing row, replacing only the fields present and non-empty in new_data.
        The event's URL is updated to the provided best_url. The update is performed in-place; if no matching event is found,
        the method logs an error and returns False.
            event_identifier (dict): Dictionary specifying the criteria to uniquely identify the event row.
                Example: {'event_name': ..., 'start_date': ..., 'start_time': ...}
            new_data (dict): Dictionary containing new values to update in the event record. Only non-empty and non-null
                values will overwrite existing fields.
            best_url (str): The URL to set as the event's 'url' field.
        Returns:
            bool: True if the event was found and updated successfully, False otherwise.
        Logs:
            - Error if no matching event is found.
            - Info when an event is successfully updated.
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


    def fuzzy_match(self, a: str, b: str, threshold: int = 85) -> bool:
        """
        Returns True if the fuzzy match score between two strings exceeds the threshold.
        Uses token sort ratio for better match on rearranged terms.
        """
        score = fuzz.token_sort_ratio(a, b)
        return score >= threshold


    def insert_address_and_return_id(self, address_dict: dict) -> int:
        """
        Inserts a new address record into the address table and returns the address_id.
        """
        address_dict = self.normalize_nulls(address_dict)
        address_dict["time_stamp"] = datetime.now().isoformat()

        if "full_address" not in address_dict or not address_dict["full_address"]:
            full_address_parts = [
                address_dict.get("building_name"),
                address_dict.get("street_number"),
                address_dict.get("street_name"),
                address_dict.get("street_type"),
                address_dict.get("direction"),
                address_dict.get("city"),
                address_dict.get("province_or_state"),
                address_dict.get("postal_code"),
                "Canada"
            ]
            address_dict["full_address"] = " ".join([str(part) for part in full_address_parts if part])

        if "country_id" not in address_dict:
            address_dict["country_id"] = 1  # Default: Canada

        query = """
            INSERT INTO address (
                full_address, building_name, street_number, street_name, street_type,
                direction, city, met_area, province_or_state, postal_code, country_id, time_stamp
            ) VALUES (
                :full_address, :building_name, :street_number, :street_name,
                :street_type, :direction, :city, :met_area, :province_or_state,
                :postal_code, :country_id, :time_stamp
            ) RETURNING address_id;
        """

        result = self.execute_query(query, address_dict)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0][0]
        else:
            logging.error("insert_address_and_return_id(): Failed to insert address.")
            return 0
        

    def match_civic_number(self, df, numbers):
        """
        Attempts to match the first numeric string from a list to the 'civic_no' column in a DataFrame of addresses.
        Parameters:
            df (pd.DataFrame): DataFrame containing address information, including a 'civic_no' column.
            numbers (list of str): List of numeric strings extracted from a location.
        Returns:
            int or None: The index of the row in the DataFrame where the first number matches the 'civic_no'.
                         If no match is found, returns the index of the first row.
                         Returns None if the DataFrame is empty.
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


    def process_event_address(self, event: dict) -> dict:
        """
        Uses the LLM to parse a structured address from the location, inserts or reuses the address in the DB,
        and updates the event with address_id and location = full_address from address table.
        """
        if location is None:
            pass  # Keep it as None
        elif isinstance(location, str):
            location = location.strip()

        if not location or len(location) < 15:
            event["address_id"] = 0
            return event

        # Step 1: Generate the LLM prompt
        prompt = self.llm_handler.generate_prompt("address_fix", location, "address_internet_fix")

        # Step 2: Query the LLM
        llm_response = self.llm_handler.query_llm(event.get("url", "").strip(), prompt)

        # Step 3: Parse the LLM response into a usable dict
        parsed_results = self.llm_handler.extract_and_parse_json(llm_response, "address_fix")
        if not parsed_results or not isinstance(parsed_results, list) or not isinstance(parsed_results[0], dict):
            logging.warning("process_event_address: Could not parse address from LLM response")
            event["address_id"] = 0
            return event

        # ✅ Normalize null-like strings in one place
        parsed_address = self.normalize_nulls(parsed_results[0])

        # Step 4: Get or insert address_id
        address_id = self.resolve_or_insert_address(parsed_address)

        # Step 5: Force consistency: always use address.full_address
        full_address = self.get_full_address_from_id(address_id)
        event["address_id"] = address_id
        if full_address:
            event["location"] = full_address

        return event
    

    def sync_event_locations_with_address_table(self):
        """
        Updates all events so that location = full_address from the address table for consistency.
        """
        query = """
            UPDATE events e
            SET location = a.full_address
            FROM address a
            WHERE e.address_id = a.address_id
            AND (e.location IS DISTINCT FROM a.full_address);
        """
        affected_rows = self.execute_query(query)
        logging.info(f"sync_event_locations_with_address_table(): Updated {affected_rows} events to use canonical full_address.")


    def dedup(self):
        """
        Removes duplicate entries from the 'events' table in the database.

        Duplicates in the 'events' table are identified based on matching 'address_id', 'start_date', 'end_date',
        and start/end times within 15 minutes (900 seconds) of each other. Only the latest entry (with the highest event_id)
        is retained for each group of duplicates; all others are deleted.

        Returns:
            int: The number of rows deleted from the 'events' table during deduplication.

        Raises:
            Exception: If an error occurs during the deduplication process, it is logged and re-raised.
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
        Fetch all events from the database and return them as a pandas DataFrame sorted by start date and time.

        Returns:
            pandas.DataFrame: A DataFrame containing all events from the 'events' table,
            sorted by 'start_date' and 'start_time' columns.
        """
        query = "SELECT * FROM events"
        df = pd.read_sql(query, self.conn)
        df.sort_values(by=['start_date', 'start_time'], inplace=True)
        return df

    def decide_preferred_row(self, row1, row2):
        """
        Determines the preferred row between two given rows based on the following criteria:
            1. Prefer the row with a non-empty 'url' field.
            2. If both or neither have a 'url', prefer the row with more filled (non-empty) columns, excluding 'event_id'.
            3. If still tied, prefer the row with the most recent 'time_stamp'.
        Args:
            row1 (pandas.Series): The first row to compare.
            row2 (pandas.Series): The second row to compare.
            tuple:
                preferred_row (pandas.Series): The row selected as preferred based on the criteria.
                other_row (pandas.Series): The other row that was not preferred.
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
        Update missing values in the preferred row with corresponding values from the other row for specified columns.

        For each column in `columns`, if the value in `preferred` is missing (NaN or empty string), 
        and the value in `other` is present (not NaN and not empty string), the value from `other` 
        is copied to `preferred`.

        Args:
            preferred (pd.Series): The row to be updated, typically the preferred or primary row.
            other (pd.Series): The row to use as a source for missing values.
            columns (Iterable[str]): List or iterable of column names to check and update.

        Returns:
            pd.Series: The updated preferred row with missing values filled from the other row where applicable.
        """
        for col in columns:
            if pd.isna(preferred[col]) or preferred[col] == '':
                if not (pd.isna(other[col]) or other[col] == ''):
                    preferred[col] = other[col]
        return preferred


    def fuzzy_duplicates(self):
        """
        Identifies and removes fuzzy duplicate events from the events table in the database.
        This method performs the following steps:
            1. Fetches all events and sorts them by 'start_date' and 'start_time'.
            2. Groups events that share the same 'start_date' and 'start_time'.
            3. Within each group, compares event names using fuzzy string matching.
            4. If two events have a fuzzy match score greater than 80:
                a. Determines which event to keep based on the following criteria:
                    i. Prefer the event with a URL.
                    ii. If neither has a URL, prefer the event with more filled columns.
                    iii. If still tied, prefer the most recently updated event.
                b. Updates the kept event with any missing data from the duplicate.
                c. Updates the database with the merged event and deletes the duplicate.
            5. Logs actions taken during the process.
        Returns:
            None. The method updates the database in place and logs the actions performed.
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
        The number of days is retrieved from the configuration under 'clean_up' -> 'old_events'.
        Events with an 'End_Date' earlier than the current date minus the specified number of days are deleted.

        Returns:
            int: The number of events deleted from the database.

        Raises:
            Exception: If an error occurs during the deletion process, it is logged and re-raised.
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
        Deletes likely invalid or irrelevant events from the database based on several criteria:
        1. Deletes events where 'source', 'dance_style', and 'url' are empty strings, unless the event has an associated 'address_id'.
        2. Deletes events whose associated address (if present) has a 'province_or_state' that is not 'BC' (British Columbia).
        3. Deletes events whose associated address (if present) has a 'country_id' that is not 'CA' (Canada).
        4. Deletes events where 'dance_style' and 'url' are empty strings, 'event_type' is 'other', and both 'location' and 'description' are NULL.
        For each deletion step, logs the number of events deleted.
        Returns:
            None
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
        Deletes an event from the 'events' table based on the provided event name and start date.

            url (str): The URL of the event to be deleted. (Note: This parameter is currently unused in the deletion query.)

        Returns:
            None

        Raises:
            Exception: If an error occurs during the deletion process, it is logged and the exception is propagated.
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
        Deletes events from the 'events' table where both 'start_date' and 'start_time' are NULL,
        or both 'start_time' and 'end_time' are NULL.

        Returns:
            int: The number of events deleted from the table.

        Raises:
            Exception: If an error occurs during the deletion process.
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
        Deletes an event from the 'events' table based on the provided event_id.

        Args:
            event_id (int): The unique identifier of the event to be deleted.

        Returns:
            None

        Raises:
            Exception: If the deletion fails, an exception is logged and propagated.
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
        Deletes multiple events from the 'events' table based on a list of event IDs.
            event_ids (list): A list of event IDs (int) to be deleted from the database.
            bool: 
                - True if all specified events were successfully deleted.
                - False if one or more deletions failed, or if the input list is empty.
        Logs:
            - A warning if no event IDs are provided.
            - An error for each event ID that fails to be deleted.
            - An info message summarizing the number of successful deletions.
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
        Inserts or updates multiple records in the specified table using an upsert strategy.

            table_name (str): The name of the table to insert or update. Supported values are "address" and "events".
            values (list of dict): A list of dictionaries, each representing a row to insert or update. Each dictionary's keys should match the table's column names.

        Returns:
            None

        Raises:
            ValueError: If the specified table_name is not supported.
            Exception: If an error occurs during the insert or update operation.

        Logs:
            - An info message if no values are provided.
            - An info message upon successful insertion or update.
            - An error message if an exception occurs during the operation.
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
        Executes a SQL query to aggregate and count the number of events per source in the events table.

        Returns:
            pandas.DataFrame: A DataFrame containing two columns:
                - 'source': The source of each event.
                - 'counted': The number of events associated with each source, sorted in descending order.
        """
        query = "SELECT source, COUNT(*) AS counted FROM events GROUP BY source ORDER BY counted DESC"
        groupby_df = pd.read_sql_query(query, self.conn)
        logging.info(f"def groupby_source(): Retrieved groupby results from events table.")
        return groupby_df


    def count_events_urls_start(self, file_name):
        """
        Counts the number of events and distinct URLs in the database at the start time and returns a DataFrame with the results.

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
        sql = "SELECT COUNT(DISTINCT link) as urls_count_start FROM urls"
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
        sql = "SELECT COUNT(DISTINCT link) as urls_count_end FROM urls"
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


    def stale_date(self, url):
        """
        Determines whether the most recent event associated with the given URL is considered "stale" based on a configurable age threshold.

        Args:
            url (str): The URL whose most recent event's staleness is to be checked.

        Returns:
            bool: 
                - True if there are no events for the URL, the most recent event's start date is older than the configured threshold, 
                  or if an error occurs during the check (defaulting to stale).
                - False if the most recent event's start date is within the allowed threshold (i.e., not stale).

        Raises:
            None: All exceptions are caught internally and logged; the method will return True in case of errors.
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
        Determines whether a given URL should be processed based on its history in the database.

        The decision is made according to the following rules:
        1. If the URL has never been seen before (i.e., no records in `self.urls_df`), it should be processed.
        2. If the most recent record for the URL has 'relevant' set to True, it should be processed.
        3. If the most recent record for the URL has 'relevant' set to False, the method checks the grouped statistics in `self.urls_gb`:
            - If the `hit_ratio` for the URL is greater than 0.1, or
            - If the number of crawl attempts (`crawl_try`) is less than or equal to 3,
            then the URL should be processed.
        4. If none of the above conditions are met, the URL should not be processed.

        Args:
             url (str): The URL to evaluate for processing.

        Returns:
             bool: True if the URL should be processed according to the criteria above, False otherwise.
        """
        # 1. Filter all rows for this URL
        df_url = self.urls_df[self.urls_df['link'] == url]
        # If we've never recorded this URL, process it
        if df_url.empty:
            logging.info(f"should_process_url: URL {url} has never been seen before, processing it.")
            return True

        # 2. Look at the most recent "relevant" value
        last_relevant = df_url.iloc[-1]['relevant']
        if last_relevant and self.stale_date(url):
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


    def update_dow_date(self, event_id: int, corrected_date) -> bool:
        """
        Updates the start_date and end_date fields of the event with the specified event_id to the given corrected_date.

        Args:
            event_id (int): The unique identifier of the event to update.
            corrected_date: The new date to set for both start_date and end_date. The expected type should match the database schema (e.g., str or datetime).

        Returns:
            bool: True if the update operation was executed (does not guarantee that a row was actually updated).
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
        Ensures that the start_date of each event in the database matches its specified day_of_week.

        This method performs the following steps:
            1. Retrieves all events' event_id, start_date, and day_of_week from the database.
            2. For each event, checks if the start_date's weekday matches the stored day_of_week.
            3. If there is a mismatch, computes the minimal shift (within ±3 days) required to align the start_date
               with the correct weekday.
            4. Calls update_dow_date(...) to update both start_date and end_date when a shift is needed.
            5. Logs every adjustment, including warnings for unrecognized day_of_week values and errors if updates fail.

        Returns:
            None
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


    def check_image_events_exist(self, image_url: str) -> bool:
        """
        Checks if there are any events associated with the specified image URL.

        This method performs the following steps:
        1. Checks the `events` table for any events with the given image URL.
        2. If no events are found, checks the `events_history` table for matching events.
        3. If events are found in the history, copies only the most recent version of each unique event (grouped by all fields except `time_stamp`) from `events_history` into the `events` table.

        Args:
            image_url (str): The URL of the image to check for associated events.

        Returns:
            bool: True if events exist for the given image URL (either already present or copied from history), False otherwise.
        """
        # 1) Check live events table
        sql_live = """
        SELECT COUNT(*)
        FROM events
        WHERE url = :url
        """
        params = {'url': image_url}
        logging.info(f"check_image_events_exist(): image_url is: {image_url}")
        live = self.execute_query(sql_live, params)
        if live and live[0][0] > 0:
            logging.info(f"check_image_events_exist(): live is: {live[0][0]}")
            logging.info(f"check_image_events_exist(): Events already exist for URL: {image_url}")
            return True

        # 2) Check history table
        sql_hist = """
        SELECT COUNT(*)
        FROM events_history
        WHERE url = :url
        """
        hist = self.execute_query(sql_hist, params)
        if not (hist and hist[0][0] > 0):
            logging.info(f"check_image_events_exist(): No history events for URL: {image_url}")
            return False

        # 3) Copy only the most‐recent history row per unique event into events
        sql_copy = """
        INSERT INTO events (
            event_name, dance_style, description, day_of_week,
            start_date, end_date, start_time, end_time,
            source, location, price, url,
            event_type, address_id, time_stamp
        )
        SELECT
            sub.event_name, sub.dance_style, sub.description, sub.day_of_week,
            sub.start_date, sub.end_date, sub.start_time, sub.end_time,
            sub.source, sub.location, sub.price, sub.url,
            sub.event_type, sub.address_id, sub.time_stamp
        FROM (
            SELECT DISTINCT ON (
                event_name, dance_style, description, day_of_week,
                start_date, end_date, start_time, end_time,
                source, location, price, url,
                event_type, address_id
            )
                event_name, dance_style, description, day_of_week,
                start_date, end_date, start_time, end_time,
                source, location, price, url,
                event_type, address_id, time_stamp
            FROM events_history
            WHERE url = :url
            AND start_date >= (CURRENT_DATE - (:days * INTERVAL '1 day'))
            ORDER BY
                event_name, dance_style, description, day_of_week,
                start_date, end_date, start_time, end_time,
                source, location, price, url,
                event_type, address_id,
                time_stamp DESC
        ) AS sub
        """
        params_copy = {
            'url':  image_url,
            'days': self.config['clean_up']['old_events']  # e.g. 3 → includes start_date ≥ today−3d
        }
        self.execute_query(sql_copy, params_copy)

        logging.info(f"check_image_events_exist(): Copied most‐recent history events into events for URL: {image_url}")
        return True
    

    def sql_input(self, file_path: str):
        """
        Reads a JSON file containing a flat dictionary of SQL fixes and executes them.

        Args:
            file_path (str): Path to the .json file containing SQL statements.
        """
        logging.info("sql_input(): Starting SQL input execution from %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sql_dict = json.load(f)
            logging.info("sql_input(): Successfully loaded %d SQL entries", len(sql_dict))
        except Exception as e:
            logging.error("sql_input(): Failed to load or parse JSON file: %s", e)
            return

        for name, query in sql_dict.items():
            logging.info("sql_input(): Executing [%s]: %s", name, query)
            result = self.execute_query(query)
            if result is None:
                logging.error("sql_input(): Failed to execute [%s]", name)
            else:
                logging.info("sql_input(): Successfully executed [%s]", name)

        logging.info("sql_input(): All queries processed.")

    
    def normalize_nulls(self, record: dict) -> dict:
        """
        Replaces string values like 'null', 'none', 'nan', or empty strings with Python None (i.e., SQL NULL).
        Applies to all keys in the given dictionary.
        """
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str) and value.strip().lower() in {"null", "none", "nan", ""}:
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    

    def clean_null_strings_in_address(self):
        """
        Replaces string 'null', 'none', 'nan', and '' with actual SQL NULLs in address table.
        """
        fields = [
            "full_address", "building_name", "street_number", "street_name", "direction"
            "city", "met_area", "province_or_state", "country", "postal_code", "country_id"
        ]
        for field in fields:
            query = f"""
                UPDATE address
                SET {field} = NULL
                WHERE TRIM(LOWER({field})) IN ('null', 'none', 'nan', '');
            """
            self.execute_query(query)
        logging.info("Cleaned up string 'null's in address table.")


    def driver(self):
        """
        Main driver function to perform database operations based on configuration.

        If the 'drop_tables' flag in the testing configuration is set to True,
        the function will recreate the database tables. Otherwise, it will perform
        a series of data cleaning and maintenance operations, including deduplication,
        deletion of old or invalid events, fuzzy duplicate detection, foreign key checks,
        address deduplication, and fixing address IDs in events.

        At the end of the process, the database connection is properly closed.

        Returns:
            None
        """
        if self.config['testing']['drop_tables'] == True:
            self.create_tables()
        else:
            self.check_dow_date_consistent()
            self.delete_old_events()
            self.delete_events_with_nulls()
            self.delete_likely_dud_events()
            self.fuzzy_duplicates()
            self.is_foreign()
            self.sql_input(self.config['input']['sql_input'])
            self.sync_event_locations_with_address_table()
            self.clean_null_strings_in_address()
            self.dedup()

        # Close the database connection
        self.conn.dispose()  # Using dispose() for SQLAlchemy Engine
        logging.info("driver(): Database operations completed successfully.")
        

if __name__ == "__main__":
    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

     # Build log_file name
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logging_file = f"logs/{script_name}_log.txt" 
    logging.basicConfig(
        filename=logging_file,
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
