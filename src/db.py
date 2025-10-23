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
from typing import Optional, List, Dict, Any
import sys
import yaml
import warnings

# Import database configuration utility
from db_config import get_database_config

# Import new utilities
from utils.fuzzy_utils import FuzzyMatcher
from config_manager import ConfigManager


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

        # Get database configuration using centralized utility
        # This automatically handles local, render_dev, and render_prod environments
        connection_string, env_name = get_database_config()
        self.conn = create_engine(connection_string, isolation_level="AUTOCOMMIT")
        logging.info(f"def __init__(): Database connection established: {env_name}")

        # Note: The 'locations' table (Canadian postal code database) is now part of social_dance_db
        # Previously this was in a separate address_db, but has been consolidated for simplicity

        if self.conn is None:
                raise ConnectionError("def __init__(): DatabaseHandler: Failed to establish a database connection.")

        self.metadata = MetaData()
        # Reflect the existing database schema into metadata
        self.metadata.reflect(bind=self.conn)

        # Get google api key
        self.google_api_key = os.getenv("GOOGLE_KEY_PW")

        # Create df from urls table (only if not on production - production doesn't have urls table)
        from db_config import is_production_target
        if not is_production_target():
            self.urls_df = self.create_urls_df()
            logging.info("__init__(): URLs DataFrame created with %d rows.", len(self.urls_df))
        else:
            self.urls_df = pd.DataFrame()
            logging.info("__init__(): Skipping URLs table load on production (not needed for web service)")

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
        if not is_production_target():
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

            # Create raw_locations table for caching location strings (only needed for pipeline)
            self.create_raw_locations_table()
        else:
            self.urls_gb = pd.DataFrame()
            logging.info("__init__(): Skipping urls_gb and raw_locations table creation on production")


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

        DEPRECATED: This method now uses get_database_config() internally.
        Kept for backward compatibility with existing code.

        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine instance if connection is successful.
            None: If the connection could not be established.

        Note:
            New code should use get_database_config() from db_config module directly.
            This method is maintained for backward compatibility with:
            - ebs.py, clean_up.py, fb.py, scraper.py, irrelevant_rows.py
        """
        try:
            # Use centralized database configuration
            connection_string, env_name = get_database_config()
            logging.info(f"get_db_connection(): Connecting to {env_name}")

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
                    # Extract query type and key info for logging
                    query_type = query.strip().split()[0].upper()
                    if query_type == "SELECT":
                        # Extract table name from SELECT query
                        table_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): SELECT from %s returned %d rows", 
                            table_name, len(rows)
                        )
                    else:
                        logging.info(
                            "execute_query(): %s query returned %d rows", 
                            query_type, len(rows)
                        )
                    return rows
                else:
                    affected = result.rowcount
                    connection.commit()
                    # Extract query type for non-select queries
                    query_type = query.strip().split()[0].upper()
                    if query_type == "INSERT":
                        # Extract table name from INSERT query
                        table_match = re.search(r'INSERT\s+(?:INTO\s+)?(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): INSERT into %s affected %d rows", 
                            table_name, affected
                        )
                    elif query_type == "UPDATE":
                        # Extract table name from UPDATE query
                        table_match = re.search(r'UPDATE\s+(\w+)', query, re.IGNORECASE)
                        table_name = table_match.group(1) if table_match else "unknown"
                        logging.info(
                            "execute_query(): UPDATE %s affected %d rows", 
                            table_name, affected
                        )
                    else:
                        logging.info(
                            "execute_query(): %s query affected %d rows", 
                            query_type, affected
                        )
                    return affected

        except SQLAlchemyError as e:
            # Handle unique constraint violations for address table gracefully
            error_str = str(e)
            if "UniqueViolation" in error_str and ("unique_full_address" in error_str or "address_full_address_key" in error_str):
                logging.info(
                    "execute_query(): Address already exists (unique constraint), skipping insert"
                )
                return None
            else:
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
        df = pd.read_sql(query, self.conn, params=(postal_code,))

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
        Resolves an address by checking multiple matching strategies in order of specificity.
        Uses improved fuzzy matching to prevent duplicate addresses.

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
        postal_code = (parsed_address.get("postal_code") or "").strip()
        city = (parsed_address.get("city") or "").strip()
        country_id = (parsed_address.get("country_id") or "").strip()

        # Step 1: Exact match on postal code + street number (most specific)
        if postal_code and street_number:
            logging.debug(f"resolve_or_insert_address: Trying postal_code + street_number match: {postal_code}, {street_number}")
            postal_match_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(postal_code) = LOWER(:postal_code)
                AND LOWER(street_number) = LOWER(:street_number)
            """
            postal_matches = self.execute_query(postal_match_query, {
                "postal_code": postal_code,
                "street_number": street_number
            })

            for addr_id, b_name, s_num, s_name, p_code in postal_matches or []:
                if building_name and b_name:
                    # Use multiple fuzzy matching algorithms
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    # More sophisticated matching: any high score indicates a match
                    if ratio_score >= 85 or partial_score >= 95 or token_set_score >= 90:
                        logging.debug(f"Postal+street+fuzzy building match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id
                else:
                    # Same postal code + street number is very likely the same location
                    logging.debug(f"Postal+street match (no building comparison) → address_id={addr_id}")
                    return addr_id

        # Step 2: Street number + street name match with improved building name fuzzy matching
        if street_number and street_name:
            select_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(street_number) = LOWER(:street_number)
                AND (LOWER(street_name) = LOWER(:street_name) OR LOWER(street_name) = LOWER(:street_name_alt))
            """
            # Handle common street name variations (Niagra vs Niagara)
            street_name_alt = street_name.replace('Niagra', 'Niagara').replace('Niagara', 'Niagra')
            
            street_matches = self.execute_query(select_query, {
                "street_number": street_number,
                "street_name": street_name,
                "street_name_alt": street_name_alt
            })

            for addr_id, b_name, s_num, s_name, p_code in street_matches or []:
                if building_name and b_name:
                    # Multiple fuzzy algorithms
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    if ratio_score >= 75 or partial_score >= 90 or token_set_score >= 85:
                        logging.debug(f"Street+fuzzy building match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id
                else:
                    logging.debug(f"Street match (no building name) → address_id={addr_id}")
                    return addr_id
        else:
            logging.debug("resolve_or_insert_address: Missing street_number or street_name; skipping street match")

        # Step 3: City + building name fuzzy match (broader search)
        if city and building_name:
            logging.debug(f"resolve_or_insert_address: Trying city + building_name match: {city}, {building_name}")
            city_building_query = """
                SELECT address_id, building_name, city, postal_code
                FROM address
                WHERE LOWER(city) = LOWER(:city) AND building_name IS NOT NULL
            """
            city_matches = self.execute_query(city_building_query, {"city": city})

            for addr_id, b_name, addr_city, p_code in city_matches or []:
                if b_name:
                    ratio_score = ratio(building_name, b_name)
                    partial_score = fuzz.partial_ratio(building_name, b_name)
                    token_set_score = fuzz.token_set_ratio(building_name, b_name)

                    # Higher thresholds for city-only matches to avoid false positives
                    if ratio_score >= 90 or partial_score >= 95 or token_set_score >= 95:
                        logging.debug(f"City+building fuzzy match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id

        # Step 4: Legacy building name-only fuzzy match (least reliable)
        if building_name:
            logging.debug(f"resolve_or_insert_address: Trying building_name-only fuzzy match: {building_name}")
            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            candidates = self.execute_query(query)

            for addr_id, existing_name in candidates or []:
                if existing_name:
                    ratio_score = ratio(building_name, existing_name)
                    partial_score = fuzz.partial_ratio(building_name, existing_name)
                    token_set_score = fuzz.token_set_ratio(building_name, existing_name)

                    # Very high thresholds for building-name-only matches
                    if ratio_score >= 95 or (partial_score >= 98 and token_set_score >= 95):
                        logging.debug(f"Building-name-only fuzzy match → address_id={addr_id}")
                        logging.debug(f"  Scores: ratio={ratio_score}, partial={partial_score}, token_set={token_set_score}")
                        return addr_id

        # Step 3: Normalize null values and prepare required fields for insert
        parsed_address = self.normalize_nulls(parsed_address)
        
        # Ensure ALL fields expected by INSERT query are present (set to None if missing)
        required_fields = [
            "building_name", "street_number", "street_name", "city",
            "province_or_state", "postal_code", "country_id"
        ]
        
        for field in required_fields:
            if field not in parsed_address:
                parsed_address[field] = None
        
        # Set specific fields from extracted values
        parsed_address["building_name"] = building_name or parsed_address.get("building_name")
        parsed_address["street_number"] = street_number or parsed_address.get("street_number")
        parsed_address["street_name"] = street_name or parsed_address.get("street_name")
        parsed_address["country_id"] = country_id or parsed_address.get("country_id")

        # Build standardized full_address from components
        standardized_full_address = self.build_full_address(
            building_name=parsed_address.get("building_name"),
            street_number=parsed_address.get("street_number"),
            street_name=parsed_address.get("street_name"),
            street_type=parsed_address.get("street_type"),
            city=parsed_address.get("city"),
            province_or_state=parsed_address.get("province_or_state"),
            postal_code=parsed_address.get("postal_code"),
            country_id=parsed_address.get("country_id")
        )
        parsed_address["full_address"] = standardized_full_address

        # Set time_stamp for the new address
        parsed_address["time_stamp"] = datetime.now().isoformat()

        # FINAL DEDUPLICATION CHECK: Before inserting, check if building_name already exists
        building_name = parsed_address.get("building_name", "").strip()
        if building_name and len(building_name) > 2:
            # Try to find existing address with same building name
            existing_addr_id = self.find_address_by_building_name(building_name, threshold=80)
            if existing_addr_id:
                logging.info(f"resolve_or_insert_address: Found existing address (dedup) with building_name='{building_name}' → address_id={existing_addr_id}")
                return existing_addr_id

        insert_query = """
            INSERT INTO address (
                building_name, street_number, street_name, city,
                province_or_state, postal_code, country_id, full_address, time_stamp
            ) VALUES (
                :building_name, :street_number, :street_name, :city,
                :province_or_state, :postal_code, :country_id, :full_address, :time_stamp
            )
            RETURNING address_id;
        """

        result = self.execute_query(insert_query, parsed_address)
        if result:
            address_id = result[0][0]
            logging.info(f"Inserted new address with address_id: {address_id}")
            return address_id
        else:
            # If insert failed (likely due to unique constraint), try to find existing address
            full_address = parsed_address.get("full_address")
            if full_address:
                lookup_query = "SELECT address_id FROM address WHERE full_address = :full_address"
                lookup_result = self.execute_query(lookup_query, {"full_address": full_address})
                if lookup_result:
                    address_id = lookup_result[0][0]
                    logging.info(f"Found existing address with address_id: {address_id}")
                    return address_id
            
            logging.error("resolve_or_insert_address: Failed to insert or find existing address")
            return None
    

    def build_full_address(self, building_name: str = None, street_number: str = None, 
                          street_name: str = None, street_type: str = None, 
                          city: str = None, province_or_state: str = None, 
                          postal_code: str = None, country_id: str = None) -> str:
        """
        Builds a standardized full_address string from address components.
        
        Format: "building_name, street_number street_name street_type, city, province_or_state postal_code, country_id"
        
        Args:
            building_name: Building or venue name (optional)
            street_number: Street number
            street_name: Street name  
            street_type: Street type (St, Ave, Rd, etc.)
            city: City name
            province_or_state: Province or state
            postal_code: Postal code
            country_id: Country code
            
        Returns:
            str: Formatted full address
        """
        address_parts = []
        
        # Add building name first if present
        if building_name and building_name.strip():
            address_parts.append(building_name.strip())
        
        # Build street address
        street_parts = []
        if street_number and street_number.strip():
            street_parts.append(street_number.strip())
        if street_name and street_name.strip():
            street_parts.append(street_name.strip())
        if street_type and street_type.strip():
            street_parts.append(street_type.strip())
        
        if street_parts:
            address_parts.append(' '.join(street_parts))
        
        # Add city
        if city and city.strip():
            address_parts.append(city.strip())
        
        # Add province/state and postal code
        if province_or_state and province_or_state.strip():
            if postal_code and postal_code.strip():
                address_parts.append(f"{province_or_state.strip()} {postal_code.strip()}")
            else:
                address_parts.append(province_or_state.strip())
        elif postal_code and postal_code.strip():
            address_parts.append(postal_code.strip())
        
        # Add country
        if country_id and country_id.strip():
            address_parts.append(country_id.strip())
        
        return ', '.join(address_parts)

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

        source = source if source else (url.split('.')[-2] if url and '.' in url and len(url.split('.')) >= 2 else 'unknown')
        df['source'] = df.get('source', pd.Series([''] * len(df))).replace('', source).fillna(source)
        df['url'] = df.get('url', pd.Series([''] * len(df))).replace('', url).fillna(url)

        self._convert_datetime_fields(df)

        if 'price' not in df.columns:
            logging.warning("write_events_to_db: 'price' column is missing. Filling with empty string.")
            df['price'] = ''

        df['time_stamp'] = datetime.now()

        # Clean day_of_week field to handle compound/invalid values
        df = self._clean_day_of_week_field(df)

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

        # Write debug CSV (only locally, not on Render)
        if os.getenv('RENDER') != 'true':
            os.makedirs('output', exist_ok=True)
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

    def _clean_day_of_week_field(self, df):
        """
        Cleans and standardizes the day_of_week field to handle compound/invalid values.
        
        This method fixes common issues with day_of_week values:
        - Compound values like "Friday, Saturday" -> takes first day ("Friday")
        - Special values like "Daily" -> converts to empty string
        - Normalizes case and whitespace
        
        Args:
            df (pd.DataFrame): DataFrame containing event data with day_of_week column
            
        Returns:
            pd.DataFrame: DataFrame with cleaned day_of_week values
        """
        if 'day_of_week' not in df.columns:
            return df
            
        original_count = len(df)
        logging.info(f"_clean_day_of_week_field: Processing {original_count} events")
        
        # Track changes for logging
        changes_made = 0
        
        for i, row in df.iterrows():
            original_value = row.get('day_of_week', '')
            if pd.isna(original_value) or not str(original_value).strip():
                continue
                
            day_str = str(original_value).strip()
            cleaned_value = original_value
            
            # Handle compound values like "Friday, Saturday" - take first day
            if ',' in day_str:
                cleaned_value = day_str.split(',')[0].strip()
                changes_made += 1
                logging.info(f"_clean_day_of_week_field: Changed compound day '{original_value}' to '{cleaned_value}' for event at index {i}")
                
            # Handle special values like "Daily"
            elif day_str.lower() in ['daily', 'every day', 'everyday']:
                cleaned_value = ''  # Set to empty, will be handled by validation later
                changes_made += 1
                logging.info(f"_clean_day_of_week_field: Changed special day '{original_value}' to empty for event at index {i}")
                
            # Normalize standard day names (capitalize first letter)
            else:
                valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                day_lower = day_str.lower()
                if day_lower in valid_days:
                    cleaned_value = day_lower.capitalize()
                    if cleaned_value != original_value:
                        changes_made += 1
                
            # Update the DataFrame if value changed
            if cleaned_value != original_value:
                df.at[i, 'day_of_week'] = cleaned_value
        
        logging.info(f"_clean_day_of_week_field: Made {changes_made} changes to day_of_week values")
        return df

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
        logging.info(f"_filter_events: Input DataFrame has {len(df)} events")
        
        important_cols = ['start_date', 'end_date', 'start_time', 'end_time', 'location', 'description']
        
        # Log missing columns
        missing_cols = [col for col in important_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"_filter_events: Missing important columns: {missing_cols}")
        
        df[important_cols] = df[important_cols].replace(r'^\s*$', pd.NA, regex=True)
        rows_before_dropna = len(df)
        df = df.dropna(subset=important_cols, how='all')
        rows_after_dropna = len(df)
        
        if rows_before_dropna != rows_after_dropna:
            logging.info(f"_filter_events: Dropped {rows_before_dropna - rows_after_dropna} rows with all important columns empty")

        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.config['clean_up']['old_events'])
        rows_before_date_filter = len(df)
        filtered_df = df[df['end_date'] >= cutoff].reset_index(drop=True)
        rows_after_date_filter = len(filtered_df)
        
        if rows_before_date_filter != rows_after_date_filter:
            logging.info(f"_filter_events: Dropped {rows_before_date_filter - rows_after_date_filter} events older than {cutoff.date()}")
        
        logging.info(f"_filter_events: Output DataFrame has {len(filtered_df)} events")
        return filtered_df
    
    
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

    def _get_building_name_dictionary(self):
        """
        Creates and caches a dictionary mapping building names to address_ids from the address table.
        
        Returns:
            dict: Dictionary with building_name (lowercase) as keys and address_id as values
        """
        if not hasattr(self, '_building_name_cache'):
            logging.info("_get_building_name_dictionary: Building building name lookup cache")
            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL AND building_name != ''"
            results = self.execute_query(query)
            
            self._building_name_cache = {}
            if results:
                for address_id, building_name in results:
                    if building_name and building_name.strip():
                        # Use lowercase for case-insensitive matching
                        self._building_name_cache[building_name.lower().strip()] = address_id
                        
            logging.info(f"_get_building_name_dictionary: Cached {len(self._building_name_cache)} building names")
        
        return self._building_name_cache

    def _extract_address_from_event_details(self, event: Dict[str, Any]) -> Optional[int]:
        """
        Attempts to extract building names from event_name and description, 
        then matches them against existing addresses in the database.
        
        Args:
            event: Dictionary containing event data
            
        Returns:
            int or None: address_id if a match is found, None otherwise
        """
        building_dict = self._get_building_name_dictionary()
        if not building_dict:
            return None
            
        # Collect text to search from event details
        search_texts = []
        event_name = event.get("event_name", "")
        description = event.get("description", "")
        
        if event_name:
            search_texts.append(str(event_name))
        if description:
            search_texts.append(str(description))
            
        if not search_texts:
            return None
            
        # Search for building names in the text
        combined_text = " ".join(search_texts).lower()
        
        # First try exact matches
        for building_name, address_id in building_dict.items():
            if building_name in combined_text:
                logging.info(f"_extract_address_from_event_details: Found exact match '{building_name}' -> address_id={address_id}")
                return address_id
                
        # Then try fuzzy matching for partial matches
        best_match = None
        best_score = 0
        
        for building_name, address_id in building_dict.items():
            # Skip very short building names for fuzzy matching to avoid false positives
            if len(building_name) < 6:
                continue
                
            score = fuzz.partial_ratio(building_name, combined_text)
            if score > 80 and score > best_score:  # High threshold for fuzzy matching
                best_score = score
                best_match = (building_name, address_id)
                
        if best_match:
            building_name, address_id = best_match
            logging.info(f"_extract_address_from_event_details: Found fuzzy match '{building_name}' (score: {best_score}) -> address_id={address_id}")
            return address_id
            
        return None

    def process_event_address(self, event: dict) -> dict:
        """
        Uses the LLM to parse a structured address from the location, inserts or reuses the address in the DB,
        and updates the event with address_id and location = full_address from address table.
        """
        location = event.get("location", None)
        event_name = event.get("event_name", "Unknown Event")
        source = event.get("source", "Unknown Source")

        if location is None:
            pass  # Keep it as None
        elif isinstance(location, str):
            location = location.strip()

        # Handle case where location might be NaN (float), empty string, or 'Unknown'
        if (location is None or pd.isna(location) or not isinstance(location, str) or
            len(location) < 5 or 'Unknown' in str(location)):
            logging.info(
                "process_event_address: Location missing/invalid for event '%s' from %s, attempting building name extraction",
                event_name, source
            )

            # Try to extract building name from event details and match to existing addresses
            extracted_address_id = self._extract_address_from_event_details(event)
            if extracted_address_id:
                event["address_id"] = extracted_address_id
                full_address = self.get_full_address_from_id(extracted_address_id)
                if full_address:
                    event["location"] = full_address
                    logging.info(f"process_event_address: Found existing address via building name extraction: address_id={extracted_address_id}")
                    return event

            # DEDUPLICATION CHECK: Before creating a new address, check if source/event_name matches existing building
            dedup_addr_id = self.find_address_by_building_name(source, threshold=75)
            if dedup_addr_id:
                event["address_id"] = dedup_addr_id
                full_address = self.get_full_address_from_id(dedup_addr_id)
                if full_address:
                    event["location"] = full_address
                    logging.info(f"process_event_address: Found existing address via deduplication check: source='{source}' → address_id={dedup_addr_id}")
                    return event

            # If no match found, create a minimal but valid address entry
            minimal_address = {
                "address_id": 0,
                "full_address": f"Location details unavailable - {source}",
                "building_name": str(event_name)[:50],  # Use event name as building
                "street_number": "",
                "street_name": "",
                "street_type": "",
                "direction": None,
                "city": "Unknown",
                "met_area": None,
                "province_or_state": "BC", 
                "postal_code": None,
                "country_id": "CA",
                "time_stamp": None
            }
            
            address_id = self.resolve_or_insert_address(minimal_address)
            if address_id:
                event["address_id"] = address_id
                full_address = self.get_full_address_from_id(address_id)
                if full_address:
                    event["location"] = full_address
                else:
                    event["location"] = minimal_address["full_address"]  # Fallback to our description
                logging.info(f"process_event_address: Created minimal address entry with address_id={address_id}")
                return event
            else:
                logging.error("process_event_address: Failed to create minimal address entry, setting default values")
                # Final fallback - set reasonable defaults instead of None
                event["address_id"] = 0
                event["location"] = f"Location unavailable - {source}"
                return event

        # STEP 1: Check raw_locations cache (fastest - exact string match)
        cached_addr_id = self.lookup_raw_location(location)
        if cached_addr_id:
            full_address = self.get_full_address_from_id(cached_addr_id)
            event["address_id"] = cached_addr_id
            if full_address:
                event["location"] = full_address
            logging.info(f"process_event_address: Cache hit for '{location}' → address_id={cached_addr_id}")
            return event

        # STEP 2: Try intelligent address parsing (fuzzy matching, regex)
        quick_addr_id = self.quick_address_lookup(location)
        if quick_addr_id:
            # Cache this mapping for future use
            self.cache_raw_location(location, quick_addr_id)
            full_address = self.get_full_address_from_id(quick_addr_id)
            event["address_id"] = quick_addr_id
            if full_address:
                event["location"] = full_address
            logging.info(f"process_event_address: Quick lookup found address_id={quick_addr_id} for '{location}'")
            return event

        # STEP 3: LLM processing (last resort)
        # Generate the LLM prompt
        prompt, schema_type = self.llm_handler.generate_prompt(event.get("url", "address_fix"), location, "address_internet_fix")

        # Query the LLM
        llm_response = self.llm_handler.query_llm(event.get("url", "").strip(), prompt, schema_type)

        # Parse the LLM response into a usable dict
        parsed_results = self.llm_handler.extract_and_parse_json(llm_response, "address_fix", schema_type)
        if not parsed_results or not isinstance(parsed_results, list) or not isinstance(parsed_results[0], dict):
            logging.warning("process_event_address: Could not parse address from LLM response, creating minimal address")
            # Create minimal address using event name and location
            event_name = event.get("event_name") or "Unknown Event"
            minimal_address = {
                "building_name": str(event_name)[:50],
                "street_name": location[:50] if location else "Unknown Location",
                "city": "Unknown",
                "province_or_state": "BC",
                "country_id": "Canada"
            }
            address_id = self.resolve_or_insert_address(minimal_address)
            if address_id:
                event["address_id"] = address_id
                return event
            else:
                logging.error("process_event_address: Failed to create minimal address")
                return event

        # ✅ Normalize null-like strings in one place
        parsed_address = self.normalize_nulls(parsed_results[0])

        # Step 4: Get or insert address_id
        address_id = self.resolve_or_insert_address(parsed_address)
        
        # Ensure we got a valid address_id  
        if not address_id:
            logging.warning("process_event_address: resolve_or_insert_address failed, creating minimal address as last resort")
            # Last resort: create very minimal address entry
            event_name = event.get("event_name") or "Event"
            minimal_address = {
                "building_name": str(event_name)[:50],
                "city": "Location Unknown", 
                "province_or_state": "BC",
                "country_id": "Canada"
            }
            address_id = self.resolve_or_insert_address(minimal_address)
            if not address_id:
                logging.error("process_event_address: All address resolution attempts failed")
                return event

        # STEP 3: Cache the raw location → address_id mapping for future use
        self.cache_raw_location(location, address_id)

        # Step 5: Force consistency: always use address.full_address
        full_address = self.get_full_address_from_id(address_id)
        event["address_id"] = address_id
        if full_address:
            event["location"] = full_address

        return event


    def find_address_by_building_name(self, building_name: str, threshold: int = 75) -> Optional[int]:
        """
        Find an existing address by fuzzy matching on building_name.
        Prevents creation of duplicate addresses with the same venue name.

        Args:
            building_name (str): The venue/building name to search for
            threshold (int): Fuzzy match score threshold (0-100)

        Returns:
            address_id if found, None otherwise
        """
        from fuzzywuzzy import fuzz

        if not building_name or not isinstance(building_name, str):
            return None

        building_name = building_name.strip()

        try:
            # Query all addresses with building names
            building_matches = self.execute_query(
                "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            )

            best_score = 0
            best_addr_id = None

            for addr_id, existing_building in building_matches or []:
                if existing_building and existing_building.strip():
                    # Use partial_ratio which is more lenient for substring matches
                    score = fuzz.partial_ratio(building_name.lower().strip(), existing_building.lower().strip())

                    if score >= threshold and score > best_score:
                        best_score = score
                        best_addr_id = addr_id
                        logging.debug(f"find_address_by_building_name: '{building_name}' vs '{existing_building}' = {score}")

            if best_addr_id:
                logging.info(f"find_address_by_building_name: Found address_id={best_addr_id} for '{building_name}' (score={best_score})")
                return best_addr_id

            logging.debug(f"find_address_by_building_name: No match found for '{building_name}'")
            return None

        except Exception as e:
            logging.warning(f"find_address_by_building_name: Error looking up '{building_name}': {e}")
            return None

    def quick_address_lookup(self, location: str) -> Optional[int]:
        """
        Attempts to find an existing address without using LLM by:
        1. Exact string match on full_address
        2. Regex parsing to extract street_number + street_name for exact match
        3. Fuzzy matching on building names for the same street
        
        Returns address_id if found, None if LLM is needed
        """
        from fuzzywuzzy import fuzz
        import re
        
        # Step 1: Exact string match (already implemented)
        exact_match = self.execute_query(
            "SELECT address_id FROM address WHERE LOWER(full_address) = LOWER(:location)",
            {"location": location}
        )
        if exact_match:
            logging.info(f"quick_address_lookup: Exact match → address_id={exact_match[0][0]}")
            return exact_match[0][0]
        
        # Step 2: Parse basic components with regex
        street_pattern = r'(\d+)\s+([A-Za-z\s]+?)(?:,|\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd))'
        street_match = re.search(street_pattern, location, re.IGNORECASE)
        
        if street_match:
            street_number = street_match.group(1).strip()
            street_name_raw = street_match.group(2).strip()
            
            # Clean street name (remove common suffixes if they got included)
            street_name = re.sub(r'\b(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd)\b', 
                               '', street_name_raw, flags=re.IGNORECASE).strip()
            
            # Step 3: Find addresses with same street_number + street_name
            street_matches = self.execute_query("""
                SELECT address_id, building_name, full_address 
                FROM address 
                WHERE LOWER(street_number) = LOWER(:street_number) 
                AND LOWER(street_name) = LOWER(:street_name)
            """, {"street_number": street_number, "street_name": street_name})
            
            if street_matches:
                # Step 4: If only one match, use it regardless of building name
                if len(street_matches) == 1:
                    addr_id, building_name, full_addr = street_matches[0]
                    logging.info(f"quick_address_lookup: Single street match → address_id={addr_id}")
                    return addr_id
                
                # Step 5: Try fuzzy matching on building names
                building_pattern = r'^([^,\d]+?)(?:,|\s+\d+)'  # Text before first comma or number
                building_match = re.search(building_pattern, location.strip())
                
                if building_match:
                    location_building = building_match.group(1).strip()
                    
                    best_score = 0
                    best_addr_id = None
                    
                    for addr_id, existing_building, full_addr in street_matches:
                        if existing_building and existing_building.strip():
                            score = fuzz.ratio(location_building.lower(), existing_building.lower())
                            if score >= 85 and score > best_score:
                                best_score = score
                                best_addr_id = addr_id
                    
                    if best_addr_id:
                        logging.info(f"quick_address_lookup: Fuzzy building match (score={best_score}) → address_id={best_addr_id}")
                        return best_addr_id
        
        # Step 6: Fuzzy match on building names for locations without street numbers
        if not street_match:
            building_matches = self.execute_query(
                "SELECT address_id, building_name, full_address FROM address WHERE building_name IS NOT NULL"
            )
            
            best_score = 0
            best_addr_id = None
            
            for addr_id, building_name, full_addr in building_matches or []:
                if building_name and building_name.strip():
                    # Check if location is contained in building name or vice versa
                    score = fuzz.ratio(location.lower().strip(), building_name.lower().strip())
                    partial_score = fuzz.partial_ratio(location.lower().strip(), building_name.lower().strip())
                    
                    # Use the higher score
                    final_score = max(score, partial_score)
                    
                    if final_score >= 80 and final_score > best_score:
                        best_score = final_score
                        best_addr_id = addr_id
                        logging.debug(f"Building match candidate: '{location}' vs '{building_name}' = {final_score}")
            
            if best_addr_id:
                logging.info(f"quick_address_lookup: Fuzzy building name match (score={best_score}) → address_id={best_addr_id}")
                return best_addr_id
        
        # Step 7: Last resort - fuzzy match on full addresses for very similar ones
        all_addresses = self.execute_query("SELECT address_id, full_address FROM address")
        for addr_id, full_addr in all_addresses or []:
            if full_addr and fuzz.ratio(location.lower(), full_addr.lower()) >= 90:
                logging.info(f"quick_address_lookup: Fuzzy full address match → address_id={addr_id}")
                return addr_id
        
        logging.info(f"quick_address_lookup: No match found for '{location}', LLM required")
        return None

    def cache_raw_location(self, raw_location: str, address_id: int):
        """
        Cache a raw location string to address_id mapping for fast future lookups.
        Uses PostgreSQL ON CONFLICT to avoid duplicate key errors.
        """
        try:
            # PostgreSQL syntax: INSERT ... ON CONFLICT DO NOTHING
            insert_query = """
                INSERT INTO raw_locations (raw_location, address_id, created_at)
                VALUES (:raw_location, :address_id, :created_at)
                ON CONFLICT (raw_location) DO NOTHING
            """
            result = self.execute_query(insert_query, {
                "raw_location": raw_location,
                "address_id": address_id,
                "created_at": datetime.now()
            })
            logging.info(f"cache_raw_location: Cached '{raw_location}' → address_id={address_id}")
        except Exception as e:
            logging.warning(f"cache_raw_location: Failed to cache '{raw_location}': {e}")

    def lookup_raw_location(self, raw_location: str) -> Optional[int]:
        """
        Look up a raw location string in the cache to get its address_id.
        Returns address_id if found, None if not cached.
        """
        try:
            result = self.execute_query(
                "SELECT address_id FROM raw_locations WHERE raw_location = :raw_location",
                {"raw_location": raw_location}
            )
            if result:
                address_id = result[0][0]
                logging.info(f"lookup_raw_location: Cache hit for '{raw_location}' → address_id={address_id}")
                return address_id
            return None
        except Exception as e:
            logging.warning(f"lookup_raw_location: Cache lookup failed for '{raw_location}': {e}")
            return None

    def create_raw_locations_table(self):
        """
        Create the raw_locations table for caching location string to address_id mappings.
        Creates address table first if it doesn't exist to satisfy foreign key constraint.
        """
        # First ensure address table exists for foreign key constraint
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

        # PostgreSQL syntax (not SQLite)
        create_table_query = """
            CREATE TABLE IF NOT EXISTS raw_locations (
                raw_location_id SERIAL PRIMARY KEY,
                raw_location TEXT NOT NULL UNIQUE,
                address_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (address_id) REFERENCES address(address_id)
            )
        """
        try:
            # Create address table first
            self.execute_query(address_table_query)
            logging.info("create_raw_locations_table: Address table creation/verification completed")

            # Then create raw_locations table with foreign key
            self.execute_query(create_table_query)
            logging.info("create_raw_locations_table: Raw locations table creation/verification completed")
            
            # Create index for faster lookups
            index_query = "CREATE INDEX IF NOT EXISTS idx_raw_location ON raw_locations(raw_location)"
            self.execute_query(index_query)
            logging.info("create_raw_locations_table: Index creation/verification completed")
        except Exception as e:
            logging.error(f"create_raw_locations_table: Failed to create table: {e}")
            logging.error(f"create_raw_locations_table: SQL was: {create_table_query}")
    

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

    def clean_orphaned_references(self):
        """
        Clean up orphaned references in related tables to maintain referential integrity.

        This function removes:
        1. raw_locations records that reference non-existent addresses
        2. events records that reference non-existent addresses (if any)

        Returns:
            dict: Count of cleaned up records by table
        """
        cleanup_counts = {}

        try:
            # Clean up orphaned raw_locations
            cleanup_raw_locations_sql = """
            DELETE FROM raw_locations
            WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            raw_locations_count = self.execute_query(cleanup_raw_locations_sql)
            cleanup_counts['raw_locations'] = raw_locations_count or 0

            # Clean up events with non-existent address_ids (should be rare)
            cleanup_events_sql = """
            DELETE FROM events
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            events_count = self.execute_query(cleanup_events_sql)
            cleanup_counts['events'] = events_count or 0

            # Clean up events_history with non-existent address_ids (critical for preventing corruption)
            cleanup_events_history_sql = """
            DELETE FROM events_history
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            events_history_count = self.execute_query(cleanup_events_history_sql)
            cleanup_counts['events_history'] = events_history_count or 0

            logging.info(f"clean_orphaned_references(): Cleaned up {cleanup_counts['raw_locations']} raw_locations, {cleanup_counts['events']} events, and {cleanup_counts['events_history']} events_history records with orphaned address references")

        except Exception as e:
            logging.error(f"clean_orphaned_references(): Error cleaning orphaned references: {e}")

        return cleanup_counts

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

            # Clean up any orphaned references that might have been created
            self.clean_orphaned_references()

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

        # Write the df to a csv file (only locally, not on Render)
        if os.getenv('RENDER') != 'true':
            output_file = self.config['output']['events_urls_diff']
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if not os.path.isfile(output_file):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a', header=False, index=False)
            logging.info(f"def count_events_urls_end(): Wrote events and urls statistics to: {output_file}")
        else:
            logging.info(f"def count_events_urls_end(): Skipping CSV write on Render (ephemeral filesystem)")


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


    def normalize_url(self, url):
        """
        Normalize URLs by removing dynamic cache parameters that don't affect the underlying content.
        
        This is particularly important for Instagram and Facebook CDN URLs that include
        dynamic parameters like _nc_gid, _nc_ohc, oh, oe, etc. that change between requests
        but point to the same underlying image.
        
        Args:
            url (str): The original URL with potentially dynamic parameters
            
        Returns:
            str: Normalized URL with dynamic parameters removed
        """
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        
        parsed = urlparse(url)
        
        # Check if this is an Instagram or Facebook CDN URL
        # Be specific about Instagram/FB domains to avoid affecting other CDN URLs
        instagram_domains = {
            'instagram.com',
            'www.instagram.com', 
            'scontent.cdninstagram.com',
            'instagram.fcxh2-1.fna.fbcdn.net',
            'scontent.cdninstagram.com'
        }
        
        fb_cdn_domains = {
            domain for domain in [parsed.netloc] 
            if 'fbcdn.net' in domain and ('instagram' in domain or 'scontent' in domain)
        }
        
        is_instagram_cdn = (parsed.netloc in instagram_domains or 
                           any(domain in parsed.netloc for domain in instagram_domains) or
                           bool(fb_cdn_domains))
        
        if not is_instagram_cdn:
            return url
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # List of dynamic parameters to remove for Instagram/FB CDN URLs
        dynamic_params = {
            '_nc_gid',     # Cache group ID - changes between sessions
            '_nc_ohc',     # Cache hash - changes between requests  
            '_nc_oc',      # Cache parameter - changes between requests
            'oh',          # Hash parameter - changes between requests
            'oe',          # Expiration parameter - changes over time
            '_nc_zt',      # Zoom/time parameter
            '_nc_ad',      # Ad parameter
            '_nc_cid',     # Cache ID
            'ccb',         # Cache control parameter (sometimes)
        }
        
        # Remove dynamic parameters
        filtered_params = {k: v for k, v in query_params.items() 
                          if k not in dynamic_params}
        
        # Reconstruct URL with filtered parameters
        new_query = urlencode(filtered_params, doseq=True)
        normalized_url = urlunparse((
            parsed.scheme,
            parsed.netloc, 
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return normalized_url

    def should_process_url(self, url):
        """
        Determines whether a given URL should be processed based on its history in the database.

        The decision is made according to the following rules:
        0. If the URL is in the whitelist (data/urls/aaa_urls.csv), it should ALWAYS be processed.
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
        # If urls_df is empty (e.g., on production), always process
        if self.urls_df.empty:
            logging.info(f"should_process_url: URLs table not loaded (production mode), processing URL.")
            return True

        # Normalize URL to handle Instagram/FB CDN dynamic parameters
        normalized_url = self.normalize_url(url)

        # Log normalization if URL changed
        if normalized_url != url:
            logging.info(f"should_process_url: Normalized Instagram URL for comparison")

        # 0. Check if URL is in whitelist (always process) - URLs in data/urls/aaa_urls.csv should always be processed
        try:
            aaa_urls_path = os.path.join(self.config['input']['urls'], 'aaa_urls.csv')
            if os.path.exists(aaa_urls_path):
                aaa_urls_df = pd.read_csv(aaa_urls_path)
                if 'link' in aaa_urls_df.columns:
                    # Check if normalized_url matches any whitelist URL
                    if normalized_url in aaa_urls_df['link'].values:
                        logging.info(f"should_process_url: URL {normalized_url[:100]}... is in whitelist (aaa_urls.csv), processing it.")
                        return True
        except Exception as e:
            logging.warning(f"should_process_url: Could not check whitelist: {e}")

        # 1. Filter all rows for this normalized URL
        df_url = self.urls_df[self.urls_df['link'] == normalized_url]
        # If we've never recorded this normalized URL, process it
        if df_url.empty:
            logging.info(f"should_process_url: URL {normalized_url[:100]}... has never been seen before, processing it.")
            return True

        # 2. Look at the most recent "relevant" value
        last_relevant = df_url.iloc[-1]['relevant']
        if last_relevant and self.stale_date(normalized_url):
            logging.info(f"should_process_url: URL {normalized_url[:100]}... was last seen as relevant, processing it.")
            return True

        # 3. Last was False → check hit_ratio in self.urls_gb
        hit_row = self.urls_gb[self.urls_gb['link'] == normalized_url]

        if not hit_row.empty:
            # Extract scalars from the grouped DataFrame
            hit_ratio = hit_row.iloc[0]['hit_ratio']
            crawl_trys = hit_row.iloc[0]['crawl_try']

            if hit_ratio > 0.1 or crawl_trys <= 3:
                logging.info(
                    "should_process_url: URL %s was last seen as not relevant "
                    "but hit_ratio (%.2f) > 0.1 or crawl_try (%d) ≤ 3, processing it.",
                    normalized_url[:100] + "...", hit_ratio, crawl_trys
                )
                return True

        # 4. Otherwise, do not process this URL
        logging.info(f"should_process_url: URL {normalized_url[:100]}... does not meet criteria for processing, skipping it.")
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
        Always returns False to force re-scraping of images/PDFs.

        DISABLED: Previously checked events table and copied from events_history, but this
        caused data corruption issues. All images/PDFs are now re-scraped on every run to
        ensure fresh, accurate data with correct address normalization.

        Args:
            image_url (str): The URL of the image to check for associated events.

        Returns:
            bool: Always returns False to force re-scraping.
        """
        logging.info(f"check_image_events_exist(): Forcing re-scrape for URL: {image_url}")
        return False

        # DISABLED CODE - DO NOT USE (address_ids in events_history are corrupted):
        # # 2) Check history table
        # sql_hist = """
        # SELECT COUNT(*)
        # FROM events_history
        # WHERE url = :url
        # """
        # hist = self.execute_query(sql_hist, params)
        # if not (hist and hist[0][0] > 0):
        #     logging.info(f"check_image_events_exist(): No history events for URL: {image_url}")
        #     return False
        #
        # # 3) Copy only the most‐recent history row per unique event into events
        # sql_copy = """
        # INSERT INTO events (
        #     event_name, dance_style, description, day_of_week,
        #     start_date, end_date, start_time, end_time,
        #     source, location, price, url,
        #     event_type, address_id, time_stamp
        # )
        # SELECT
        #     sub.event_name, sub.dance_style, sub.description, sub.day_of_week,
        #     sub.start_date, sub.end_date, sub.start_time, sub.end_time,
        #     sub.source, sub.location, sub.price, sub.url,
        #     sub.event_type, sub.address_id, sub.time_stamp
        # FROM (
        #     SELECT DISTINCT ON (
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id
        #     )
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id, time_stamp
        #     FROM events_history
        #     WHERE url = :url
        #     AND start_date >= (CURRENT_DATE - (:days * INTERVAL '1 day'))
        #     ORDER BY
        #         event_name, dance_style, description, day_of_week,
        #         start_date, end_date, start_time, end_time,
        #         source, location, price, url,
        #         event_type, address_id,
        #         time_stamp DESC
        # ) AS sub
        # """
        # params_copy = {
        #     'url':  image_url,
        #     'days': self.config['clean_up']['old_events']  # e.g. 3 → includes start_date ≥ today−3d
        # }
        # self.execute_query(sql_copy, params_copy)
        #
        # logging.info(f"check_image_events_exist(): Copied most‐recent history events into events for URL: {image_url}")
        # return True
    

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
        Also handles pandas NaN values and various null representations.
        Applies to all keys in the given dictionary.
        """
        import pandas as pd
        import numpy as np
        
        cleaned = {}
        null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}
        
        for key, value in record.items():
            # Handle pandas/numpy NaN
            if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                cleaned[key] = None
            # Handle string nulls (case insensitive, strip whitespace)
            elif isinstance(value, str) and value.strip().lower() in null_strings:
                cleaned[key] = None
            # Handle numeric 0 that shouldn't be null (keep as is)
            elif value == 0:
                cleaned[key] = value
            # Handle other falsy values that should be None
            elif not value and value != 0:
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned
    

    def clean_null_strings_in_address(self):
        """
        Replaces string 'null', 'none', 'nan', and '' with actual SQL NULLs in address table.
        """
        fields = [
            "full_address", "building_name", "street_number", "street_name", "direction",
            "city", "met_area", "province_or_state", "postal_code", "country_id"
        ]
        for field in fields:
            query = f"""
                UPDATE address
                SET {field} = NULL
                WHERE TRIM(LOWER({field})) IN ('null', 'none', 'nan', '', '[null]', '(null)', 'n/a', 'na');
            """
            self.execute_query(query)
        logging.info("Cleaned up string 'null's in address table.")
    
    
    def standardize_postal_codes(self):
        """
        Standardizes Canadian postal codes to format V8N 1S3 (with space).
        """
        query = """
            UPDATE address
            SET postal_code = UPPER(
                CASE 
                    WHEN LENGTH(REPLACE(postal_code, ' ', '')) = 6 
                    THEN SUBSTRING(REPLACE(postal_code, ' ', ''), 1, 3) || ' ' || SUBSTRING(REPLACE(postal_code, ' ', ''), 4, 3)
                    ELSE postal_code
                END
            )
            WHERE postal_code IS NOT NULL 
            AND postal_code ~ '^[A-Za-z][0-9][A-Za-z][ ]?[0-9][A-Za-z][0-9]$'
        """
        result = self.execute_query(query)
        logging.info("Standardized postal code formats to V8N 1S3 pattern.")
        return result


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
            # Clean up any remaining orphaned references before sequence reset
            self.clean_orphaned_references()

            # COMMENTED OUT: reset_address_id_sequence() causes race conditions with concurrent processes
            # This function renumbers all address_ids sequentially (1, 2, 3...) and updates all foreign keys.
            # Problem: If multiple processes (pipeline, web service, or manual operations) access the database
            # simultaneously, the renumbering creates corruption as IDs change mid-operation.
            #
            # To re-enable safely:
            # 1. Ensure NO other processes are accessing the database (stop web service, CRON jobs, manual queries)
            # 2. Uncomment the line below
            # 3. Run the pipeline once for maintenance
            # 4. Re-comment this line before resuming normal operations
            #
            # self.reset_address_id_sequence()

            self.update_full_address_with_building_names()
            
            # Fix events with address_id = 0 using existing deduplication logic
            logging.info("driver(): Creating DeduplicationHandler for fix_problem_events")
            from dedup_llm import DeduplicationHandler
            dedup_handler = DeduplicationHandler(config_path='config/config.yaml')
            dedup_handler.fix_problem_events(dry_run=False)
            logging.info("driver(): Completed fix_problem_events via DeduplicationHandler")

        # Close the database connection
        self.conn.dispose()  # Using dispose() for SQLAlchemy Engine
        logging.info("driver(): Database operations completed successfully.")

    def reset_address_id_sequence(self):
        """
        Reset the address_id sequence to start from 1, updating all references in the events table.

        This method:
        0. Cleans up orphaned raw_locations records pointing to non-existent addresses
        1. Creates a mapping of old address_ids to new sequential IDs (1, 2, 3, ...)
        2. Updates all events table records with the new address_id values
        3. Updates the address table with new sequential IDs
        4. Resets the PostgreSQL sequence to continue from the max ID + 1

        Returns:
            int: Number of addresses that were renumbered
        """
        try:
            logging.info("reset_address_id_sequence(): Starting address ID sequence reset...")

            # Step 0: Clean up orphaned raw_locations records that reference non-existent addresses
            cleanup_orphaned_sql = """
            DELETE FROM raw_locations
            WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            orphaned_count = self.execute_query(cleanup_orphaned_sql)
            logging.info(f"reset_address_id_sequence(): Cleaned up {orphaned_count} orphaned raw_locations records")

            # Step 1: Get current addresses ordered by address_id and create mapping
            get_addresses_sql = """
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, met_area, province_or_state, 
                   postal_code, country_id, time_stamp
            FROM address 
            ORDER BY address_id;
            """
            
            addresses_df = pd.read_sql(get_addresses_sql, self.conn)
            
            if addresses_df.empty:
                logging.info("reset_address_id_sequence(): No addresses found to renumber.")
                return 0
            
            # Create mapping from old address_id to new sequential ID
            address_mapping = {}
            for idx, row in addresses_df.iterrows():
                old_id = row['address_id']
                new_id = idx + 1  # Start from 1
                address_mapping[old_id] = new_id
            
            logging.info(f"reset_address_id_sequence(): Created mapping for {len(address_mapping)} addresses")
            
            # Step 2: Create temporary table with new sequential IDs
            create_temp_table_sql = """
            CREATE TEMPORARY TABLE address_temp AS 
            SELECT * FROM address WHERE 1=0;
            """
            self.execute_query(create_temp_table_sql)
            
            # Insert addresses with new sequential IDs
            for idx, row in addresses_df.iterrows():
                new_id = idx + 1
                insert_temp_sql = """
                INSERT INTO address_temp (address_id, full_address, building_name, street_number, 
                                        street_name, street_type, direction, city, met_area, 
                                        province_or_state, postal_code, country_id, time_stamp)
                VALUES (:new_id, :full_address, :building_name, :street_number, :street_name, 
                        :street_type, :direction, :city, :met_area, :province_or_state, 
                        :postal_code, :country_id, :time_stamp);
                """
                params = {
                    'new_id': new_id,
                    'full_address': row['full_address'],
                    'building_name': row['building_name'],
                    'street_number': row['street_number'],
                    'street_name': row['street_name'],
                    'street_type': row['street_type'],
                    'direction': row['direction'],
                    'city': row['city'],
                    'met_area': row['met_area'],
                    'province_or_state': row['province_or_state'],
                    'postal_code': row['postal_code'],
                    'country_id': row['country_id'],
                    'time_stamp': row['time_stamp']
                }
                self.execute_query(insert_temp_sql, params)
            
            # Step 3: Update all tables that reference address_id with new address_ids
            events_updated = 0
            events_history_updated = 0
            raw_locations_updated = 0
            
            for old_id, new_id in address_mapping.items():
                # Update events table
                update_events_sql = """
                UPDATE events 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_events_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_updated += 1
                
                # Update events_history table (only for address_ids that exist)
                update_events_history_sql = """
                UPDATE events_history 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_events_history_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_history_updated += 1
                
                # Update raw_locations table (this has the foreign key constraint)
                update_raw_locations_sql = """
                UPDATE raw_locations 
                SET address_id = :new_id 
                WHERE address_id = :old_id;
                """
                result = self.execute_query(update_raw_locations_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    raw_locations_updated += 1
            
            logging.info(f"reset_address_id_sequence(): Updated address_id in events table for {events_updated} different address IDs")
            logging.info(f"reset_address_id_sequence(): Updated address_id in events_history table for {events_history_updated} different address IDs")
            logging.info(f"reset_address_id_sequence(): Updated address_id in raw_locations table for {raw_locations_updated} different address IDs")
            
            # Step 4: Replace original address table with renumbered version
            # Delete all from original table
            self.execute_query("DELETE FROM address;")
            
            # Insert from temp table
            copy_back_sql = """
            INSERT INTO address (address_id, full_address, building_name, street_number, 
                               street_name, street_type, direction, city, met_area, 
                               province_or_state, postal_code, country_id, time_stamp)
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, met_area, province_or_state, 
                   postal_code, country_id, time_stamp
            FROM address_temp;
            """
            self.execute_query(copy_back_sql)
            
            # Step 5: Reset the PostgreSQL sequence
            max_id = len(addresses_df)
            
            # First, get the actual sequence name for address_id
            sequence_query = "SELECT pg_get_serial_sequence('address', 'address_id');"
            sequence_result = self.execute_query(sequence_query)
            
            if sequence_result and sequence_result[0][0]:
                sequence_name = sequence_result[0][0].split('.')[-1]  # Remove schema prefix if present
                reset_sequence_sql = f"SELECT setval('{sequence_name}', {max_id}, true);"
                self.execute_query(reset_sequence_sql)
                logging.info(f"reset_address_id_sequence(): Reset sequence '{sequence_name}' to {max_id}")
            else:
                # Create proper sequence if it doesn't exist
                create_seq_sql = f"""
                CREATE SEQUENCE IF NOT EXISTS address_address_id_seq
                START WITH {max_id + 1}
                INCREMENT BY 1
                NO MINVALUE
                NO MAXVALUE
                CACHE 1;
                """
                self.execute_query(create_seq_sql)
                
                # Update column default
                alter_col_sql = """
                ALTER TABLE address 
                ALTER COLUMN address_id SET DEFAULT nextval('address_address_id_seq');
                """
                self.execute_query(alter_col_sql)
                
                # Set sequence ownership
                alter_seq_sql = """
                ALTER SEQUENCE address_address_id_seq OWNED BY address.address_id;
                """
                self.execute_query(alter_seq_sql)
                
                logging.info(f"reset_address_id_sequence(): Created new sequence 'address_address_id_seq' starting from {max_id + 1}")
            
            # Clean up temp table
            self.execute_query("DROP TABLE address_temp;")
            
            logging.info(f"reset_address_id_sequence(): Successfully reset address_id sequence. "
                        f"Renumbered {len(address_mapping)} addresses, sequence reset to start from {max_id + 1}")
            
            return len(address_mapping)
            
        except Exception as e:
            logging.error(f"reset_address_id_sequence(): Error during address ID sequence reset: {e}")
            # Clean up temp table if it exists
            try:
                self.execute_query("DROP TABLE IF EXISTS address_temp;")
            except:
                pass
            raise

    def update_full_address_with_building_names(self):
        """
        Update existing full_address records using the standardized format.
        
        Rebuilds full_address for all records using the build_full_address method
        to ensure consistency across the database.
        
        Returns:
            int: Number of addresses updated
        """
        try:
            logging.info("update_full_address_with_building_names(): Starting full_address standardization...")
            
            # Get all addresses to standardize their full_address
            find_addresses_sql = """
            SELECT address_id, full_address, building_name, street_number, street_name, 
                   street_type, direction, city, province_or_state, postal_code, country_id
            FROM address 
            ORDER BY address_id;
            """
            
            addresses_df = pd.read_sql(find_addresses_sql, self.conn)
            
            if addresses_df.empty:
                logging.info("update_full_address_with_building_names(): No addresses found.")
                return 0
            
            logging.info(f"update_full_address_with_building_names(): Processing {len(addresses_df)} addresses for standardization")
            
            updated_count = 0
            for _, row in addresses_df.iterrows():
                # Build standardized full_address using the new method
                new_full_address = self.build_full_address(
                    building_name=row['building_name'],
                    street_number=row['street_number'],
                    street_name=row['street_name'],
                    street_type=row['street_type'],
                    city=row['city'],
                    province_or_state=row['province_or_state'],
                    postal_code=row['postal_code'],
                    country_id=row['country_id']
                )
                
                # Only update if the new address is different from current
                current_address = row['full_address'] or ""
                if new_full_address != current_address:
                    update_sql = """
                    UPDATE address 
                    SET full_address = :new_full_address 
                    WHERE address_id = :address_id;
                    """
                    
                    result = self.execute_query(update_sql, {
                        'new_full_address': new_full_address,
                        'address_id': row['address_id']
                    })
                    
                    if result is not None:  # Query executed successfully
                        updated_count += 1
                        logging.debug(f"Updated address_id {row['address_id']}: '{current_address}' -> '{new_full_address}'")
            
            logging.info(f"update_full_address_with_building_names(): Successfully updated {updated_count} addresses")
            return updated_count
            
        except Exception as e:
            logging.error(f"update_full_address_with_building_names(): Error updating full_address records: {e}")
            raise
        

if __name__ == "__main__":
    # Load configuration from a YAML file
    # Setup centralized logging
    from logging_config import setup_logging
    setup_logging('db')

    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

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
