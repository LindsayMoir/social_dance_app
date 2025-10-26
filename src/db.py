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
from repositories.address_repository import AddressRepository
from repositories.url_repository import URLRepository
from repositories.event_repository import EventRepository
from repositories.event_management_repository import EventManagementRepository
from repositories.event_analysis_repository import EventAnalysisRepository
from repositories.address_resolution_repository import AddressResolutionRepository


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
        # Initialize llm_handler to None - will be set later via set_llm_handler()
        self.llm_handler = None

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

        # Initialize AddressRepository for centralized address management
        self.address_repo = AddressRepository(self)
        logging.info("__init__(): AddressRepository initialized")

        # Initialize URLRepository for centralized URL management
        self.url_repo = URLRepository(self)
        logging.info("__init__(): URLRepository initialized")

        # Load blacklist domains AFTER URLRepository is initialized (it depends on url_repo)
        self.load_blacklist_domains()

        # Initialize EventRepository for centralized event management
        self.event_repo = EventRepository(self)
        logging.info("__init__(): EventRepository initialized")

        # Initialize EventManagementRepository for data quality operations
        self.event_mgmt_repo = EventManagementRepository(self)
        logging.info("__init__(): EventManagementRepository initialized")

        # Initialize EventAnalysisRepository for reporting/analysis operations
        self.event_analysis_repo = EventAnalysisRepository(self)
        logging.info("__init__(): EventAnalysisRepository initialized")

        # Initialize AddressResolutionRepository for LLM-based address resolution
        self.address_resolution_repo = AddressResolutionRepository(self, self.llm_handler)
        logging.info("__init__(): AddressResolutionRepository initialized")

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
        Wrapper delegating to URLRepository.load_blacklist().

        Loads a set of blacklisted domains from a CSV file specified in the configuration.
        This method is maintained for backward compatibility.
        """
        self.url_repo.load_blacklist()
        # Also maintain local reference for backward compatibility
        self.blacklisted_domains = self.url_repo.blacklisted_domains

    def avoid_domains(self, url):
        """
        Wrapper delegating to URLRepository.is_blacklisted().

        Check if the given URL contains any blacklisted domain.
        This method is maintained for backward compatibility.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL contains any domain from the blacklist, False otherwise.
        """
        return self.url_repo.is_blacklisted(url)
    

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

        # Create the 'run_results' table - tracks execution statistics for each scraper run
        run_results_table_query = """
            CREATE TABLE IF NOT EXISTS run_results (
                run_result_id SERIAL PRIMARY KEY,
                file_name TEXT,
                start_time_df TEXT,
                events_count_start INTEGER,
                urls_count_start INTEGER,
                events_count_end INTEGER,
                urls_count_end INTEGER,
                new_events_in_db INTEGER,
                new_urls_in_db INTEGER,
                time_stamp TIMESTAMP,
                elapsed_time TEXT
            )
        """
        self.execute_query(run_results_table_query)
        logging.info("create_tables: 'run_results' table created or already exists.")

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
        Wrapper delegating to URLRepository.write_url_to_db().

        Appends a new URL activity record to the 'urls' table in the database.
        This method is maintained for backward compatibility.

        Args:
            url_row (tuple): A tuple containing the following fields in order:
                - link (str): The URL to be logged.
                - parent_url (str): The parent URL from which this link was found.
                - source (str): The source or context of the URL.
                - keywords (str | list | tuple | set): Associated keywords, which can be a string or an iterable.
                - relevant (bool | int): Indicator of relevance.
                - crawl_try (int): Number of crawl attempts.
                - time_stamp (str | datetime): Timestamp of the activity.
        """
        return self.url_repo.write_url_to_db(url_row)
    

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

        This method now delegates to AddressRepository for centralized address management.

        Args:
            parsed_address (dict): Dictionary of parsed address fields.

        Returns:
            int or None: The address_id of the matched or newly inserted address.
        """
        return self.address_repo.resolve_or_insert_address(parsed_address)
    

    def build_full_address(self, building_name: str = None, street_number: str = None,
                          street_name: str = None, street_type: str = None,
                          city: str = None, province_or_state: str = None,
                          postal_code: str = None, country_id: str = None) -> str:
        """
        Builds a standardized full_address string from address components.

        This method now delegates to AddressRepository for centralized address formatting.

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
        return self.address_repo.build_full_address(
            building_name=building_name,
            street_number=street_number,
            street_name=street_name,
            street_type=street_type,
            city=city,
            province_or_state=province_or_state,
            postal_code=postal_code,
            country_id=country_id
        )

    def get_full_address_from_id(self, address_id: int) -> Optional[str]:
        """
        Returns the full_address from the address table for the given address_id.

        This method now delegates to AddressRepository for centralized address lookup.
        """
        return self.address_repo.get_full_address_from_id(address_id)


    def format_address_from_db_row(self, db_row):
        """
        Constructs a formatted address string from a database row.

        This method now delegates to AddressRepository for centralized address formatting.

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
        return self.address_repo.format_address_from_db_row(db_row)
    

    def write_events_to_db(self, df, url, parent_url, source, keywords):
        """
        Wrapper delegating to EventRepository.write_events_to_db().

        Processes and writes event data to the 'events' table in the database.
        This method is maintained for backward compatibility.

        Args:
            df (pd.DataFrame): DataFrame containing raw event data.
            url (str): The URL from which the events data was sourced.
            parent_url (str): The parent URL, if applicable.
            source (str): The source identifier for the event data.
            keywords (str or list): Keywords or dance styles associated with events.

        Returns:
            bool: True if events were written, False otherwise.
        """
        return self.event_repo.write_events_to_db(df, url, parent_url, source, keywords)


    def _rename_google_calendar_columns(self, df):
        """
        Wrapper delegating to EventRepository._rename_google_calendar_columns().

        Renames columns of a DataFrame containing Google Calendar event data.
        This method is maintained for backward compatibility.

        Args:
            df (pd.DataFrame): The input DataFrame with Google Calendar column names.

        Returns:
            pd.DataFrame: A DataFrame with columns renamed to standardized names.
        """
        return self.event_repo._rename_google_calendar_columns(df)

    def _convert_datetime_fields(self, df):
        """
        Wrapper delegating to EventRepository._convert_datetime_fields().

        Convert datetime-related columns to appropriate date and time types.
        This method is maintained for backward compatibility.

        Args:
            df (pd.DataFrame): The DataFrame to convert (modified in place).
        """
        return self.event_repo._convert_datetime_fields(df)

    def _clean_day_of_week_field(self, df):
        """
        Wrapper delegating to EventRepository._clean_day_of_week_field().

        Clean and standardize the day_of_week field.
        This method is maintained for backward compatibility.

        Args:
            df (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: DataFrame with cleaned day_of_week values.
        """
        return self.event_repo._clean_day_of_week_field(df)

    def _filter_events(self, df):
        """
        Wrapper delegating to EventRepository._filter_events().

        Filter a DataFrame of events by removing incomplete and old events.
        This method is maintained for backward compatibility.

        Args:
            df (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return self.event_repo._filter_events(df)
    
    
    def update_event(self, event_identifier, new_data, best_url):
        """
        Wrapper delegating to EventRepository.update_event().

        Update an existing event in the database by overlaying new data.
        This method is maintained for backward compatibility.

        Args:
            event_identifier (dict): Dictionary specifying criteria to identify the event.
            new_data (dict): Dictionary containing new values to update.
            best_url (str): The URL to set as the event's 'url' field.

        Returns:
            bool: True if the event was found and updated, False otherwise.
        """
        return self.event_repo.update_event(event_identifier, new_data, best_url)


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
        Wrapper: Attempts to extract building names from event_name and description,
        then matches them against existing addresses in the database.

        Delegates to AddressResolutionRepository.
        """
        return self.address_resolution_repo._extract_address_from_event_details(event)

    def process_event_address(self, event: dict) -> dict:
        """
        Wrapper: Uses the LLM to parse a structured address from the location, inserts or reuses
        the address in the DB, and updates the event with address_id and location.

        Delegates to AddressResolutionRepository.
        """
        return self.address_resolution_repo.process_event_address(event)


    def find_address_by_building_name(self, building_name: str, threshold: int = 75) -> Optional[int]:
        """
        Find an existing address by fuzzy matching on building_name.
        Prevents creation of duplicate addresses with the same venue name.

        This method now delegates to AddressRepository for centralized address matching.

        Args:
            building_name (str): The venue/building name to search for
            threshold (int): Fuzzy match score threshold (0-100)

        Returns:
            address_id if found, None otherwise
        """
        return self.address_repo.find_address_by_building_name(building_name, threshold)

    def quick_address_lookup(self, location: str) -> Optional[int]:
        """
        Attempts to find an existing address without using LLM by:
        1. Exact string match on full_address
        2. Regex parsing to extract street_number + street_name for exact match
        3. Fuzzy matching on building names for the same street

        This method now delegates to AddressRepository for centralized address lookup.

        Returns address_id if found, None if LLM is needed
        """
        return self.address_repo.quick_address_lookup(location)

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
        Wrapper: Updates all events so that location matches canonical address table values.
        Delegates to EventAnalysisRepository.
        """
        return self.event_analysis_repo.sync_event_locations_with_address_table()

    def clean_orphaned_references(self):
        """
        Wrapper: Clean up orphaned references in related tables.
        Delegates to EventAnalysisRepository.
        """
        return self.event_analysis_repo.clean_orphaned_references()

    def dedup(self):
        """
        Wrapper: Deduplicates the events table.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.dedup()


    def fetch_events_dataframe(self):
        """
        Wrapper: Fetch all events from the database and return as a sorted DataFrame.
        Delegates to EventRepository.
        """
        return self.event_repo.fetch_events_dataframe()

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
        Wrapper: Deletes events older than the configured threshold.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.delete_old_events()


    def delete_likely_dud_events(self):
        """
        Wrapper: Deletes low-quality/invalid events based on multiple criteria.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.delete_likely_dud_events()
        

    def delete_event(self, url, event_name, start_date):
        """
        Wrapper: Deletes an event from the 'events' table based on event name and start date.
        Delegates to EventRepository.
        """
        return self.event_repo.delete_event(url, event_name, start_date)


    def delete_events_with_nulls(self):
        """
        Wrapper: Deletes events with critical null values.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.delete_events_with_nulls()


    def delete_event_with_event_id(self, event_id):
        """
        Wrapper: Deletes an event from the 'events' table based on event_id.
        Delegates to EventRepository.
        """
        return self.event_repo.delete_event_with_event_id(event_id)

    
    def delete_multiple_events(self, event_ids):
        """
        Wrapper: Deletes multiple events from the 'events' table based on a list of event IDs.
        Delegates to EventRepository.
        """
        return self.event_repo.delete_multiple_events(event_ids)


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


    def count_events_urls_start(self, file_name):
        """
        Wrapper: Counts events and URLs at process start.
        Delegates to EventAnalysisRepository.
        """
        return self.event_analysis_repo.count_events_urls_start(file_name)
    

    def count_events_urls_end(self, start_df, file_name):
        """
        Wrapper: Counts events and URLs at process end and writes statistics.
        Delegates to EventAnalysisRepository.
        """
        return self.event_analysis_repo.count_events_urls_end(start_df, file_name)


    def stale_date(self, url):
        """
        Wrapper delegating to URLRepository.stale_date().

        Determines whether the most recent event associated with the given URL is considered "stale".
        This method is maintained for backward compatibility.

        Args:
            url (str): The URL whose most recent event's staleness is to be checked.

        Returns:
            bool: True if stale or no events, False if recent events exist.
        """
        return self.url_repo.stale_date(url)


    def normalize_url(self, url):
        """
        Wrapper delegating to URLRepository.normalize_url().

        Normalize URLs by removing dynamic cache parameters that don't affect the underlying content.
        This method is maintained for backward compatibility.

        Args:
            url (str): The original URL with potentially dynamic parameters

        Returns:
            str: Normalized URL with dynamic parameters removed
        """
        return self.url_repo.normalize_url(url)

    def should_process_url(self, url):
        """
        Wrapper delegating to URLRepository.should_process_url().

        Determines whether a given URL should be processed based on its history in the database.
        This method is maintained for backward compatibility and passes the necessary DataFrames
        to URLRepository for decision-making.

        Args:
            url (str): The URL to evaluate for processing.

        Returns:
            bool: True if the URL should be processed according to the criteria, False otherwise.
        """
        # Pass urls_df and urls_gb to URLRepository for complex decision logic
        return self.url_repo.should_process_url(
            url,
            urls_df=self.urls_df if hasattr(self, 'urls_df') else None,
            urls_gb=self.urls_gb if hasattr(self, 'urls_gb') else None
        )


    def update_dow_date(self, event_id: int, corrected_date) -> bool:
        """
        Wrapper: Updates the start_date and end_date for an event.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.update_dow_date(event_id, corrected_date)


    def check_dow_date_consistent(self) -> dict:
        """
        Wrapper: Validates and corrects day-of-week date consistency for all events.
        Delegates to EventManagementRepository.
        """
        return self.event_mgmt_repo.check_dow_date_consistent()


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
