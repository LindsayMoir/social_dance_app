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

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fuzzywuzzy import fuzz
import logging
import numpy as np
import os
import pandas as pd
import re  # Added missing import
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
import sys
import yaml


class DatabaseHandler():
    def __init__(self, config):
        """
        Initializes the DatabaseHandler with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters for the database connection.
        """
        self.config = config

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
                "DROP TABLE IF EXISTS address CASCADE;",
                "DROP TABLE IF EXISTS events CASCADE;",
                "DROP TABLE IF EXISTS organization CASCADE;",
                "DROP TABLE IF EXISTS urls CASCADE;"
            ]
        elif self.config['testing']['drop_tables'] == 'events':
            drop_queries = [
                "DROP TABLE IF EXISTS events CASCADE;"
            ]
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
                link TEXT PRIMARY KEY,
                source TEXT,
                keywords TEXT,
                other_link TEXT,
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
                street_number TEXT,
                street_name TEXT,
                street_type TEXT,
                postal_box TEXT,
                city TEXT,
                province_or_state TEXT,
                postal_code TEXT,
                country_id TEXT,
                time_stamp TIMESTAMP
            )
        """
        self.execute_query(address_table_query)
        logging.info("create_tables: 'address' table created or already exists.")

        # Create the 'address' table
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


    def execute_query(self, query, params=None):
        """
        Executes a given SQL query with optional parameters.

        Args:
            query (str): The SQL query to execute.
            params (dict): Optional dictionary of parameters for parameterized queries.

        Returns:
            list of dict or None: A list of rows as dictionaries if the query returns rows, else None.
        """
        if self.conn is None:
            logging.error("execute_query: No database connection available.")
            return None

        # Handle NaN values in params (such as address_id)
        if params:
            for key, value in params.items():
                if pd.isna(value):
                    if isinstance(value, (list, np.ndarray, pd.Series)):
                        if pd.isna(value).any():  # Check if the value is NaN
                            params[key] = None  # Set to None (NULL in SQL)
                    elif pd.isna(value):  # Check if the single value is NaN
                        params[key] = None  # Set to None (NULL in SQL)

        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(query), params or {})
                if result.returns_rows:
                    rows = result.fetchall()
                    return rows
                connection.commit()
                return None
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


    def update_url(self, url, update_other_link, relevant, increment_crawl_try):
        """
        Updates an entry in the 'urls' table with the provided URL and other details.

        Args:
            url (str): The URL to be updated.
            update_other_link (str): The new value for 'other_link'. If 'No', it won't be updated.
            relevant (bool): The new value for 'relevant'.
            increment_crawl_try (int): The number to increment 'crawl_try' by.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        # See if url is in the urls table
        select_query = """
            SELECT * FROM urls
            WHERE link = :url
        """
        select_params = {'url': url}
        result = self.execute_query(select_query, select_params)
        if result and len(result) > 0:

            # Prepare the update query with conditional 'other_link'
            if update_other_link != 'No':
                update_query = """
                    UPDATE urls
                    SET 
                        time_stamp = :current_time,
                        other_link = :other_link,
                        crawl_try = crawl_try + :increment,
                        relevant = :relevant
                    WHERE link = :url
                """
                update_params = {
                    'current_time': datetime.now(),
                    'other_link': update_other_link,
                    'increment': increment_crawl_try,
                    'relevant': relevant,
                    'url': url
                }
            else:
                update_query = """
                    UPDATE urls
                    SET 
                        time_stamp = :current_time,
                        crawl_try = crawl_try + :increment,
                        relevant = :relevant
                    WHERE link = :url
                """
                update_params = {
                    'current_time': datetime.now(),
                    'increment': increment_crawl_try,
                    'relevant': relevant,
                    'url': url
                }

            # Execute the update
            update_result = self.execute_query(update_query, update_params)
            if update_result:
                logging.info("def update_url(): Updated URL '%s' successfully.", url)
                return True
            else:
                logging.info("def update_url(): Failed to update URL '%s'.", url)
                return False
        else:
            logging.info("def update_url(): URL '%s' not found for update.", url)
            return False
        

    def write_url_to_db(self, source, keywords, url, other_link, relevant, increment_crawl_try):
        """
        Inserts a new URL into the 'urls' table or updates it if it already exists.

        Args:
            source (str): Organization names related to the URL.
            keywords (list): Keywords related to the URL.
            url (str): The URL to be written or updated.
            other_link (str): Other link associated with the URL.
            relevant (bool): Indicates if the URL is relevant.
            increment_crawl_try (int): The number of times the URL has been crawled.
        """
        insert_query = """
            INSERT INTO urls (time_stamp, source, keywords, link, other_link, relevant, crawl_try)
            VALUES (:time_stamp, :source, :keywords, :link, :other_link, :relevant, :crawl_try)
            ON CONFLICT (link) DO UPDATE
            SET 
                time_stamp = EXCLUDED.time_stamp,
                other_link = CASE 
                                WHEN :other_link != 'No' THEN EXCLUDED.other_link 
                                ELSE urls.other_link 
                            END,
                relevant = EXCLUDED.relevant,
                crawl_try = urls.crawl_try + :increment_crawl_try;
        """
        insert_params = {
            'time_stamp': datetime.now(),
            'source': source,
            'keywords': keywords,
            'link': url,
            'other_link': other_link,
            'relevant': relevant,
            'crawl_try': increment_crawl_try,
            'increment_crawl_try': increment_crawl_try
        }
        
        try:
            self.execute_query(insert_query, insert_params)
            logging.info("def write_url_to_db(): URL '%s' inserted or updated in the 'urls' table.", url)
        except Exception as e:
            logging.error("def write_url_to_db(): Failed to insert/update URL '%s': %s", url, e)
    

    def clean_events(self, df):
        """
        Cleans and processes the events DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing event data.

        Returns:
            pandas.DataFrame: The cleaned and processed DataFrame.
        """
        # Avoid modifying the original DataFrame
        df = df.copy()

        # Ensure required columns exist
        required_columns = ['htmlLink', 'summary', 'start.date', 'end.date', 'location', 'start.dateTime', 'end.dateTime', 'description']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''

        # Move values from 'start.date' and 'end.date' to 'start.dateTime' and 'end.dateTime' if necessary
        df['start.dateTime'] = df['start.dateTime'].fillna(df['start.date'])
        df['end.dateTime'] = df['end.dateTime'].fillna(df['end.date'])

        # Drop the 'start.date' and 'end.date' columns
        df.drop(columns=['start.date', 'end.date'], inplace=True)

        # Subset df to only useful columns 
        df = df[['htmlLink', 'summary', 'location', 'start.dateTime', 'end.dateTime', 'description']]

        # Extract and convert the price
        df['Price'] = df['description'].str.extract(r'\$(\d{1,5})')[0]

        # Clean the description
        df['description'] = df['description'].apply(
            lambda x: re.sub(r'\s{2,}', ' ', re.sub(r'<[^>]*>', ' ', str(x) if pd.notnull(x) else '')).strip()
        ).str.replace('&#39;', "'").str.replace("you're", "you are")

        # Function to split datetime into date and time
        def split_datetime(datetime_str):
            if 'T' in datetime_str:
                date_str, time_str = datetime_str.split('T')
                time_str = time_str[:8]  # Remove the timezone part
            else:
                date_str = datetime_str
                time_str = None
            return date_str, time_str

        # Apply the function to extract dates and times
        df['Start_Date'], df['Start_Time'] = zip(*df['start.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))
        df['End_Date'], df['End_Time'] = zip(*df['end.dateTime'].apply(lambda x: split_datetime(x) if x else ('', '')))

        # Drop columns
        df.drop(columns=['start.dateTime', 'end.dateTime'], inplace=True)

        # Rename columns
        df = df.rename(columns={
            'htmlLink': 'URL',
            'summary': 'Name_of_the_Event',
            'location': 'Location',
            'description': 'Description'
        })

        # Add 'Type_of_Event' column
        # Dictionary to map words of interest (woi) to 'Type_of_Event'
        event_type_map = {
            'class': 'class',
            'dance': 'social dance',
            'dancing': 'social dance',
            'weekend': 'workshop',
            'workshop': 'workshop'
        }

        # Function to determine 'Type_of_Event'
        def determine_event_type(name):
            name_lower = name.lower()
            if 'class' in name_lower and 'dance' in name_lower:
                return 'social dance'  # Priority rule
            for woi, event_type in event_type_map.items():
                if woi in name_lower:
                    return event_type
            return 'other'  # Default if no woi match

        # Apply the function to determine 'Type_of_Event'
        df['Type_of_Event'] = df['Description'].apply(determine_event_type)

        # Convert Start_Date and End_Date to date format
        df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.date
        df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce').dt.date

        # Extract the day of the week from Start_Date and add it to the Day_of_Week column
        df['Day_of_Week'] = pd.to_datetime(df['Start_Date'], errors='coerce').dt.day_name()

        # Reorder the columns
        df = df[['URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week', 'Start_Date', 
                 'End_Date', 'Start_Time', 'End_Time', 'Price', 'Location', 'Description']]

        # Sort the DataFrame by Start_Date and Start_Time
        df = df.sort_values(by=['Start_Date', 'Start_Time']).reset_index(drop=True)

        return df
    

    def write_events_to_db(self, df, url, source, keywords):
        """
        Writes event data to the 'events' table in the database.

        Args:
            df (pandas.DataFrame): DataFrame containing events data.
            url (str): URL from which the events data was sourced.

        Notes:
            - The 'event_id' column is auto-generated and should not be included in the DataFrame.
            - Ensures that only relevant columns are written to the database.
        """
        # Save the events data to a CSV file for debugging purposes
        df.to_csv('output/events.csv', index=False)

        # Need to check if it is from google calendar or from the LLM.
        if 'calendar' in url:
            
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

        # Update the 'url' column with the source URL if there is no URL in the DataFrame
        df['url'] = df['url'].fillna('').replace('', url)

        # Ensure 'start_date' and 'start_date' are in datetime.date format
        for col in ['start_date', 'end_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

        # Ensure 'start_time' and 'end_time' are in datetime.time format
        for col in ['start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

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

        # Save the cleaned events data to a CSV file for debugging purposes
        cleaned_df.to_csv('output/cleaned_events.csv', index=False)

        # Log the number of events to be written
        logging.info(f"write_events_to_db: Number of events to write: {len(df)}")

        cleaned_df.to_csv('output/cleaned_events.csv', index=False)

        # Write the cleaned events data to the 'events' table
        cleaned_df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        logging.info("write_events_to_db: Events data written to the 'events' table.")

        return None
    
    
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

        Args:
            address_dict (dict): Dictionary containing address details.

        Returns:
            int or None: The address_id if successful, else None.
        """
        try:
            # Define the INSERT query with ON CONFLICT DO UPDATE and RETURNING address_id
            insert_query = """
                INSERT INTO address (
                    full_address, street_number, street_name, street_type, 
                    postal_box, city, province_or_state, postal_code, country_id
                ) VALUES (
                    :full_address, :street_number, :street_name, :street_type,
                    :postal_box, :city, :province_or_state, :postal_code, :country_id
                )
                ON CONFLICT (full_address) DO UPDATE SET
                    street_number = EXCLUDED.street_number,
                    street_name = EXCLUDED.street_name,
                    street_type = EXCLUDED.street_type,
                    postal_box = EXCLUDED.postal_box,
                    city = EXCLUDED.city,
                    province_or_state = EXCLUDED.province_or_state,
                    postal_code = EXCLUDED.postal_code,
                    country_id = EXCLUDED.country_id
                RETURNING address_id;
            """

            # Execute the INSERT query
            insert_result = self.execute_query((insert_query), address_dict)

            if insert_result:
                # Insert succeeded or update occurred; retrieve the address_id
                row = insert_result[0]
                address_id = row[0]  # address_id is the first column in the result

                # *** KEY CHANGE: Convert to Python int ***
                if isinstance(address_id, np.int64):
                    address_id = int(address_id)

                logging.info(f"get_address_id: Inserted or updated address_id {address_id} for address '{address_dict['full_address']}'.")
                return address_id

            logging.error(f"get_address_id: Failed to insert or update address_id for '{address_dict['full_address']}'.")
            return None

        except SQLAlchemyError as e:
            logging.error(f"get_address_id: Database error: {e}")
            return None


    def clean_up_address(self, events_df):
        """
        Cleans up and standardizes address data from the 'events' table.
        It parses the 'location' field, inserts unique addresses into the 'address' table,
        and updates the 'events' table with the corresponding address_id.
        """
        logging.info(f"def clean_up_address(): events_df.shape at beginning of function is: {events_df.shape}")
        # Iterate over each row in the DataFrame
        for index, row in events_df.iterrows():
            location = row['location']
            location = str(location).strip() if pd.notna(location) else None

            if location:
                # Extract postal codes using regex
                postal_code = re.search(r'[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d', location)
                postal_code = postal_code.group() if postal_code else None
                postal_code = postal_code.replace(' ', '') if postal_code else None
                logging.info(f"clean_up_address: Extracted postal code '{postal_code}' from location '{location}'.")

                # Need to figure out WHICH address is the right one. Addresses share postal codes
                # Extract all numbers from location
                numbers = re.findall(r'\d+', location)

                # We can get the correct components of the address from the postal code
                if postal_code:
                    sql = ("""
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
                    )
                    result_df = pd.read_sql((sql),
                        self.address_db_engine,
                        params=(postal_code,)) # params is now a list
                    
                    # If the civic_no is already in the address, then it will match and 
                    # cause a break on the correct address. This will set the idx of the row. 
                    # If not, we will be using that last address, which hopefully is not a problem:(
                    if result_df.shape[0] > 0:
                        for idx, row in result_df.iterrows():
                            if row.civic_no in numbers:
                                break
                            else:
                                pass

                        # Create a properly formatted address string
                        location = (
                            f"{result_df.civic_no.values[idx]} "
                            f"{result_df.civic_no_suffix.values[idx]} "
                            f"{result_df.official_street_name.values[idx]} "
                            f"{result_df.official_street_type.values[idx]} "
                            f"{result_df.official_street_dir.values[idx]}, "
                            f"{result_df.mail_mun_name.values[idx]}, "
                            f"{result_df.mail_prov_abvn.values[idx]}, "
                            f"{result_df.mail_postal_code.values[idx]}, "
                            f"CA"
                        )
                        # Remove any 'None' values
                        location = location.replace('None', '')

                        # We may have double spaces, so we will replace them with single spaces
                        location = location.replace('  ', ' ')

                        # We may have a space before a comma, so we will remove it
                        location = location.replace(' ,', ',')

                        events_df.loc[index, 'location'] = location
                        logging.info(f"def clean_up_address(): events_df.shape after updating location is: {events_df.shape}")

                        # Write the address to the address table
                        address_dict = {
                            'full_address': location,
                            'street_number': str(result_df.civic_no.values[idx]),
                            'street_name': result_df.official_street_name.values[idx],
                            'street_type': result_df.official_street_type.values[idx],
                            'postal_box': None,
                            'city': result_df.mail_mun_name.values[idx],
                            'province_or_state': result_df.mail_prov_abvn.values[idx],
                            'postal_code': result_df.mail_postal_code.values[idx],
                            'country_id': 'CA'
                        }
                        # Write this dictionary to the address table
                        address_id = self.get_address_id(address_dict)
                        events_df.loc[index, 'address_id'] = address_id
                    else:
                        logging.info(f"def clean_up_address(): No address found for postal code: {postal_code}.")
                else:
                    logging.info(f"def clean_up_address(): No postal code found for location: '{location}'.")
            else:
                logging.info(f"def clean_up_address(): No location provided for row: {index}.")

        logging.info(f"def clean_up_address(): events_df.shape at end of function is: {events_df.shape}")
        return events_df
    

    def dedup(self):
        """
        Removes duplicates from the 'events' and 'urls' tables in the database.

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
                  AND e1.event_name = e2.event_name
                  AND e1.start_date = e2.start_date;
            """
            self.execute_query(dedup_events_query)
            logging.info("dedup: Deduplicated 'events' table successfully.")

            # Deduplicate 'urls' table based on 'link' using ctid for row identification
            dedup_urls_query = """
                DELETE FROM urls a
                USING urls b
                WHERE a.ctid < b.ctid
                AND a.link = b.link;
            """
            self.execute_query(dedup_urls_query)
            logging.info("dedup: Deduplicated 'urls' table successfully.")

        except Exception as e:
            logging.error("dedup: Failed to deduplicate tables: %s", e)


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
            logging.info("delete_old_events: Deleted events older than %d days.", days)
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
        AND address_id IS NULL;
        """
        params = {
            'source': '',
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }

        self.execute_query(delete_query_1, params)
        logging.info("delete_likely_dud_events: Deleted events with empty source, dance_style, and url, and no address_id.")

        # 2. Delete events outside of British Columbia (BC)
        delete_query_2 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE province_or_state IS NOT NULL
            AND province_or_state != :province_or_state
        );
        """
        params = {
            'province_or_state': 'BC'
            }
        
        self.execute_query(delete_query_2, params)
        logging.info("delete_likely_dud_events: Deleted events outside of British Columbia (BC).")

        # 3. Delete events that are not in Canada
        delete_query_3 = """
        DELETE FROM events
        WHERE address_id IN (
        SELECT address_id
        FROM address
        WHERE country_id IS NOT NULL
            AND country_id != :country_id
        );
        """
        params = {
            'country_id': 'CA'
            }
        
        self.execute_query(delete_query_3, params)
        logging.info("delete_likely_dud_events: Deleted events that are not in Canada (CA).")

        # 4. Delete rows in events where dance_style and url are == '' AND event_type == 'other' AND location IS NULL and description IS NULL
        delete_query_4 = """
        DELETE FROM events
        WHERE dance_style = :dance_style
            AND url = :url
            AND event_type = :event_type
            AND location IS NULL
            AND description IS NULL;
        """
        params = {
            'dance_style': '',
            'url': '',
            'event_type': 'other'
            }
        
        self.execute_query(delete_query_4, params)
        logging.info(
            "def delete_likely_dud_events(): Deleted events with empty "
            "dance_style, url, event_type 'other', and null location "
            "and description."
        )
        

    def delete_event_and_update_url(self, url, event_name, start_date):
        """
        Deletes an event from the 'events' table and the corresponding URL from the 'urls' table.

        Args:
            url (str): The URL of the event to be deleted.
            event_name (str): The name of the event to be deleted.
            start_date (str): The start date of the event to be deleted.
        """
        try:
            logging.info("delete_event_and_update_url: Deleting event with URL: %s, Event Name: %s, Start Date: %s", url, event_name, start_date)

            # Delete the event from 'events' table
            delete_event_query = """
                DELETE FROM events
                WHERE Name_of_the_Event = :event_name
                  AND Start_Date = :start_date;
            """
            params = {'event_name': event_name, 'start_date': start_date}
            self.execute_query(delete_event_query, params)
            logging.info("delete_event_and_update_url: Deleted event from 'events' table.")

            # Update the corresponding URL from 'urls' table
            update_other_link = 'No'
            relevant = False
            increment_crawl_try = 1
            db_handler.update_url(url, update_other_link, relevant, increment_crawl_try)
            logging.info("delete_event_and_update_url: Deleted URL from 'urls' table.")

        except Exception as e:
            logging.error("delete_event_and_update_url: Failed to delete event and URL: %s", e)


    def delete_events_with_nulls(self):
        """
        Deletes events with start_date and start_time being null in the 'events' table.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE start_date IS NULL AND start_time IS NULL;
            """
            self.execute_query(delete_query)
            logging.info("def delete_events_with_nulls(): Both start_date and start_time being null deleted successfully.")
        except Exception as e:
            logging.error("def delete_events_with_nulls(): Failed to delete events with start_date and start_time being null: %s", e)


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


    def multiple_db_inserts(self, values):
        """
        Inserts or updates multiple records in the events table.

        Parameters:
        values (list of dict): List of dictionaries where each dict represents a row to insert/update.
        """
        if not values:
            logging.info("multiple_db_inserts(): No values to insert or update.")
            return

        try:
            # Get reference to the 'events' table from metadata
            events_table = Table("events", self.metadata, autoload_with=self.conn)

            with self.conn.begin() as conn:
                for row in values:
                    stmt = insert(events_table).values(row)
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["event_id"],  # Primary key column
                        set_={col: stmt.excluded[col] for col in row.keys() if col != "event_id"}
                    )
                    conn.execute(stmt)

            logging.info(f"multiple_db_inserts(): Successfully inserted/updated {len(values)} rows.")

        except Exception as e:
            logging.error(f"multiple_db_inserts(): Error inserting/updating records - {e}")


    def driver(self):
        """
        Main driver function to perform database operations.
        """
        self.create_tables()
        self.dedup()
        self.delete_old_events()
        self.delete_events_with_nulls()
        self.delete_likely_dud_events()
        self.fuzzy_duplicates()

        # Close the database connection
        self.conn.dispose()  # Using dispose() for SQLAlchemy Engine

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
    logging.info("db.py starting...")

    start_time = datetime.now()
    logging.info("\n\nMain: Started the process at %s", start_time)

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)

    # Create tables
    db_handler.create_tables()

    # Perform deduplication and delete old events
    db_handler.driver()

    end_time = datetime.now()
    logging.info("Main: Finished the process at %s", end_time)
    logging.info("Main: Total time taken: %s", end_time - start_time)
