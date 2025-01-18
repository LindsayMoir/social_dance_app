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
    - pyap: For parsing addresses from location strings.
    - sqlalchemy: For database connection, query execution, and ORM support.
    - yaml: For loading configuration from YAML files.

Note:
    The module assumes a valid YAML configuration file containing database 
    credentials, logging configurations, and application-specific settings. 
    Logging is configured in the main block to record operations and any errors 
    encountered during database interactions.
"""

from datetime import datetime
from fuzzywuzzy import fuzz
import logging
import numpy as np
import os
import pandas as pd
import pyap
import re  # Added missing import
from sqlalchemy import create_engine, update, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
import yaml



class DatabaseHandler():
    def __init__(self, config):
        """
        Initializes the DatabaseHandler with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters for the database connection.
        """
        self.config = config

        self.conn = self.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        logging.info("def __init__(): Database connection established.")


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
                f"postgresql://{self.config['database']['user']}:" 
                f"{self.config['database']['password']}@"
                f"{self.config['database']['host']}/"
                f"{self.config['database']['name']}"
            )

            # Create and return the SQLAlchemy engine
            engine = create_engine(connection_string, isolation_level="AUTOCOMMIT")
            return engine

        except Exception as e:
            logging.error("DatabaseHandler: Database connection failed: %s", e)
            return None
        

    def create_tables(self):
        """
        Creates the 'urls', 'events', 'address', and 'fb_urls' tables in the database if they do not already exist.
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
                links TEXT PRIMARY KEY,
                time_stamps TIMESTAMP,
                org_names TEXT,
                keywords TEXT,
                other_links TEXT,
                relevant BOOLEAN,
                crawl_trys INTEGER
            )
        """
        self.execute_query(urls_table_query)
        logging.info("create_tables: 'urls' table created or already exists.")

        # Create the 'events' table
        events_table_query = """
            CREATE TABLE IF NOT EXISTS events (
                event_id SERIAL PRIMARY KEY,
                org_name TEXT,
                dance_style TEXT,
                url TEXT,
                event_type TEXT,
                event_name TEXT,
                day_of_week TEXT,
                start_date DATE,
                end_date DATE,
                start_time TIME,
                end_time TIME,
                price NUMERIC,
                location TEXT,
                address_id INTEGER,
                description TEXT,
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
                floor TEXT,
                postal_box TEXT,
                city TEXT,
                province_or_state TEXT,
                postal_code TEXT,
                country_id TEXT
            )
        """
        self.execute_query(address_table_query)
        logging.info("create_tables: 'address' table created or already exists.")

        # Create the 'organizations' table
        organizations_table_query = """
            CREATE TABLE IF NOT EXISTS organizations (
                org_id SERIAL PRIMARY KEY,
                org_name TEXT,
                web_url TEXT,
                fb_url TEXT,
                ig_url TEXT,
                phone TEXT CHECK (phone ~ '^\(\d{3}\) \d{3}-\d{4}$'),
                email TEXT,
                address_id INTEGER
            )
        """
        self.execute_query(organizations_table_query)
        logging.info("create_tables: 'organizations' table created or already exists.")

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
            result: The result of the executed query, if any.
        """
        if self.conn is None:
            logging.error("execute_query: No database connection available.")
            return None

        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(query), params or {})
                connection.commit()
                return result
        except SQLAlchemyError as e:
            logging.error("def execute_query(): {query}\nQuery execution failed: %s", e)
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


    def update_url(self, url, update_other_links, relevant, increment_crawl_trys):
        """
        Updates an entry in the 'urls' table with the provided URL and other details.

        Args:
            url (str): The URL to be updated.
            update_other_links (str): The new value for 'other_links'. If 'No', it won't be updated.
            relevant (bool): The new value for 'relevant'.
            increment_crawl_trys (int): The number to increment 'crawl_trys' by.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        # See if url is in the urls table
        select_query = """
            SELECT * FROM urls
            WHERE links = :url
        """
        select_params = {'url': url}
        result = self.execute_query(select_query, select_params)

        if result:
            # Prepare the update query with conditional 'other_links'
            if update_other_links != 'No':
                update_query = """
                    UPDATE urls
                    SET 
                        time_stamps = :current_time,
                        other_links = :other_links,
                        crawl_trys = crawl_trys + :increment,
                        relevant = :relevant
                    WHERE links = :url
                """
                update_params = {
                    'current_time': datetime.now(),
                    'other_links': update_other_links,
                    'increment': increment_crawl_trys,
                    'relevant': relevant,
                    'url': url
                }
            else:
                update_query = """
                    UPDATE urls
                    SET 
                        time_stamps = :current_time,
                        crawl_trys = crawl_trys + :increment,
                        relevant = :relevant
                    WHERE links = :url
                """
                update_params = {
                    'current_time': datetime.now(),
                    'increment': increment_crawl_trys,
                    'relevant': relevant,
                    'url': url
                }

            # Execute the update
            update_result = self.execute_query(update_query, update_params)
            if update_result and update_result.rowcount > 0:
                logging.info("update_url: Updated URL '%s' successfully.", url)
                return True
            else:
                logging.info("update_url: Failed to update URL '%s'.", url)
                return False
        else:
            logging.info("update_url: URL '%s' not found for update.", url)
            return False
        

    def write_url_to_db(self, org_names, keywords, url, other_links, relevant, increment_crawl_trys):
        """
        Writes or updates a URL in the 'urls' table.

        Args:
            keywords (str): Keywords related to the URL.
            url (str): The URL to be written or updated.
            other_links (str): Other links associated with the URL.
            relevant (bool): Indicates if the URL is relevant.
            increment_crawl_trys (int): The number of times the URL has been crawled.
        """
        if self.update_url(url, other_links, relevant, increment_crawl_trys + 1):
            logging.info("write_url_to_db: URL '%s' updated in the 'urls' table.", url)
        else:
            logging.info("write_url_to_db: Inserting new URL '%s' into the 'urls' table.", url)
            new_df = pd.DataFrame({
                "time_stamps": [datetime.now()],
                "org_names": [org_names],  
                "keywords": [keywords],
                "links": [url],
                "other_links": [other_links],
                "relevant": [relevant],
                "crawl_trys": [increment_crawl_trys]
            })
            try:
                new_df.to_sql('urls', self.conn, if_exists='append', index=False, method='multi')
                logging.info("write_url_to_db: URL '%s' inserted into the 'urls' table.", url)
            except Exception as e:
                logging.error("write_url_to_db: Failed to insert URL '%s': %s", url, e)


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
        df['Price'] = pd.to_numeric(df['description'].str.extract(r'\$(\d{1,5})')[0], errors='coerce')

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
    

    def write_events_to_db(self, df, url, org_name, keywords):
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
            df['org_name'] = org_name
            if isinstance(keywords, list):
                keywords = ', '.join(keywords)
            df['dance_style'] = keywords

        # Ensure 'start_date' and 'start_date' are in datetime.date format
        for col in ['start_date', 'end_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

        # Ensure 'start_time' and 'end_time' are in datetime.time format
        for col in ['start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

        # Clean and convert the 'price' column to numeric format
        if 'price' in df.columns and not df['price'].isnull().all():
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        else:
            logging.warning("write_events_to_db: 'price' column is missing or empty. Filling with NaN.")
            df['price'] = float('nan')

        # Add a 'time_stamp' column with the current timestamp
        df['time_stamp'] = datetime.now()

        # Put the url in if it is NOT social media or calendar url
        if any(social in url for social in ['facebook', 'instagram', 'calendar']):
            pass
        else:
            df['url'] = url

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

        # Write the cleaned events data to the 'events' table
        cleaned_df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        logging.info("write_events_to_db: Events data written to the 'events' table.")

        return None
    

    def get_address_id(self, address_dict):
        """
        Retrieves the address_id for a given address. If the address does not exist,
        it inserts the address into the 'address' table and returns the new address_id.
        
        Parameters:
            address_dict (dict): Dictionary containing address components.
        
        Returns:
            int: The address_id corresponding to the address.
        """
        select_query = "SELECT address_id FROM address WHERE full_address = :full_address"
        select_params = {'full_address': address_dict['full_address']}
        
        try:
            # Attempt to retrieve the address_id if it exists
            result = self.execute_query(select_query, select_params)
            if result:
                row = result.fetchone()
                if row:
                    return row[0]

            # If not found, insert the new address
            columns = ', '.join(address_dict.keys())
            placeholders = ', '.join([f":{k}" for k in address_dict.keys()])
            insert_query = f"""
                INSERT INTO address ({columns})
                VALUES ({placeholders})
                RETURNING address_id
            """
            
            logging.debug(f"Insert Query: {insert_query}")
            logging.debug(f"Insert Params: {address_dict}")
            
            # Execute the insert query
            insert_result = self.execute_query(insert_query, address_dict)
            if insert_result:
                row = insert_result.fetchone()
                if row:
                    address_id = row[0]
                    logging.info(f"get_address_id: Inserted new address_id {address_id} for address '{address_dict['full_address']}'.")
                    return address_id

            logging.error(f"get_address_id: Failed to insert or retrieve address_id for '{address_dict['full_address']}'.")
            return None

        except Exception as e:
            logging.error(f"get_address_id: Failed to retrieve or insert address '{address_dict['full_address']}': {e}")
            return None
        

    def clean_up_address(self, events_df):
        """
        Cleans up and standardizes address data from the 'events' table.
        It parses the 'location' field, inserts unique addresses into the 'address' table,
        and updates the 'events' table with the corresponding address_id.
        """
        # Iterate over each row in the DataFrame
        for index, row in events_df.iterrows():
            location = row.get('location') or ''
            location = str(location).strip()
            if not location:
                logging.warning(f"clean_up_address: Skipping row {index} due to empty 'location'.")
                continue  # Skip if 'location' is empty

            parsed_addresses = pyap.parse(location, country='CA')
            if not parsed_addresses:
                logging.warning(f"clean_up_address: No address found in 'location' for row {index}.")
                continue

            address = parsed_addresses[0]
            logging.debug(f"Row {index} Address Attributes: {address.__dict__}")
            address_dict = {
                'full_address': address.full_address or '',
                'street_number': getattr(address, 'street_number', ''),
                'street_name': getattr(address, 'street_name', ''),
                'street_type': getattr(address, 'street_type', ''),
                'floor': getattr(address, 'floor', ''),
                'postal_box': getattr(address, 'postal_box', ''),
                'city': getattr(address, 'city', ''),
                'province_or_state': getattr(address, 'region1', ''),
                'postal_code': getattr(address, 'postal_code', ''),
                'country_id': getattr(address, 'country_id', 'CA')
            }
            logging.debug(f"Row {index} Address Dict: {address_dict}")
            address_id = self.get_address_id(address_dict)
            if address_id is None:
                logging.error(f"clean_up_address: Failed to obtain address_id for row {index}.")
                continue

            events_df.at[index, 'address_id'] = address_id

        return events_df  # Moved outside of the loop
    

    def dedup(self):
        """
        Removes duplicates from the 'events' and 'urls' tables in the database.

        For the 'events' table, duplicates are identified based on the combination of
        'Name_of_the_Event' and 'Start_Date'. Only the latest entry is kept.

        For the 'urls' table, duplicates are identified based on the 'links' column.
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

            # Deduplicate 'urls' table based on 'links' using ctid for row identification
            dedup_urls_query = """
                DELETE FROM urls a
                USING urls b
                WHERE a.ctid < b.ctid
                AND a.links = b.links;
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


    def delete_likley_dud_events(self):
        """
        1. If the event org_name, dance_style, and url == '', delete the event, UNLESS it has an address_id then keep it.
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
        try:
            # 1. Delete events where org_name, dance_style, and url are empty, unless they have an address_id
            delete_query_1 = """
            DELETE FROM events
            WHERE org_name = ''
              AND dance_style = ''
              AND url = ''
              AND address_id IS NULL;
            """
            self.execute_query(delete_query_1)
            logging.info("delete_likley_dud_events: Deleted events with empty org_name, dance_style, and url, and no address_id.")

            # 2. Delete events outside of British Columbia (BC)
            delete_query_2 = """
            DELETE FROM events
            WHERE address_id IN (
            SELECT address_id
            FROM address
            WHERE province_or_state IS NOT NULL
              AND province_or_state != 'BC'
            );
            """
            self.execute_query(delete_query_2)
            logging.info("delete_likley_dud_events: Deleted events outside of British Columbia (BC).")

            # 3. Delete events that are not in Canada
            delete_query_3 = """
            DELETE FROM events
            WHERE address_id IN (
            SELECT address_id
            FROM address
            WHERE country_id IS NOT NULL
              AND country_id != 'CA'
            );
            """
            self.execute_query(delete_query_3)
            logging.info("delete_likley_dud_events: Deleted events that are not in Canada (CA).")

            # 4. Delete rows in events where dance_style and url are == '' AND event_type == 'other' AND location IS NULL and description IS NULL
            delete_query = """
            DELETE FROM events
            WHERE dance_style = ''
              AND url = ''
              AND event_type = 'other'
              AND location IS NULL
              AND description IS NULL;
            """
            self.execute_query(delete_query)
            logging.info(
                "delete_likley_dud_events: Deleted events with empty "
                "dance_style, url, event_type 'other', and null location "
                "and description."
            )
        except Exception as e:
            logging.error(
                "delete_likley_dud_events: Failed to delete events with "
                "empty dance_style, url, event_type 'other', and null "
                "location and description: %s", e
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
            update_other_links = 'No'
            relevant = False
            increment_crawl_trys = 1
            db_handler.update_url(url, update_other_links, relevant, increment_crawl_trys)
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

        try:
            delete_query = """
            DELETE FROM events
            WHERE 
                org_name = ''
                AND dance_style = ''
                AND url = ''
                AND address_id IS NULL;
            """
            self.execute_query(delete_query)
            logging.info("def delete_events_with_nulls(): org_name, dance_style, url, (all NULL) and address_id IS NULL deleted successfully.")
        except Exception as e:
            logging.error("def delete_events_with_nulls(): Failed to delete events with start_date and start_time being null: %s", e)

    
    def driver(self):
        """
        Main driver function to perform database operations.
        """
        self.create_tables()
        self.dedup()
        self.delete_old_events()
        self.delete_events_with_nulls()
        self.delete_likley_dud_events()
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

    start_time = datetime.now()
    logging.info("Main: Started the process at %s", start_time)

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)

    # Perform deduplication and delete old events
    db_handler.driver()

    end_time = datetime.now()
    logging.info("Main: Finished the process at %s", end_time)
    logging.info("Main: Total time taken: %s", end_time - start_time)
