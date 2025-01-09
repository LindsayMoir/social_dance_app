import logging
import os
from datetime import datetime
import pandas as pd
import pyap
import re  # Added missing import
from sqlalchemy import create_engine, update, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
import yaml

class DatabaseHandler:
    def __init__(self, config):
        """
        Initializes the DatabaseHandler with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters for the database connection.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("\n\nDatabaseHandler: Config initialized.")

        self.conn = self.get_db_connection()
        if self.conn is None:
            raise ConnectionError("DatabaseHandler: Failed to establish a database connection.")
        self.logger.info("DatabaseHandler: Database connection established.")

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
            engine = create_engine(connection_string)
            return engine

        except Exception as e:
            self.logger.error("DatabaseHandler: Database connection failed: %s", e)
            return None

    def create_tables(self):
        """
        Creates the 'urls', 'events', 'address', and 'fb_urls' tables in the database if they do not already exist.
        If config['testing']['drop_tables'] is True, it will drop existing tables before creation.
        """
        try:
            # Check if we need to drop tables as per configuration
            if self.config['testing']['drop_tables'] == True:
                drop_queries = [
                    "DROP TABLE IF EXISTS fb_urls CASCADE;",
                    "DROP TABLE IF EXISTS address CASCADE;",
                    "DROP TABLE IF EXISTS events CASCADE;",
                    "DROP TABLE IF EXISTS urls CASCADE;"
                ]
            elif self.config['testing']['drop_tables'] == 'events':
                drop_queries = [
                    "DROP TABLE IF EXISTS events CASCADE;"
                ]
                for query in drop_queries:
                    self.execute_query(query)
                self.logger.info(f"create_tables: Existing tables dropped as per configuration value of '{self.config['testing']['drop_tables']}'.")
            else:
                # Don't drop any tables
                pass

            # Create the 'urls' table
            urls_table_query = """
                CREATE TABLE IF NOT EXISTS urls (
                    url_id SERIAL PRIMARY KEY,
                    time_stamps TIMESTAMP,
                    org_names TEXT,
                    keywords TEXT,
                    links TEXT UNIQUE,
                    other_links TEXT,
                    relevant BOOLEAN,
                    crawl_trys INTEGER
                )
            """
            self.execute_query(urls_table_query)
            self.logger.info("create_tables: 'urls' table created or already exists.")

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
            self.logger.info("create_tables: 'events' table created or already exists.")

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
            self.logger.info("create_tables: 'address' table created or already exists.")

            # Create the 'fb_urls' table
            fb_urls_table_query = """
                CREATE TABLE IF NOT EXISTS fb_urls (
                    url TEXT PRIMARY KEY,
                    time_stamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            self.execute_query(fb_urls_table_query)
            self.logger.info("create_tables: 'fb_urls' table created or already exists.")

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
            self.logger.info("create_tables: 'organizations' table created or already exists.")

        except Exception as e:
            self.logger.error("create_tables: Failed to create tables: %s", e)

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
            self.logger.error("execute_query: No database connection available.")
            return None

        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(query), params or {})
                connection.commit()
                self.logger.info("execute_query: Query executed successfully.")
                return result
        except SQLAlchemyError as e:
            self.logger.error("execute_query: Query execution failed: %s", e)
            return None
    
    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.conn:
            try:
                self.conn.dispose()
                self.logger.info("close_connection: Database connection closed successfully.")
            except Exception as e:
                self.logger.error("close_connection: Failed to close database connection: %s", e)
        else:
            self.logger.warning("close_connection: No database connection to close.")

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
        try:
            # Prepare the update query with conditional 'other_links'
            if update_other_links != 'No':
                query = """
                    UPDATE urls
                    SET 
                        time_stamps = :current_time,
                        other_links = :other_links,
                        crawl_trys = crawl_trys + :increment,
                        relevant = :relevant
                    WHERE links = :url
                """
                params = {
                    'current_time': datetime.now(),
                    'other_links': update_other_links,
                    'increment': increment_crawl_trys,
                    'relevant': relevant,
                    'url': url
                }
            else:
                query = """
                    UPDATE urls
                    SET 
                        time_stamps = :current_time,
                        crawl_trys = crawl_trys + :increment,
                        relevant = :relevant
                    WHERE links = :url
                """
                params = {
                    'current_time': datetime.now(),
                    'increment': increment_crawl_trys,
                    'relevant': relevant,
                    'url': url
                }

            result = self.execute_query(query, params)
            if result and result.rowcount > 0:
                self.logger.info("update_url: Updated URL '%s' successfully.", url)
                return True
            else:
                self.logger.info("update_url: URL '%s' not found for update.", url)
                return False

        except Exception as e:
            self.logger.error("update_url: Failed to update URL '%s': %s", url, e)
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
            self.logger.info("write_url_to_db: URL '%s' updated in the 'urls' table.", url)
        else:
            self.logger.info("write_url_to_db: Inserting new URL '%s' into the 'urls' table.", url)
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
                self.logger.info("write_url_to_db: URL '%s' inserted into the 'urls' table.", url)
            except Exception as e:
                self.logger.error("write_url_to_db: Failed to insert URL '%s': %s", url, e)

    def write_url_to_fb_table(self, url):
        """
        Writes a URL to the 'fb_urls' table in the database.

        Args:
            url (str): The URL to be written to the 'fb_urls' table.
        """
        try:
            query = """
            INSERT INTO fb_urls (url, time_stamp) 
            VALUES (:url, :time_stamp)
            ON CONFLICT (url) 
            DO UPDATE SET time_stamp = EXCLUDED.time_stamp;
            """
            params = {'url': url, 'time_stamp': datetime.now()}
            self.execute_query(query, params)
            self.logger.info("write_url_to_fb_table: URL '%s' written to 'fb_urls' table.", url)
        except Exception as e:
            self.logger.error("write_url_to_fb_table: Failed to write URL '%s': %s", url, e)

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
            self.logger.warning("write_events_to_db: 'price' column is missing or empty. Filling with NaN.")
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
        cleaned_df = cleaned_df.dropna(subset=['start_date', 'end_date', 'start_time', 
                                               'end_time', 'location', 'description'], how='all')

        # Save the cleaned events data to a CSV file for debugging purposes
        cleaned_df.to_csv('output/cleaned_events.csv', index=False)

        # Log the number of events to be written
        logging.info(f"write_events_to_db: Number of events to write: {len(df)}")

        # Write the cleaned events data to the 'events' table
        cleaned_df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        self.logger.info("write_events_to_db: Events data written to the 'events' table.")

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
        params = {'full_address': address_dict['full_address']}
        try:
            with self.conn.connect() as connection:
                result = connection.execute(text(select_query), params).fetchone()
                if result:
                    return result[0]
                else:
                    # Insert the new address and retrieve the generated address_id
                    columns = ', '.join(address_dict.keys())
                    placeholders = ', '.join([f":{k}" for k in address_dict.keys()])
                    insert_query = f"INSERT INTO address ({columns}) VALUES ({placeholders}) RETURNING address_id"

                    logging.debug(f"Insert Query: {insert_query}")
                    logging.debug(f"Insert Params: {address_dict}")

                    result = connection.execute(text(insert_query), address_dict).fetchone()
                    address_id = result[0]
                    connection.commit()
                    logging.info(f"get_address_id: Inserted new address_id {address_id} for address '{address_dict['full_address']}'.")
                    return address_id
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
            self.logger.info("dedup: Deduplicated 'events' table successfully.")

            # Deduplicate 'urls' table based on 'links'
            dedup_urls_query = """
                DELETE FROM urls u1
                USING urls u2
                WHERE u1.url_id < u2.url_id
                  AND u1.links = u2.links;
            """
            self.execute_query(dedup_urls_query)
            self.logger.info("dedup: Deduplicated 'urls' table successfully.")

        except Exception as e:
            self.logger.error("dedup: Failed to deduplicate tables: %s", e)

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
            self.logger.info("delete_old_events: Deleted events older than %d days.", days)
        except Exception as e:
            self.logger.error("delete_old_events: Failed to delete old events: %s", e)

    def delete_event_and_url(self, url, event_name, start_date):
        """
        Deletes an event from the 'events' table and the corresponding URL from the 'urls' table.

        Args:
            url (str): The URL of the event to be deleted.
            event_name (str): The name of the event to be deleted.
            start_date (str): The start date of the event to be deleted.
        """
        try:
            self.logger.info("delete_event_and_url: Deleting event with URL: %s, Event Name: %s, Start Date: %s", url, event_name, start_date)

            # Delete the event from 'events' table
            delete_event_query = """
                DELETE FROM events
                WHERE Name_of_the_Event = :event_name
                  AND Start_Date = :start_date;
            """
            params = {'event_name': event_name, 'start_date': start_date}
            self.execute_query(delete_event_query, params)
            self.logger.info("delete_event_and_url: Deleted event from 'events' table.")

            # Delete the corresponding URL from 'urls' table
            delete_url_query = "DELETE FROM urls WHERE links = :url;"
            params = {'url': url}
            self.execute_query(delete_url_query, params)
            self.logger.info("delete_event_and_url: Deleted URL from 'urls' table.")

        except Exception as e:
            self.logger.error("delete_event_and_url: Failed to delete event and URL: %s", e)

if __name__ == "__main__":
    start_time = datetime.now()

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

    # Initialize DatabaseHandler
    db_handler = DatabaseHandler(config)
    db_handler.create_tables()

    # Perform deduplication and delete old events
    db_handler.dedup()
    db_handler.delete_old_events()

    # Close the database connection
    db_handler.conn.dispose()  # Using dispose() for SQLAlchemy Engine

    end_time = datetime.now()
    logging.info("Main: Finished the process at %s", end_time)
    logging.info("Main: Total time taken: %s", end_time - start_time)
