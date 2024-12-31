import logging
import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, update, MetaData
import yaml


class DatabaseHandler:
    def __init__(self, config):
        """
        Initializes the DatabaseHandler with the given configuration.

        Args:
            config (dict): A dictionary containing configuration parameters for the database connection.

        Initializes the database connection and logs the initialization and connection status.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)  # Use a class-specific logger
        self.logger.info("class DatabaseHandler: config initialized.")

        self.conn = self.get_db_connection()
        if self.conn is None:
            raise ConnectionError("class DatabaseHandler: Failed to establish a database connection.")
        self.logger.info("class DatabaseHandler: Database connection established.")


    def get_db_connection(self):
        """
        Establishes and returns a SQLAlchemy engine for connecting to the PostgreSQL database.
        This method reads the database connection parameters (user, password, host, and database name)
        from the configuration stored in `self.config` and constructs a connection string. It then
        creates and returns a SQLAlchemy engine using this connection string.
        Returns:
            sqlalchemy.engine.Engine: A SQLAlchemy engine for the PostgreSQL database connection.
            None: If the connection fails, it logs an error and returns None.
        Raises:
            Exception: If there is an error in creating the SQLAlchemy engine, it logs the error.
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
            conn = create_engine(connection_string)
            return conn
        
        except Exception as e:
            self.logger.error("Database connection failed: %s", e)
            return None
        

    def url_in_table(self, url):
        """
        Parameters:
        url (str): The URL to check in the 'urls' table.

        Returns:
        bool: True if the URL exists in the table, False otherwise.

        Logs:
        - Info if the URL exists or does not exist in the 'urls' table.
        - Error if there is an exception during the database query.
        """
        query = 'SELECT * FROM urls WHERE links = %s'
        params = (url,)
        try:
            df = pd.read_sql(query, self.conn, params=params)
            if df.shape[0] > 0:
                self.logger.info(f"def url_in_table(): URL {url} exists in the 'urls' table.")
                return True
            else:
                self.logger.info(f"def url_in_table(): URL {url} does not exist in the 'urls' table.")
                return False
        except Exception as e:
            self.logger.error(f"def url_in_table(): Failed to check URL: {e}")
            return False


    def update_url(self, url, update_other_links, relevant, increment_crawl_trys):
        """
        Update an entry in the 'urls' table with the provided URL and other details.

        Parameters:
        url (str): The URL to be updated in the database.
        update_other_links (str): The new value for the 'other_links' column. If 'No', the column will not be updated.
        relevant (bool): The new value for the 'relevant' column.
        increment_crawl_trys (int): The number to increment the 'crawl_trys' column by.

        Raises:
        ConnectionError: If the database connection is not available.

        Logs:
        Logs an info message indicating the URL that was updated.
        """
        try:
            metadata = MetaData()
            metadata.reflect(bind=self.conn)
            table = metadata.tables['urls']

            query = (
                update(table)
                .where(table.c.links == url)
                .values(
                    time_stamps=datetime.now(),
                    other_links=update_other_links if update_other_links != 'No' else table.c.other_links,
                    crawl_trys=table.c.crawl_trys + increment_crawl_trys,
                    relevant=relevant
                )
            )

            with self.conn.begin() as connection:
                connection.execute(query)
                self.logger.info(f"def update_url(): Updated URL: {url}")
                return True
            
        except Exception as e:
            self.logger.info(f"def update_url(): Failed to update URL: {e}")
            return False


    def write_url_to_db(self, keywords, url, other_links, relevant, increment_crawl_trys):
        """
        Write or update an URL in the 'urls' table. 
        Parameters:
            org_names (str): The name of the organization associated with the URL.
            keywords (str): Keywords related to the URL.
            url (str): The URL to be written or updated in the database.
            other_links (str): Other links associated with the URL.
            relevant (bool): Indicates if the URL is relevant.
            increment_crawl_trys (int): The number of times the URL has been crawled.
        """
        if self.update_url(url, other_links, relevant, increment_crawl_trys + 1):
            self.logger.info(f"def write_url_to_db: URL {url} updated in the 'urls' table.")
        else:
            self.logger.info(f"write_url_to_db: Unable to update {url}. Inserting new entry")
            new_df = pd.DataFrame({
                "time_stamps": [datetime.now()],
                "org_names": ['Faceboook'],
                "keywords": [keywords],
                "links": [url],
                "other_links": [other_links],
                "relevant": [relevant],
                "crawl_trys": [increment_crawl_trys]
            })
            new_df.to_sql('urls', self.conn, if_exists='append', index=False)
            self.logger.info(f"def write_url_to_db: URL {url} written to the 'urls' table.")

    
    def clean_events(self, df):
        """
        This function performs the following operations:
        1. Ensures required columns exist in the DataFrame.
        2. Moves values from 'start.date' and 'end.date' to 'start.dateTime' and 'end.dateTime' if necessary.
        3. Drops the 'start.date' and 'end.date' columns.
        4. Subsets the DataFrame to only useful columns.
        5. Extracts and converts the price from the 'description' column.
        6. Cleans the 'description' column by removing HTML tags and unnecessary whitespace.
        7. Splits 'start.dateTime' and 'end.dateTime' into separate date and time columns.
        8. Renames columns to more descriptive names.
        9. Adds a 'Type_of_Event' column based on keywords in the 'Description' column.
        10. Converts 'Start_Date' and 'End_Date' to date format.
        11. Extracts the day of the week from 'Start_Date' and adds it to the 'Day_of_Week' column.
        12. Reorders the columns for better readability.
        13. Sorts the DataFrame by 'Start_Date' and 'Start_Time'.

        Parameters:
        df (pandas.DataFrame): The input DataFrame containing event data.

        Returns:
        pandas.DataFrame: The cleaned and processed DataFrame with relevant event information.
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
        df['Day_of_Week'] = pd.to_datetime(df['Start_Date']).dt.day_name()

        # Reorder the columns
        df = df[['URL', 'Type_of_Event', 'Name_of_the_Event', 'Day_of_Week', 'Start_Date', 
                'End_Date', 'Start_Time', 'End_Time', 'Price', 'Location', 'Description']]

        # Sort the DataFrame by Start_Date and Start_Time
        df = df.sort_values(by=['Start_Date', 'Start_Time']).reset_index(drop=True)

        # Return the collected events as a pandas dataframe

        return df


    def write_events_to_db(self, df, url):
        """
        Write events data to the 'events' table in the database.
        Parameters:
            df (pandas.DataFrame): DataFrame containing events data.
            url (str): URL from which the events data was sourced.
            keywords (str): Keywords associated with the events.
            org_name (str): Name of the organization hosting the events.

            Returns:
            None

            Notes:
            - The function converts 'Start_Date' and 'End_Date' columns to date format.
            - The function converts 'Start_Time' and 'End_Time' columns to time format.
            - If the 'Price' column exists and is not empty, it is cleaned and converted to numeric format.
              Otherwise, a warning is logged and the 'Price' column is filled with NaN.
            - Adds a 'Time_Stamp' column with the current timestamp.
            - Adds 'Keyword' and 'Org_Name' columns with the provided keywords and organization name.
            - Reorders columns to place 'Org_Name' and 'Keyword' at the beginning.
            - Writes the DataFrame to the 'events' table in the database. If the database connection is not available,
              an error is logged and the function returns without writing to the database.
        """
        # Save the events data to a CSV file for debugging purposes
        df.to_csv('events.csv', index=False)

        for col in ['start_date', 'end_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        for col in ['start_time', 'end_time']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

        if 'price' in df.columns and not df['price'].isnull().all():
            df['price'] = df['price'].replace({'\\$': '', '': None}, regex=True).infer_objects(copy=False)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        else:
            self.logger.warning("write_events_to_db: 'price' column is missing or empty. Filling with NaN.")
            df['price'] = float('nan')

        df['time_stamp'] = datetime.now()

        if self.conn is None:
            self.logger.error("write_events_to_db: Database connection is not available.")
            return

        df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        self.logger.info(f"write_events_to_db: Events data written to database for URL: {url}")

    
    def get_events(self, query):
        """
        Extracts events data based on the sql query provided.
        Parameters:
            query (str): Properly formatted sql.
        Returns:
            pandas.DataFrame: A DataFrame containing the extracted events data.
        """
        # Extract events data from the URL
        events_data = extract_events_data(url)

        # Clean and process the events data
        cleaned_events_data = clean_events(events_data)

        return cleaned_events_data


    def dedup(self):
        """
        Remove duplicates from 'events' and 'urls' tables in the database.

        This method connects to the database and removes duplicate entries from the 'events' and 'urls' tables.
        For the 'events' table, duplicates are identified based on the columns: 'Org_Name', 'Keyword', 'URL', 
        'Type_of_Event', 'Location', 'Day_of_Week', 'Start_Date', and 'End_Date'. For the 'urls' table, duplicates 
        are identified based on the 'links' column. The last occurrence of each duplicate is kept.

        If the database connection is not available, an error is logged and the method returns without making any changes.

        Raises:
            Exception: If there is an error during the deduplication process, an error is logged with the exception message.
        """
        try:
            df = pd.read_sql('SELECT * FROM events', self.conn)
            shape_before = df.shape
            self.logger.info(f"dedup: Deduplicating events table with {shape_before} rows and columns.")

            dedup_df = df.drop_duplicates(
                subset=["event_name", "start_date"],
                keep="last"
            )
            shape_after = dedup_df.shape
            self.logger.info(f"dedup: Deduplicated events table to {shape_after} rows and columns.")

            # Write the deduplicated DataFrame back to the database
            dedup_df.to_sql("events", self.conn, index=False, if_exists="replace")
            self.logger.info(f"def dedup(): Deduplicated events table.")

            df = pd.read_sql('SELECT * FROM urls', self.conn)
            shape_before = df.shape
            self.logger.info(f"dedup: Deduplicating urls table with {shape_before} rows and columns.")

            dedup_df = df.drop_duplicates(subset=["links"], keep="last")
            shape_after = dedup_df.shape
            dedup_df.to_sql("urls", self.conn, index=False, if_exists="replace")
            self.logger.info(f"dedup: Deduplicated urls table to {shape_after} rows and columns.")

        except Exception as e:
            self.logger.error(f"dedup: Failed to deduplicate tables: {e}")


    def set_calendar_urls(self):
        """
        Mark URLs containing 'calendar' in 'other_links' as relevant.
        """
        try:
            query = "SELECT * FROM urls WHERE other_links ILIKE %s"
            params = ("%calendar%",)
            urls_df = pd.read_sql_query(query, self.conn, params=params)

            for _, row in urls_df.iterrows():
                if not row['relevant']:
                    self.update_url(row['links'], update_other_links='No', relevant=True, increment_crawl_trys=0)

            self.logger.info(f"set_calendar_urls(): Marked {len(urls_df)} calendar URLs as relevant.")

        except Exception as e:
            self.logger.error(f"set_calendar_urls: Failed to mark calendar URLs: {e}")


if __name__ == "__main__":

    start_time = datetime.now()

    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set up logging
    logging.basicConfig(
        filename=config['logging']['log_file'],
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    db_handler = DatabaseHandler(config)
    db_handler.dedup()
    db_handler.set_calendar_urls()

    end_time = datetime.now()
    logging.info(f"__main__: Finished the process at {end_time}")
    logging.info(f"__main__: Total time taken: {end_time - start_time}")