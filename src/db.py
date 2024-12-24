import logging
import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, update, MetaData
import yaml

print(os.getcwd())

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


    def write_url_to_db(self, org_names, keywords, url, other_links, relevant, increment_crawl_trys):
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
        try:
            self.update_url(url, other_links, relevant, increment_crawl_trys)
        except Exception as e:
            self.logger.info(f"write_url_to_db: Unable to update {url}. Inserting new entry: {e}")
            update_df = pd.DataFrame({
                "time_stamps": [datetime.now()],
                "org_names": [org_names],
                "keywords": [keywords],
                "links": [url],
                "other_links": [other_links],
                "relevant": [relevant],
                "crawl_trys": [increment_crawl_trys]
            })
            update_df.to_sql('urls', self.conn, if_exists='append', index=False)
            self.logger.info(f"def write_url_to_db: URL {url} written to the 'urls' table.")


    def write_events_to_db(self, df, url, keywords, org_name):
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

        for col in ['Start_Date', 'End_Date']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
        for col in ['Start_Time', 'End_Time']:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.time

        if 'Price' in df.columns and not df['Price'].isnull().all():
            df['Price'] = df['Price'].replace({'\\$': '', '': None}, regex=True)
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        else:
            self.logger.warning("write_events_to_db: 'Price' column is missing or empty. Filling with NaN.")
            df['Price'] = float('nan')

        df['Time_Stamp'] = datetime.now()
        df['Keyword'] = keywords
        df['Org_Name'] = org_name

        columns_order = ['Org_Name', 'Keyword'] + [col for col in df.columns if col not in ['Org_Name', 'Keyword']]
        df = df[columns_order]

        if self.conn is None:
            self.logger.error("write_events_to_db: Database connection is not available.")
            return

        df.to_sql('events', self.conn, if_exists='append', index=False, method='multi')
        self.logger.info(f"write_events_to_db: Events data written to database for URL: {url}")


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
                subset=["Org_Name", "Keyword", "URL", "Type_of_Event", "Location", "Day_of_Week", "Start_Date", "End_Date"],
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