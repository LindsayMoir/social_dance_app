# admin.py

import logging
import pandas as pd
from datetime import datetime

from db.py import DatabaseHandler


class Admin:
    def __init__(self, config):
        """
        Initializes the Admin class for handling administrative database tasks.
        
        Args:
            config (dict): Configuration dictionary containing database settings.
        """
        self.config = config
        self.db_handler = DatabaseHandler(config)
        self.engine = self.db_handler.get_db_connection()


    async def write_run_statistics(self, run_stats):
        """
        Writes statistics of the Eventbrite scraping run to the 'runs' table in the database.

        Args:
            run_stats (df): Dataframe containing all run statistics.
        """
        try:
            run_stats["elapsed_time"] = str(run_stats["end_time"] - run_stats["start_time"])  # Convert timedelta to string
            run_stats["time_stamp"] = datetime.now()

            # Create DataFrame from dictionary
            run_data = pd.DataFrame([run_stats])

            # Write DataFrame to PostgreSQL
            run_data.to_sql("runs", self.engine, if_exists="append", index=False)
            logging.info(f"write_run_statistics(): Run statistics written to database for {run_stats['run_name']}.")

        except Exception as e:
            logging.error(f"write_run_statistics(): Error writing run statistics: {e}")
