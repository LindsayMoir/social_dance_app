# address_update.py
"""
This script updates addresses in the database by retrieving all addresses,
cleaning them up, and writing the updated rows back to the database.
Modules:
    logging: Provides logging capabilities.
    pandas: For data manipulation and analysis.
    yaml: For parsing YAML configuration files.
    db: Custom module for database handling.
Functions:
    process_addresses(db_handler):
Usage:
    Run this script directly to process and update addresses in the database.
"""
import logging
import pandas as pd
import yaml

from db import DatabaseHandler


with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

logging.basicConfig(
    filename=config["logging"]["log_file"],
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Instantiate the DatabaseHandler
db_handler = DatabaseHandler(config)


def process_addresses(db_handler):
    """
    Processes addresses from the database by retrieving, cleaning, and updating them.

    Args:
        db_handler (DatabaseHandler): An instance of DatabaseHandler that provides
                                      methods to interact with the database.

    Retrieves all addresses from the 'address' table in the database, renames the
    'full_address' column to 'location', cleans up the addresses using the 
    clean_up_address method of db_handler, and writes the updated addresses back 
    to the database.

    The function logs the number of rows/columns retrieved, the shape of the DataFrame
    after cleaning, and a success message after updating the database.

    Returns:
        None
    """
    sql = """
        SELECT * FROM address
        """
    df = pd.read_sql(sql, db_handler.conn)
    logging.info(f"def process_addresses(): Retrieved {df.shape} rows/columns.")

    # clean_up_address expects 'location' as the column name instead of full_address.
    # Rename full_address column to location
    df = df.rename(columns={'full_address': 'location'})

    # Clean up the address
    df = db_handler.clean_up_address(df)
    logging.info(f"def process_addresses(): Returned {df.shape} from clean_up_address")

    # Now we need to change the location column back to full_address
    df = df.rename(columns={'location': 'full_address'})

    # Turn df into a list of dictionaries
    address_dict_list = df.to_dict(orient='records')
    logging.info(f'address_dict_list is:\n{address_dict_list}')

    # Write the updated location to the address table. 
    # We won't need the address_id. We are not updating the events table.
    for address_dict in address_dict_list:
        _ = db_handler.get_address_id(address_dict)  # Pass each dictionary separately

    logging.info('def process_addresses(): Success: social_dance_db address table updated')


if __name__ == "__main__":
    process_addresses(db_handler)
