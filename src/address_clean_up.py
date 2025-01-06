import logging
import pandas as pd
import pyap  # Using pyap for address parsing
import yaml

from db import DatabaseHandler
from sqlalchemy import text  # Importing text construct from SQLAlchemy

# Load configuration from a YAML file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Set up logging
logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='a',  # Changed to append mode to preserve logs
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("\n\naddress_clean_up started.")

# Initialize the DatabaseHandler and get the connection
db_handler = DatabaseHandler(config)
conn = db_handler.get_db_connection()

def create_address_table():
    """
    Creates the 'address' table if it does not exist.
    """
    query = """
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
    try:
        with conn.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        logging.info("create_address_table: 'address' table created or already exists.")
    except Exception as e:
        logging.error(f"create_address_table: Failed to create 'address' table: {e}")

def get_address_id(address_dict):
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
        with conn.connect() as connection:
            result = connection.execute(text(select_query), params).fetchone()
            if result:
                logging.info(f"get_address_id: Found existing address_id {result[0]} for address '{address_dict['full_address']}'.")
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

def clean_up_address():
    """
    Cleans up and standardizes address data from the 'events' table.
    It parses the 'location' field, inserts unique addresses into the 'address' table,
    and updates the 'events' table with the corresponding address_id.
    """
    try:
        # Read the entire 'events' table into a DataFrame
        events_df = pd.read_sql_query("SELECT * FROM events", conn)
        logging.info(f"clean_up_address: Retrieved {len(events_df)} records from 'events' table.")

        # Add 'address_id' column to 'events' table if it doesn't exist
        if 'address_id' not in events_df.columns:
            alter_query = "ALTER TABLE events ADD COLUMN address_id INTEGER"
            with conn.connect() as connection:
                connection.execute(text(alter_query))
                connection.commit()
            logging.info("clean_up_address: Added 'address_id' column to 'events' table.")

        # Iterate over each row in the DataFrame
        for index, row in events_df.iterrows():
            location = row.get('location') or ''
            location = str(location).strip()
            if not location:
                logging.warning(f"clean_up_address: Skipping row {index} due to empty 'location'.")
                continue  # Skip if 'location' is empty

            # Parse the address using pyap for Canadian addresses
            parsed_addresses = pyap.parse(location, country='CA')
            if not parsed_addresses:
                logging.warning(f"clean_up_address: No address found in 'location' for row {index}.")
                continue  # Skip if no address is found

            # Assuming one address per event; modify if multiple addresses per event exist
            address = parsed_addresses[0]

            # Debug: Log the attributes of the Address object
            logging.debug(f"Row {index} Address Attributes: {address.__dict__}")

            # Create a dictionary of address components using getattr to handle missing attributes
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
                'country_id': getattr(address, 'country_id', 'CA')  # Changed 'Canada' to 'CA' for consistency
            }

            # Log the constructed address_dict for debugging
            logging.debug(f"Row {index} Address Dict: {address_dict}")

            # Retrieve or insert the address and get its address_id
            address_id = get_address_id(address_dict)
            if address_id is None:
                logging.error(f"clean_up_address: Failed to obtain address_id for row {index}.")
                continue  # Skip updating this row if address_id is not available

            # Update the 'address_id' in the 'events' table for the current row
            event_id = row.get('event_id')  # Ensure this matches your primary key column
            if event_id is not None:
                update_query = "UPDATE events SET address_id = :address_id WHERE event_id = :event_id"
                params = {'address_id': address_id, 'event_id': event_id}
                try:
                    with conn.connect() as connection:
                        connection.execute(text(update_query), params)
                        connection.commit()
                    logging.info(f"clean_up_address: Updated 'address_id' for event_id {event_id}.")
                except Exception as e:
                    logging.error(f"clean_up_address: Failed to update 'address_id' for event_id {event_id}: {e}")
            else:
                logging.warning(f"clean_up_address: 'event_id' not found for row {index}; cannot update 'address_id'.")

    except Exception as e:
        logging.error(f"clean_up_address: An error occurred during address cleanup: {e}")

if __name__ == '__main__':
    create_address_table()
    clean_up_address()
    try:
        db_handler.close_connection()  # This should now work correctly
        logging.info("address_clean_up: Database connection closed successfully.")
    except AttributeError as e:
        logging.error(f"address_clean_up: Failed to close database connection: {e}")
    except Exception as e:
        logging.error(f"address_clean_up: Unexpected error while closing connection: {e}")
