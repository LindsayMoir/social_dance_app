import logging
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine, MetaData, Table, text
import sys
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

print(db_handler.conn)
print(db_handler.address_db_engine)


import pandas as pd

data = [[1666, "Live band night - Lloyd Arntzen", "swing, balboa, lindy hop, east coast swing", "We're bringing in live music every Friday! Details at http://www.redhotswing.com/?page_id=22", "Friday", "2025-02-21", "2025-02-22", None, None, "Red Hot Swing", "Dance Victoria, 2750 Quadra St, Victoria, BC V8T 4E8, Canada", None, "https://www.google.com/calendar/event?eid=c29kZ2F0NDlybTJocjlnMjhzNTdvM3RvdTNfMjAyNTAyMjEgNzljZnAzYXNmN2JvZDlxNGljZ2M0cnR0bm9AZw", "other", None, "40:27.9"]]
columns = ['event_id', 'name', 'dance_styles', 'description', 'day_of_week', 'start_date', 'end_date', 'start_time', 'end_time', 'host', 'location', 'address', 'url', 'event_type', 'image_url', 'time_stamp']

events_df = pd.DataFrame(data=data, columns=columns)

print(events_df)

# Iterate over each row in the DataFrame
for index, row in events_df.iterrows():
    location = row['location'] or ''
    location = str(location).strip()

    print('location:', location)

    if location:
        # Extract postal codes using regex
        postal_code = re.search(r'[A-Za-z]\d[A-Za-z] \d[A-Za-z]\d', location)
        postal_code = postal_code.group() if postal_code else None
        postal_code = postal_code.replace(' ', '') if postal_code else None
        logging.info(f"clean_up_address: Extracted postal code '{postal_code}' from location '{location}'.")

        # Need to figure out WHICH address is the right one. Addresses share postal codes
        # Extract all numbers from location
        numbers = re.findall(r'\d+', location)
        print('numbers:', numbers)

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
                WHERE mail_postal_code = :postal_code;
                """
            )
            result_df = pd.read_sql(text(sql), 
                                    db_handler.address_db_engine, 
                                    params={'postal_code': postal_code})
            
            print(result_df)
            
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

                location = location.replace(' ,', ',')

                print('location:', location)

                # Update the location in the events DataFrame
                events_df[index, 'location'] = location
                logging.info(f"def clean_up_address(): Updated events_df.location to: '{location}'.")

                # Write the address to the address table
                address_dict = {
                    'full_address': location,
                    'street_number': result_df.civic_no.values[idx],
                    'street_name': result_df.official_street_name.values[idx],
                    'street_type': result_df.official_street_type.values[idx],
                    'postal_box': None,
                    'city': result_df.mail_mun_name.values[idx],
                    'province_or_state': result_df.mail_prov_abvn.values[idx],
                    'postal_code': result_df.mail_postal_code.values[idx],
                    'country_id': 'CA'
                }
                print(address_dict)
            else:
                logging.info(f"def clean_up_address(): No address found for postal code: {postal_code}.")
        else:
            logging.info(f"def clean_up_address(): No postal code found for location: '{location}'.")
    else:
        logging.info(f"def clean_up_address(): No location provided for row: {index}.")
