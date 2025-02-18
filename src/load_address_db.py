import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

db_connection_string = os.getenv('ADDRESS_DB_CONNECTION_STRING')
engine = create_engine(db_connection_string)  # Create the engine *outside* the loop

table_name = 'locations'

try:
    with engine.connect() as connection:  # Use 'with' for connection management
        create_table_query = f"""
        CREATE TABLE {table_name} (
            location_id SERIAL PRIMARY KEY,
            loc_guid UUID,
            addr_guid UUID,
            apt_no_label TEXT,
            civic_no INTEGER,
            civic_no_suffix TEXT,
            official_street_name TEXT,
            official_street_type TEXT,
            official_street_dir TEXT,
            prov_code TEXT,
            csd_eng_name TEXT,
            csd_fre_name TEXT,
            csd_type_eng_code TEXT,
            csd_type_fre_code TEXT,
            mail_street_name TEXT,
            mail_street_type TEXT,
            mail_street_dir TEXT,
            mail_mun_name TEXT,
            mail_prov_abvn TEXT,
            mail_postal_code TEXT,
            bg_dls_lsd TEXT,
            bg_dls_qtr TEXT,
            bg_dls_sctn TEXT,
            bg_dls_twnshp TEXT,
            bg_dls_rng TEXT,
            bg_dls_mrd TEXT,
            bg_x DOUBLE PRECISION,
            bg_y DOUBLE PRECISION,
            bu_n_civic_add TEXT,
            bu_use TEXT,
            time_stamp TIMESTAMP
        );
        """
        connection.execute(text(create_table_query)) #create the table
        print(f"Table {table_name} created successfully.")

except Exception as e:
    print(f"Error creating table: {type(e).__name__}: {e}")
    exit() # exit script if table creation fails


csv_folder = '/mnt/d/large_files/locations'
csv_files = [os.path.join(root, file) for root, dirs, files in os.walk(csv_folder) for file in files if file.endswith('.csv')]

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)  # Read CSV without pre-defined names if headers are in file
        original_columns = df.columns #Capture columns from the CSV file
        columns = [col.lower() for col in original_columns] #lowercase them
        df.columns = columns #set the dataframe columns to the lower case version

        df['time_stamp'] = datetime.datetime.now()

        with engine.connect() as connection: #new connection for each file
            df.to_sql(table_name, engine, if_exists='append', index=False) #append the data from each file

        print(f"Data from {csv_file} loaded successfully.")

    except Exception as e:
        print(f"Error loading data from {csv_file}: {type(e).__name__}: {e}")
        # Optionally, you can add a break here to stop on the first error:
        # break

print("Data loading process complete.")