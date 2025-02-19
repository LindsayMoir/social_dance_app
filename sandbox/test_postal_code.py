import logging
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, text
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

postal_code = 'V8N1S3'

sql = "SELECT * FROM locations WHERE mail_postal_code = 'V8N1S3';"
df = pd.read_sql(sql, db_handler.address_db_engine)
print(df)

sql = ("""
SELECT 
    apt_no_label,
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