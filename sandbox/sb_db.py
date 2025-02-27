from datetime import datetime
from dotenv import load_dotenv
import json
from io import StringIO
import logging
import os
import pandas as pd
import re
from sqlalchemy import create_engine, text
import yaml

from llm import LLMHandler
from db import DatabaseHandler

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

db_handler = DatabaseHandler(config)

logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("sb_db.py starting...")

df = pd.read_csv('output/irrelevant_rows.csv') # temp code
df = df.head(1)

before_dropna = df.shape[0]
# Remove NaN values from the event_type column
df = df.dropna(subset=["event_type"])
after_dropna = df.shape[0]
if before_dropna != after_dropna:
    logging.error(f"update_dance_styles(): before_dropna {before_dropna} rows after_dropna {after_dropna}.")

# Prepare data for updating rows in the database
for idx, row in df.iterrows():
    print(row)
    update_query = """
    UPDATE events
    SET event_type = :event_type
    WHERE event_id = :event_id
    """
    params = {
    "event_type": row["event_type"],
    "event_id": row["event_id"]
    }
    logging.info(f"update_dance_styles(): {update_query}.")
    db_handler.execute_query(update_query, params=params)

logging.info(f"update_dance_styles(): Updated {idx} rows (dance_style) in the database.")