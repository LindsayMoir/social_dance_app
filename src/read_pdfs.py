import os
import re
import pandas as pd
import pdfplumber
import requests
import io
import logging
from datetime import datetime
from dateutil import parser as dateparser
import yaml

# Configure console logging
tlogging = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parser registry and decorator
PARSER_REGISTRY = {}

def register_parser(source_name: str):
    """
    Decorator to register PDF parsing functions for a given source.
    """
    def decorator(fn):
        PARSER_REGISTRY[source_name] = fn
        logging.info(f"Registered parser for source: {source_name}")
        return fn
    return decorator

# Database handler import
from db import DatabaseHandler
from sqlalchemy import Table, insert

class ReadPDFs:
    """
    Class to read event PDFs from URLs, parse them into DataFrames,
    and batch-insert into the database.
    """
    def __init__(self, config: dict):
        self.config = config
        self.csv_path = config.get('input', {}).get('pdfs')
        if not self.csv_path:
            raise ValueError("CSV path not found under config['input']['pdfs']")

        # File logging
        log_file = config.get('logging', {}).get('read_pdfs')
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(fh)
            logging.info(f"Logging to file: {log_file}")

        logging.info(f"Initializing ReadPDFs with CSV: {self.csv_path}")
        self.db = DatabaseHandler(config)
        logging.info("DatabaseHandler initialized.")

    def read_write_pdf(self) -> pd.DataFrame:
        logging.info(f"Reading CSV: {self.csv_path}")
        sources = pd.read_csv(self.csv_path, dtype=str)
        all_events = []

        for idx, row in sources.iterrows():
            source = row.get('source', '')
            pdf_url = row.get('pdf_url', '')
            parent_url = row.get('parent_url', '')
            keywords = row.get('keywords', None)
            logging.info(f"Row {idx}: source={source}, pdf_url={pdf_url}")

            parser = PARSER_REGISTRY.get(source)
            if not parser:
                logging.warning(f"No parser for source '{source}', skipping.")
                continue

            # download PDF
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            pdf_file = io.BytesIO(resp.content)

            # parse PDF table
            df = parser(pdf_file)
            if df.empty:
                logging.warning(f"Parser returned empty DataFrame for '{source}'.")
                continue

            # drop rows missing essential fields
            df = df.dropna(subset=['event_name','start_date'])
            if df.empty:
                logging.warning(f"All rows dropped for '{source}' after cleaning.")
                continue

            # stamp metadata
            df['source'] = source
            df['url'] = pdf_url
            df['address_id'] = None
            df['time_stamp'] = datetime.now()

            # batch insert
            records = df.to_dict(orient='records')
            logging.info(f"Batch inserting {len(records)} events for '{source}'")
            self.db.multiple_db_inserts('events', records)

            all_events.append(df)

        if not all_events:
            logging.info("No events parsed; returning empty DataFrame.")
            cols = [
                'event_name','dance_style','description','day_of_week',
                'start_date','end_date','start_time','end_time',
                'source','location','price','url','event_type',
                'address_id','time_stamp'
            ]
            return pd.DataFrame(columns=cols)

        result = pd.concat(all_events, ignore_index=True)
        logging.info(f"Total events processed: {len(result)}")
        return result

@register_parser("Victoria Summer Music")
def parse_victoria_summer_music(pdf_file) -> pd.DataFrame:
    """
    Parse the Victoria Summer Music PDF using pdfplumber tables,
    skip the header row on each page, and pad any row missing Description.
    Columns: ['Mth', 'Day', 'Date', 'Location', 'Time', 'Event', 'Description']
    """
    logging.info("Parsing PDF for 'Victoria Summer Music' via table extraction.")
    all_pages = []
    col_names = ['Mth', 'Day', 'Date', 'Location', 'Time', 'Event', 'Description']

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            table = page.extract_table()
            if not table or len(table) < 2:
                logging.info(f"Page {page_num} has no data rows, skipping.")
                continue
            raw_rows = table[1:]
            normalized = []
            for row in raw_rows:
                # Convert to list to allow padding/truncation
                row_list = list(row)
                # pad or truncate to exactly 7 columns
                if len(row_list) < len(col_names):
                    row_list += [''] * (len(col_names) - len(row_list))
                elif len(row_list) > len(col_names):
                    row_list = row_list[:len(col_names)]
                normalized.append(row_list)
            page_df = pd.DataFrame(normalized, columns=col_names)
            logging.info(f"Extracted {len(page_df)} rows from page {page_num}.")
            all_pages.append(page_df)

    if not all_pages:
        logging.warning("No tables found in PDF pages.")
        return pd.DataFrame()

    raw = pd.concat(all_pages, ignore_index=True)
    logging.info(f"Total raw rows across all pages: {len(raw)}.")

    # parse dates
    year = datetime.now().year
    raw['start_date'] = pd.to_datetime(
        raw['Mth'] + ' ' + raw['Date'] + f' {year}',
        format='%b %d %Y', errors='coerce'
    )
    raw['end_date'] = raw['start_date']

    # parse times
    def parse_times(ts):
        """
        Parse a time string, handling 'noon' by converting to 12:00pm.
        """
        if pd.isna(ts) or not isinstance(ts, str):
            return pd.Series({'start_time': None, 'end_time': None})
        # normalize 'noon' to '12:00pm'
        ts = ts.lower().replace('noon', '12:00pm')
        parts = ts.split(' to ')
        try:
            start = dateparser.parse(parts[0].strip(), fuzzy=True).time() if parts[0].strip() else None
        except Exception:
            start = None
        try:
            end = dateparser.parse(parts[1].strip(), fuzzy=True).time() if len(parts) > 1 and parts[1].strip() else None
        except Exception:
            end = None
        return pd.Series({'start_time': start, 'end_time': end})

    times_df = raw['Time'].apply(parse_times)
    df = pd.DataFrame({
        'event_name': raw['Event'],
        'description': raw['Description'],
        'location': raw['Location'],
        'start_date': raw['start_date'],
        'end_date': raw['end_date'],
    })
    df = pd.concat([df, times_df], axis=1)

    df['day_of_week'] = df['start_date'].dt.day_name()
    df['dance_style'] = 'lindy, swing, wcs'
    df['price'] = 'Free'
    df['event_type'] = 'dance, live music'

    logging.info(f"Parsed DataFrame with {len(df)} events.")
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir,'config','config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    reader = ReadPDFs(config)
    logging.info('Starting PDF processing.')
    df = reader.read_write_pdf()
    logging.info(f"Result df head:\n{df.head()}")
    logging.info(f"Completed. Events: {len(df)}")
