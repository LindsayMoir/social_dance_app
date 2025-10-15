#!/usr/bin/env python3
import os
import io
import logging
from datetime import datetime

import pandas as pd
import pdfplumber
import requests
import yaml
from dateutil import parser as dateparser

# ── 1) Load configuration ───────────────────────────────────────────────────────
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ── 2) Set up centralized logging ──────────────────────────────────────────────
from logging_config import setup_logging
setup_logging('read_pdfs')

# ── 3) Initialize external handlers ────────────────────────────────────────────
from llm import LLMHandler
from db import DatabaseHandler

llm_handler = LLMHandler(config_path='config/config.yaml')
db_handler = llm_handler.db_handler  # Use the DatabaseHandler from LLMHandler

# ── 4) Parser registry decorator ───────────────────────────────────────────────
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

# ── 5) Main PDF‐reading class ───────────────────────────────────────────────────
class ReadPDFs:
    """
    Reads event PDFs from URLs, parses them into DataFrames,
    and writes both events and URL‐metadata to the database.
    """
    def __init__(self, config: dict):
        self.config = config
        self.csv_path = config.get('input', {}).get('pdfs')
        if not self.csv_path:
            raise ValueError("CSV path not found under config['input']['pdfs']")

        logging.info("Starting ReadPDFs…")
        logging.info(f"Using PDF list: {self.csv_path}")

        # Load blacklist domains
        bl_path = config.get('constants', {}).get('black_list_domains')
        self.black_list_domains = []
        if bl_path:
            try:
                df_bl = pd.read_csv(bl_path)
                # accept either “Domain” or “Domains” or first column
                col = 'Domain' if 'Domain' in df_bl else 'Domains' if 'Domains' in df_bl else df_bl.columns[0]
                self.black_list_domains = df_bl[col].astype(str).tolist()
                logging.info(f"Loaded {len(self.black_list_domains)} blacklisted domains")
            except Exception as e:
                logging.warning(f"Failed loading blacklist from {bl_path}: {e}")
        else:
            logging.info("No black_list_domains configured.")

        # Use the global database handler
        self.db = db_handler
        logging.info("DatabaseHandler initialized.")


    def read_write_pdf(self) -> pd.DataFrame:
        file_name = os.path.basename(__file__)
        start_df = self.db.count_events_urls_start(file_name)

        sources = pd.read_csv(self.csv_path, dtype=str)
        all_events = []

        for idx, row in sources.iterrows():
            source     = row.get('source', '')
            pdf_url    = row.get('pdf_url', '')
            parent_url = row.get('parent_url', '')
            keywords   = row.get('keywords', None)

            logging.info(f"read_write_pdf(): [{idx}] source={source}, pdf_url={pdf_url}")

            # Skip blacklisted
            if any(domain in pdf_url for domain in self.black_list_domains):
                logging.info(f"read_write_pdf(): Skipping blacklisted URL: {pdf_url}")
                continue

            # Skip if already in events (or copied from history)
            if self.db.check_image_events_exist(pdf_url):
                logging.info(f"read_write_pdf(): Already have events for URL: {pdf_url}")
                self.db.write_url_to_db((pdf_url, parent_url, source, keywords, True, 1, datetime.now()))
                continue

            # Should we crawl it?
            if not self.db.should_process_url(pdf_url):
                logging.info(f"read_write_pdf():should_process_url returned False for {pdf_url}")
                self.db.write_url_to_db((pdf_url, parent_url, source, keywords, False, 1, datetime.now()))
                continue

            # Find the right parser
            parser = PARSER_REGISTRY.get(source)
            if not parser:
                logging.warning(f"read_write_pdf(): No parser registered for '{source}'")
                continue

            # Download and parse
            resp = requests.get(pdf_url, timeout=30)
            if resp.status_code == 404:
                logging.warning(f"read_write_pdf(): PDF not found (404) for '{source}': {pdf_url}")
                continue
            resp.raise_for_status()
            pdf_file = io.BytesIO(resp.content)

            df = parser(pdf_file)
            if df is None or df.empty:
                logging.warning(f"read_write_pdf(): Parser returned no events for '{source}'")
                continue

            # Clean & enrich
            df = df.dropna(subset=['event_name', 'start_date'])
            if df.empty:
                logging.warning(f"read_write_pdf(): All rows dropped for '{source}' after cleaning")
                continue

            df['source']    = source
            df['url']       = pdf_url
            df['address_id']= None
            df['time_stamp']= datetime.now()

            records = df.to_dict(orient='records')
            logging.info(f"read_write_pdf(): Inserting {len(records)} events for '{source}'")
            self.db.multiple_db_inserts('events', records)

            # Mark URL as done
            self.db.write_url_to_db((pdf_url, parent_url, source, keywords, True, 1, datetime.now()))
            all_events.append(df)

        # No events at all?
        if not all_events:
            logging.info("read_write_pdf(): pdf_url is: {pdf_url}")
            logging.info("read_write_pdf(): No NEW events parsed BUT events may have been copied from events_history.")
            logging.info("read_write_pdf(): returning empty DataFrame.")
            self.db.count_events_urls_end(start_df, file_name)
            return pd.DataFrame(columns=[
                'event_name','dance_style','description','day_of_week',
                'start_date','end_date','start_time','end_time',
                'source','location','price','url','event_type',
                'address_id','time_stamp'
            ])

        result = pd.concat(all_events, ignore_index=True)
        logging.info(f"Total events processed: {len(result)}")
        self.db.count_events_urls_end(start_df, file_name)
        return result

# ── 6) PDF parsers ─────────────────────────────────────────────────────────────
@register_parser("Victoria Summer Music")
def parse_victoria_summer_music(pdf_file) -> pd.DataFrame:
    logging.info("Parsing Victoria Summer Music PDF…")
    cols = ['Mth','Day','Date','Location','Time','Event','Description']
    rows = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table or len(table) < 2:
                continue
            for r in table[1:]:
                r = list(r or [])
                if len(r) < len(cols):
                    r += [''] * (len(cols)-len(r))
                rows.append(r[:len(cols)])

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows, columns=cols)
    year = datetime.now().year
    raw['start_date'] = pd.to_datetime(raw['Mth'] + ' ' + raw['Date'] + f' {year}',
                                       format='%b %d %Y', errors='coerce')
    raw['end_date'] = raw['start_date']

    def parse_times(ts):
        if pd.isna(ts) or not isinstance(ts, str):
            return pd.Series({'start_time': None, 'end_time': None})
        ts = ts.lower().replace('noon','12:00pm')
        parts = ts.split(' to ')
        try:
            st = dateparser.parse(parts[0].strip(), fuzzy=True).time()
        except:
            st = None
        et = None
        if len(parts) > 1:
            try:
                et = dateparser.parse(parts[1].strip(), fuzzy=True).time()
            except:
                pass
        return pd.Series({'start_time': st, 'end_time': et})

    times = raw['Time'].apply(parse_times)
    df = pd.concat([
        raw.rename(columns={'Event':'event_name','Description':'description','Location':'location'})[
            ['event_name','description','location','start_date','end_date']
        ],
        times
    ], axis=1)

    df['day_of_week'] = df['start_date'].dt.day_name()
    df['dance_style'] = 'lindy, swing, wcs'
    df['price']       = 'Free'
    df['event_type']  = 'dance, live music'
    return df

def dump_pdf_text(pdf_file) -> str:
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            pages.append(f"----- Page {i} -----\n{text}")
    return "\n".join(pages)

@register_parser("The Butchart Gardens Outdoor Summer Concerts")
def parse_butchart_gardens_concerts(pdf_file) -> pd.DataFrame:
    logging.info("Parsing Butchart Gardens concerts PDF…")
    text = dump_pdf_text(pdf_file)
    prompt, schema_type = llm_handler.generate_prompt(pdf_file, text, 'images')
    if len(prompt) > config['crawling']['prompt_max_length']:
            logging.warning(f"def process_llm_response: Prompt for URL {url} exceeds maximum length. Skipping LLM query.")
            return None
    
    llm_response = llm_handler.query_openai(
        prompt=prompt,
        model=config['llm']['openai_model'],
        image_url=config['input']['butchart_image_url'],
        schema_type=schema_type
    )

    if not llm_response:
        logging.error("No response from LLM for Butchart parser.")
        return None

    parsed = llm_handler.extract_and_parse_json(llm_response,
                                                config['input']['butchart_pdf_url'], schema_type)
    if not parsed:
        logging.error("Failed to parse JSON from LLM response.")
        return None

    df = pd.DataFrame(parsed)
    df['dance_style'] = 'ballroom, swing, wcs, west coast swing'
    df['event_type']  = 'social dance, live music'
    df['url']         = config['input']['butchart_pdf_url']
    return df

# ── 7) Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    reader = ReadPDFs(config)
    df = reader.read_write_pdf()

    if df is not None and not df.empty:
        logging.info(f"Result DataFrame head:\n{df.head()}")
        logging.info(f"Completed: {len(df)} events inserted.")
    else:
        logging.info("Completed: no events inserted.")
