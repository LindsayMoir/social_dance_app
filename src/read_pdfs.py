import os
import re
import io
import logging
from datetime import datetime

import pandas as pd
import pdfplumber
import requests
import yaml
from dateutil import parser as dateparser

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
    Class to read event PDFs from URLs, parse into DataFrames,
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

        # Load blacklisted domains
        black_list_path = config.get('constants', {}).get('black_list_domains')
        self.black_list_domains = []
        if black_list_path:
            try:
                bl_df = pd.read_csv(black_list_path)
                if 'Domain' in bl_df.columns:
                    self.black_list_domains = bl_df['Domain'].astype(str).tolist()
                elif 'Domains' in bl_df.columns:
                    self.black_list_domains = bl_df['Domains'].astype(str).tolist()
                else:
                    self.black_list_domains = bl_df.iloc[:,0].astype(str).tolist()
                logging.info(f"Loaded {len(self.black_list_domains)} blacklisted domains.")
            except Exception as e:
                logging.warning(f"Failed to load black_list_domains from {black_list_path}: {e}")
        else:
            logging.info("No black_list_domains configured.")

        logging.info(f"Initializing ReadPDFs with CSV: {self.csv_path}")
        self.db = DatabaseHandler(config)
        logging.info("DatabaseHandler initialized.")

    def read_write_pdf(self) -> pd.DataFrame:
        # Name of this file
        file_name = os.path.basename(__file__)
        # Count at start
        start_df = self.db.count_events_urls_start(file_name)

        logging.info(f"Reading CSV: {self.csv_path}")
        sources = pd.read_csv(self.csv_path, dtype=str)
        all_events = []

        for idx, row in sources.iterrows():
            source = row.get('source','')
            pdf_url = row.get('pdf_url','')
            parent_url = row.get('parent_url','')
            keywords = row.get('keywords',None)
            logging.info(f"Row {idx}: source={source}, pdf_url={pdf_url}")

            # Skip blacklisted URLs
            if any(bl in pdf_url for bl in self.black_list_domains):
                logging.info(f"Skipping blacklisted URL: {pdf_url}")
                continue

            parser = PARSER_REGISTRY.get(source)
            if not parser:
                logging.warning(f"No parser for source '{source}', skipping.")
                continue

            # Download
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            pdf_file = io.BytesIO(resp.content)

            # Parse
            df = parser(pdf_file)
            if df.empty:
                logging.warning(f"Parser returned empty DataFrame for '{source}'.")
                continue

            # Clean
            df = df.dropna(subset=['event_name','start_date'])
            if df.empty:
                logging.warning(f"All rows dropped for '{source}' after cleaning.")
                continue

            # Stamp metadata
            df['source'] = source
            df['url'] = pdf_url
            df['address_id'] = None
            df['time_stamp'] = datetime.now()

            # Insert
            records = df.to_dict(orient='records')
            logging.info(f"Batch inserting {len(records)} events for '{source}'")
            self.db.multiple_db_inserts('events', records)
            url_row = (pdf_url, parent_url, source, keywords, True, 1, datetime.now())
            self.db.write_url_to_db(url_row)

            all_events.append(df)

        # If none parsed
        if not all_events:
            logging.info("No events parsed; returning empty DataFrame.")
            cols = [
                'event_name','dance_style','description','day_of_week',
                'start_date','end_date','start_time','end_time',
                'source','location','price','url','event_type',
                'address_id','time_stamp'
            ]
            url_row = (pdf_url, parent_url, source, keywords, False, 1, datetime.now())
            self.db.write_url_to_db(url_row)
            self.db.count_events_urls_end(start_df, file_name)
            logging.info(f"Wrote events and urls statistics to: {file_name}")
            return pd.DataFrame(columns=cols)

        # Normal path
        result = pd.concat(all_events, ignore_index=True)
        logging.info(f"Total events processed: {len(result)}")
        self.db.count_events_urls_end(start_df, file_name)
        logging.info(f"Wrote events and urls statistics to: {file_name}")
        return result


@register_parser("Victoria Summer Music")
def parse_victoria_summer_music(pdf_file) -> pd.DataFrame:
    logging.info("Parsing PDF for 'Victoria Summer Music'.")
    all_pages=[]
    cols=['Mth','Day','Date','Location','Time','Event','Description']
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table=page.extract_table()
            if not table or len(table)<2: continue
            for row in table[1:]:
                row_list=list(row)
                if len(row_list)<len(cols): row_list+=['']*(len(cols)-len(row_list))
                else: row_list=row_list[:len(cols)]
                all_pages.append(row_list)
    if not all_pages: return pd.DataFrame()
    raw=pd.DataFrame(all_pages,columns=cols)

    year=datetime.now().year
    raw['start_date']=pd.to_datetime(raw['Mth']+' '+raw['Date']+f' {year}',format='%b %d %Y',errors='coerce')
    raw['end_date']=raw['start_date']

    def parse_times(ts):
        if pd.isna(ts) or not isinstance(ts,str): return pd.Series({'start_time':None,'end_time':None})
        ts=ts.lower().replace('noon','12:00pm')
        parts=ts.split(' to ')
        st,et=None,None
        try: st=dateparser.parse(parts[0].strip(),fuzzy=True).time()
        except: pass
        if len(parts)>1:
            try: et=dateparser.parse(parts[1].strip(),fuzzy=True).time()
            except: pass
        return pd.Series({'start_time':st,'end_time':et})

    times=raw['Time'].apply(parse_times)
    df=pd.concat([raw.rename(columns={'Event':'event_name','Description':'description','Location':'location'})[['event_name','description','location','start_date','end_date']],times],axis=1)
    df['day_of_week']=df['start_date'].dt.day_name()
    df['dance_style']='lindy, swing, wcs'
    df['price']='Free'
    df['event_type']='dance, live music'
    return df

@register_parser("The Butchart Gardens Outdoor Summer Concerts")
def parse_butchart_gardens_concerts(pdf_file) -> pd.DataFrame:
    logging.info("Parsing PDF for 'The Butchart Gardens Outdoor Summer Concerts'.")
    current_year = datetime.now().year
    events = []
    # Map page indices explicitly to months
    month_map = {0: "July", 1: "August"}

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            month = month_map.get(page_num)
            if not month:
                logging.warning(f"Unexpected page {page_num}; skipping.")
                continue
            logging.info(f"[DEBUG] Using month '{month}' for page {page_num}")

            table = page.extract_table()
            if not table or len(table) < 2:
                logging.info("No calendar grid on this page, skipping.")
                continue

            headers = [h.strip() for h in table[0]]
            logging.info(f"[DEBUG] Calendar headers: {headers!r}")

            for row_idx, row in enumerate(table[1:], start=1):
                for col_idx, cell in enumerate(row):
                    cell_text = cell or ""
                    weekday = headers[col_idx] if col_idx < len(headers) else f"col{col_idx}"
                    logging.info(f"[DEBUG] Cell({row_idx},{weekday}) raw: {cell_text[:100]!r}")
                    if not cell_text.strip():
                        continue

                    # Improved listing extraction: handle standalone day numbers
                    listings = []
                    current = None
                    for ln in cell_text.split("\n"):
                        text = ln.strip()
                        if not text:
                            continue
                        # pure day number
                        if re.fullmatch(r"\d{1,2}", text):
                            if current:
                                listings.append(current)
                            current = {"day_num": int(text), "rest": ""}
                        # day+rest on one line
                        elif re.match(r"^\d{1,2}\s+", text):
                            parts = text.split(None, 1)
                            if current:
                                listings.append(current)
                            current = {"day_num": int(parts[0]), "rest": parts[1] if len(parts)>1 else ""}
                        # continuation
                        elif current:
                            sep = " " if current["rest"] else ""
                            current["rest"] += sep + text
                    if current:
                        listings.append(current)

                    logging.info(f"[DEBUG] Parsed listings: {listings!r}")

                    for L in listings:
                        t_match = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:am|pm))$", L["rest"], re.IGNORECASE)
                        if t_match:
                            t_str = t_match.group(1)
                            try:
                                start_time = dateparser.parse(t_str).time()
                            except Exception:
                                start_time = None
                            title = L["rest"][:t_match.start()].strip()
                        else:
                            start_time = None
                            title = L["rest"].strip()

                        try:
                            dt = datetime.strptime(f"{month} {L['day_num']} {current_year}", "%B %d %Y")
                        except ValueError:
                            logging.warning(f"Could not parse date: {month} {L['day_num']}")
                            continue

                        logging.info(f"[DEBUG] Appending event: {title!r} on {dt} at {start_time}")
                        events.append({
                            "event_name": title,
                            "dance_style": "ballroom, swing, wcs",
                            "description": "",
                            "day_of_week": weekday,
                            "start_date": dt,
                            "end_date": dt,
                            "start_time": start_time,
                            "end_time": None,
                            "source": "The Butchart Gardens Outdoor Summer Concerts",
                            "location": "The Butchart Gardens",
                            "price": "Free with admission",
                            "url": "https://butchartgardens.com/summer-entertainment-calendar/",
                            "event_type": "live music",
                            "address_id": None,
                            "time_stamp": datetime.now()
                        })

    return pd.DataFrame(events)

def dump_pdf_text(pdf_file) -> str:
    """
    Extract and return all text from the given PDF file buffer, with page separators.
    """
    full_text = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            full_text.append(f"----- Page {page_num} -----\n{text}\n")
            logging.info(f"dump_pdf_text(): Here is the full text: \n{full_text}")
            return "\n".join(full_text)

if __name__=='__main__':
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path=os.path.join(base_dir,'config','config.yaml')
    with open(config_path) as f:
        config=yaml.safe_load(f)
    logging.info('Starting PDF processing.')
    reader=ReadPDFs(config)
    #df=reader.read_write_pdf()   # TEMP***

    full_text = dump_pdf_text('data/other/Summer-Entertainment-calendar-double-sided-2025.pdf')

    logging.info(f"Result df head:\n{df.head()}")
    logging.info(f"Completed. Events: {len(df)}")
