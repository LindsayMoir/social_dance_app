#!/usr/bin/env python3
import os
import io
import logging
import re
from datetime import date, datetime
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
import yaml
from dateutil import parser as dateparser

from config_runtime import get_config_path, load_config

# ── 1) Load configuration ───────────────────────────────────────────────────────
config = load_config()

# ── 2) Set up centralized logging ──────────────────────────────────────────────
from logging_config import setup_logging
setup_logging('read_pdfs')

# ── 3) Initialize external handlers ────────────────────────────────────────────
from llm import LLMHandler
from db import DatabaseHandler

_llm_handler: LLMHandler | None = None
_db_handler: DatabaseHandler | None = None


def get_llm_handler() -> LLMHandler:
    global _llm_handler
    if _llm_handler is None:
        logging.info("read_pdfs: Initializing LLMHandler lazily.")
        _llm_handler = LLMHandler(config_path=get_config_path())
    return _llm_handler


def get_db_handler() -> DatabaseHandler:
    global _db_handler
    if _db_handler is None:
        logging.info("read_pdfs: Initializing DatabaseHandler lazily via LLMHandler.")
        _db_handler = get_llm_handler().db_handler
    return _db_handler

# ── 4) Parser registry decorator ───────────────────────────────────────────────
PARSER_REGISTRY = {}
DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
BUTCHART_SOURCE_NAME = "The Butchart Gardens Outdoor Summer Concerts"
DOWNLOAD_LINK_TEXT_PATTERN = re.compile(r"download|calendar", re.IGNORECASE)


def _coerce_bool(value: object, default: bool = True) -> bool:
    """Interpret CSV-style truthy and falsy values safely."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_optional_date(value: object) -> date | None:
    """Parse a CSV date value into a date, returning None when blank/invalid."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()

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
        self.db = get_db_handler()
        logging.info("DatabaseHandler initialized.")

    @staticmethod
    def is_pdf_source_active(row: pd.Series, today: date | None = None) -> tuple[bool, str]:
        """
        Determine whether a PDF source row is active for the current date.

        Supported optional CSV columns:
        1. `enabled`
        2. `active_start_date`
        3. `active_end_date`
        """
        today_value = today or datetime.now().date()

        if not _coerce_bool(row.get("enabled"), default=True):
            return False, "disabled"

        active_start = _parse_optional_date(row.get("active_start_date"))
        active_end = _parse_optional_date(row.get("active_end_date"))

        if active_start and today_value < active_start:
            return False, "before_active_start_date"
        if active_end and today_value > active_end:
            return False, "after_active_end_date"
        return True, "active"

    @staticmethod
    def resolve_parent_page_assets(
        source: str,
        parent_url: str,
        fallback_pdf_url: str = "",
        fallback_image_url: str = "",
    ) -> dict[str, str]:
        """
        Resolve the current PDF and optional preview image from a stable parent page.

        Falls back to legacy URLs when the parent page cannot be fetched or its markup changes.
        """
        resolved = {
            "parent_url": parent_url,
            "pdf_url": fallback_pdf_url or "",
            "image_url": fallback_image_url or "",
        }
        if not parent_url:
            return resolved

        try:
            response = requests.get(
                parent_url,
                headers=DEFAULT_REQUEST_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.warning(
                "resolve_butchart_calendar_assets(): Failed fetching parent page %s: %s. Using fallback URLs.",
                parent_url,
                exc,
            )
            return resolved

        html = response.text or ""
        anchor_pattern = re.compile(
            r'<a[^>]+href=["\'](?P<href>[^"\']+)["\'][^>]*>(?P<text>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        anchor_matches: list[tuple[str, str]] = []
        for match in anchor_pattern.finditer(html):
            href = match.group("href")
            text = re.sub(r"<[^>]+>", " ", match.group("text"))
            anchor_matches.append((href, " ".join(text.split())))

        prioritized_pdf_href = ""
        fallback_pdf_href = ""
        for href, text in anchor_matches:
            if ".pdf" not in href.lower():
                continue
            if not fallback_pdf_href:
                fallback_pdf_href = href
            if DOWNLOAD_LINK_TEXT_PATTERN.search(text):
                prioritized_pdf_href = href
                break

        pdf_href = prioritized_pdf_href or fallback_pdf_href
        if pdf_href:
            resolved["pdf_url"] = urljoin(parent_url, pdf_href)

        if source == BUTCHART_SOURCE_NAME:
            image_match = re.search(
                r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\'](?P<content>[^"\']+)["\']',
                html,
                re.IGNORECASE,
            )
            if not image_match:
                image_match = re.search(
                    r'(?P<image>https?://[^"\']+Entertainment-calendar[^"\']+\.(?:png|jpg|jpeg))',
                    html,
                    re.IGNORECASE,
                )
            if image_match:
                image_value = image_match.groupdict().get("content") or image_match.groupdict().get("image")
                if image_value:
                    resolved["image_url"] = urljoin(parent_url, image_value)

        return resolved

    def resolve_source_urls(self, row: pd.Series) -> dict[str, str | None]:
        """Normalize source URLs, resolving the live PDF from the parent page when possible."""
        source = str(row.get("source", "") or "")
        parent_url = str(row.get("parent_url", "") or "")
        pdf_url = str(row.get("pdf_url", "") or "")
        keywords = row.get("keywords", None)
        resolved: dict[str, str | None] = {
            "source": source,
            "parent_url": parent_url,
            "pdf_url": pdf_url,
            "keywords": keywords,
            "image_url": None,
        }

        if not parent_url:
            return resolved

        resolved_assets = self.resolve_parent_page_assets(
            source=source,
            parent_url=parent_url,
            fallback_pdf_url=pdf_url,
        )
        resolved["parent_url"] = resolved_assets["parent_url"] or parent_url
        resolved["pdf_url"] = resolved_assets["pdf_url"] or pdf_url
        resolved["image_url"] = resolved_assets["image_url"] or None
        if resolved["pdf_url"] != pdf_url:
            logging.info(
                "resolve_source_urls(): Resolved current PDF URL for '%s' from parent page: %s",
                source,
                resolved["pdf_url"],
            )
        return resolved


    def read_write_pdf(self) -> pd.DataFrame:
        file_name = os.path.basename(__file__)
        start_df = self.db.count_events_urls_start(file_name)

        sources = pd.read_csv(self.csv_path, dtype=str)
        all_events = []

        for idx, row in sources.iterrows():
            resolved_source = self.resolve_source_urls(row)
            source = str(resolved_source.get('source', '') or '')
            pdf_url = str(resolved_source.get('pdf_url', '') or '')
            parent_url = str(resolved_source.get('parent_url', '') or '')
            keywords = resolved_source.get('keywords', None)
            parser_context = {
                "source": source,
                "pdf_url": pdf_url,
                "parent_url": parent_url,
                "image_url": resolved_source.get("image_url"),
            }

            logging.info(f"read_write_pdf(): [{idx}] source={source}, pdf_url={pdf_url}")

            is_active, active_reason = self.is_pdf_source_active(row)
            if not is_active:
                logging.info(
                    "read_write_pdf(): Skipping inactive seasonal PDF source '%s' (%s): %s",
                    source,
                    active_reason,
                    pdf_url,
                )
                continue

            if not pdf_url:
                logging.warning(
                    "read_write_pdf(): Could not resolve a PDF URL for '%s' from parent page: %s",
                    source,
                    parent_url,
                )
                continue

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
            try:
                resp = requests.get(pdf_url, headers=DEFAULT_REQUEST_HEADERS, timeout=30)
            except requests.RequestException as exc:
                logging.warning(
                    "read_write_pdf(): Request failed for '%s' PDF %s: %s",
                    source,
                    pdf_url,
                    exc,
                )
                continue
            if resp.status_code == 404:
                logging.warning(f"read_write_pdf(): PDF not found (404) for '{source}': {pdf_url}")
                continue
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                logging.warning(
                    "read_write_pdf(): HTTP failure for '%s' PDF %s: %s",
                    source,
                    pdf_url,
                    exc,
                )
                continue
            pdf_file = io.BytesIO(resp.content)

            try:
                df = parser(pdf_file, parser_context=parser_context)
            except Exception as exc:
                logging.warning(
                    "read_write_pdf(): Parser for '%s' failed for PDF %s: %s",
                    source,
                    pdf_url,
                    exc,
                )
                continue
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
def parse_victoria_summer_music(pdf_file, parser_context: dict[str, str | None] | None = None) -> pd.DataFrame:
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
def parse_butchart_gardens_concerts(
    pdf_file,
    parser_context: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    logging.info("Parsing Butchart Gardens concerts PDF…")
    text = dump_pdf_text(pdf_file)
    llm_handler = get_llm_handler()
    parser_context = parser_context or {}
    image_url = (
        parser_context.get("image_url")
        or None
    )
    pdf_url = str(
        parser_context.get("pdf_url")
        or parser_context.get("parent_url")
        or ""
    )
    prompt, schema_type = llm_handler.generate_prompt(pdf_file, text, 'images')
    if len(prompt) > config['crawling']['prompt_max_length']:
            logging.warning(
                "parse_butchart_gardens_concerts(): Prompt for URL %s exceeds maximum length. Skipping LLM query.",
                pdf_url,
            )
            return None
    
    try:
        llm_response = llm_handler.query_openai(
            prompt=prompt,
            model=config['llm']['openai_model'],
            image_url=image_url,
            schema_type=schema_type
        )
    except Exception as exc:
        logging.warning(
            "parse_butchart_gardens_concerts(): LLM query failed for %s: %s",
            pdf_url,
            exc,
        )
        return None

    if not llm_response:
        logging.error("No response from LLM for Butchart parser.")
        return None

    parsed = llm_handler.extract_and_parse_json(llm_response,
                                                pdf_url, schema_type)
    if not parsed:
        logging.error("Failed to parse JSON from LLM response.")
        return None

    df = pd.DataFrame(parsed)
    df['dance_style'] = 'ballroom, swing, wcs, west coast swing'
    df['event_type']  = 'social dance, live music'
    df['url']         = pdf_url
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
