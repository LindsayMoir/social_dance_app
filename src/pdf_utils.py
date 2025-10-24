"""
PDF extraction and parsing utilities for web scrapers.

This module consolidates PDF download and parsing patterns used in read_pdfs.py,
including PDF text extraction, table parsing, and LLM-based event parsing.

Classes:
    PDFExtractor: Unified PDF handling and event extraction

Key responsibilities:
    - PDF download and validation
    - Text extraction from PDFs
    - Table extraction and processing
    - Parser registration and management
    - LLM-based PDF parsing
"""

import logging
import io
import requests
import pandas as pd
import pdfplumber
from typing import Optional, Callable, Dict, Any, ByteString
from datetime import datetime
from dateutil import parser as dateparser


class PDFExtractor:
    """
    Unified PDF extraction and parsing for event discovery.

    Consolidates PDF download, extraction, and parsing patterns
    used across read_pdfs.py and event processing workflows.
    """

    # Parser registry for source-specific PDF parsing
    PARSER_REGISTRY: Dict[str, Callable] = {}

    def __init__(self, llm_handler=None, logger: Optional[logging.Logger] = None):
        """
        Initialize PDFExtractor.

        Args:
            llm_handler: Optional LLMHandler for LLM-based parsing
            logger (logging.Logger, optional): Logger instance
        """
        self.llm_handler = llm_handler
        self.logger = logger or logging.getLogger(__name__)

    @classmethod
    def register_parser(cls, source_name: str):
        """
        Decorator to register PDF parsing functions for a given source.

        Args:
            source_name (str): Source identifier for the parser

        Returns:
            Callable: Decorator function
        """
        def decorator(func: Callable):
            cls.PARSER_REGISTRY[source_name] = func
            logging.info(f"Registered PDF parser for source: {source_name}")
            return func
        return decorator

    def download_pdf(self, pdf_url: str, timeout: int = 30) -> Optional[io.BytesIO]:
        """
        Download PDF from URL.

        Args:
            pdf_url (str): URL of the PDF
            timeout (int): Download timeout in seconds

        Returns:
            Optional[io.BytesIO]: PDF file object, or None if download failed
        """
        try:
            response = requests.get(pdf_url, timeout=timeout)

            if response.status_code == 404:
                self.logger.warning(f"PDF not found (404): {pdf_url}")
                return None

            response.raise_for_status()
            return io.BytesIO(response.content)

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout downloading PDF: {pdf_url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download PDF: {pdf_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading PDF: {e}")
            return None

    def extract_text_from_pdf(self, pdf_file: io.BytesIO) -> str:
        """
        Extract all text from PDF.

        Args:
            pdf_file (io.BytesIO): PDF file object

        Returns:
            str: Combined text from all pages
        """
        try:
            pages = []
            with pdfplumber.open(pdf_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ''
                    pages.append(f"----- Page {i} -----\n{text}")
            return "\n".join(pages)

        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            return ""

    def extract_tables_from_pdf(self, pdf_file: io.BytesIO) -> list:
        """
        Extract all tables from PDF.

        Args:
            pdf_file (io.BytesIO): PDF file object

        Returns:
            list: List of extracted tables
        """
        try:
            tables = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend(page_tables)
            return tables

        except Exception as e:
            self.logger.error(f"Failed to extract tables from PDF: {e}")
            return []

    def parse_pdf_with_registered_parser(self, pdf_file: io.BytesIO, source: str) -> Optional[pd.DataFrame]:
        """
        Parse PDF using registered parser for source.

        Args:
            pdf_file (io.BytesIO): PDF file object
            source (str): Source identifier to select parser

        Returns:
            Optional[pd.DataFrame]: Parsed events DataFrame, or None
        """
        parser = self.PARSER_REGISTRY.get(source)
        if not parser:
            self.logger.warning(f"No parser registered for source: {source}")
            return None

        try:
            df = parser(pdf_file)
            if df is None or df.empty:
                self.logger.warning(f"Parser returned no events for source: {source}")
                return None
            return df

        except Exception as e:
            self.logger.error(f"Parser failed for source {source}: {e}")
            return None

    async def parse_pdf_with_llm(self, pdf_file: io.BytesIO, pdf_url: str,
                                 schema_type: str = 'events') -> Optional[pd.DataFrame]:
        """
        Parse PDF using LLM for event extraction.

        Args:
            pdf_file (io.BytesIO): PDF file object
            pdf_url (str): URL of the PDF (for context)
            schema_type (str): Schema type for LLM response

        Returns:
            Optional[pd.DataFrame]: Parsed events DataFrame, or None
        """
        if not self.llm_handler:
            self.logger.warning("LLM handler not available for PDF parsing")
            return None

        try:
            # Extract PDF text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                self.logger.warning(f"No text extracted from PDF: {pdf_url}")
                return None

            # Generate LLM prompt
            prompt, prompt_schema_type = self.llm_handler.generate_prompt(
                pdf_url, text, schema_type
            )

            # Check prompt length
            max_length = self.llm_handler.config.get('crawling', {}).get('prompt_max_length', 4000)
            if len(prompt) > max_length:
                self.logger.warning(
                    f"Prompt for PDF {pdf_url} exceeds maximum length "
                    f"({len(prompt)} > {max_length}). Skipping LLM query."
                )
                return None

            # Query LLM
            llm_response = self.llm_handler.query_llm(pdf_url, prompt, prompt_schema_type)
            if not llm_response:
                self.logger.warning(f"No LLM response for PDF: {pdf_url}")
                return None

            # Parse response
            parsed_data = self.llm_handler.extract_and_parse_json(
                llm_response, "events", prompt_schema_type
            )

            if not parsed_data:
                self.logger.warning(f"Failed to parse LLM response for PDF: {pdf_url}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(parsed_data)
            self.logger.info(f"LLM parsed {len(df)} events from PDF: {pdf_url}")
            return df

        except Exception as e:
            self.logger.error(f"LLM PDF parsing failed: {e}")
            return None

    def clean_pdf_events(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Clean and standardize parsed PDF events.

        Args:
            df (pd.DataFrame): Parsed events
            source (str): Source identifier

        Returns:
            pd.DataFrame: Cleaned events
        """
        try:
            # Drop rows with missing required fields
            required_cols = ['event_name', 'start_date']
            existing_required = [col for col in required_cols if col in df.columns]
            if existing_required:
                df = df.dropna(subset=existing_required, how='any')

            if df.empty:
                self.logger.warning(f"No events remain after cleaning for source: {source}")
                return df

            # Standardize date columns
            for date_col in ['start_date', 'end_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.date

            # Standardize time columns
            for time_col in ['start_time', 'end_time']:
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce').dt.time

            # Add source if not present
            if 'source' not in df.columns:
                df['source'] = source

            # Add timestamp if not present
            if 'time_stamp' not in df.columns:
                df['time_stamp'] = datetime.now()

            return df

        except Exception as e:
            self.logger.error(f"Failed to clean PDF events: {e}")
            return pd.DataFrame()

    def parse_time_string(self, time_string: str) -> Optional[tuple]:
        """
        Parse time string into start and end times.

        Handles formats like "10:00am to 2:00pm", "2:30pm", etc.

        Args:
            time_string (str): Time string to parse

        Returns:
            Optional[tuple]: (start_time, end_time) or None if parsing failed
        """
        try:
            if not time_string or not isinstance(time_string, str):
                return None

            # Normalize input
            time_string = time_string.lower().strip()
            time_string = time_string.replace('noon', '12:00pm').replace('midnight', '12:00am')

            # Split by 'to' for time range
            parts = time_string.split(' to ')
            start_time = None
            end_time = None

            # Parse start time
            if len(parts) > 0:
                try:
                    start_time = dateparser.parse(parts[0].strip(), fuzzy=True).time()
                except:
                    pass

            # Parse end time
            if len(parts) > 1:
                try:
                    end_time = dateparser.parse(parts[1].strip(), fuzzy=True).time()
                except:
                    pass

            return (start_time, end_time) if start_time else None

        except Exception as e:
            self.logger.warning(f"Failed to parse time string '{time_string}': {e}")
            return None

    def get_registered_parsers(self) -> list:
        """
        Get list of registered source parsers.

        Returns:
            list: List of registered source names
        """
        return list(self.PARSER_REGISTRY.keys())

    def is_parser_registered(self, source: str) -> bool:
        """
        Check if parser is registered for source.

        Args:
            source (str): Source identifier

        Returns:
            bool: True if parser is registered
        """
        return source in self.PARSER_REGISTRY
