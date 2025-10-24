"""
Database utilities for event and URL data writing.

This module consolidates database write patterns used across multiple scrapers
(fb.py, scraper.py, rd_ext.py, images.py, ebs.py) including event writing,
URL tracking, and batch database operations.

Classes:
    DBWriter: Unified database write operations for scrapers

Key responsibilities:
    - Event data normalization for database insertion
    - URL tracking and write operations
    - Batch database inserts
    - Database operation error handling
"""

import logging
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


class DBWriter:
    """
    Unified database write operations for web scrapers.

    Consolidates database write patterns that are repeated across all scrapers,
    providing consistent event and URL data writing.
    """

    def __init__(self, db_handler, logger: Optional[logging.Logger] = None):
        """
        Initialize DBWriter.

        Args:
            db_handler: DatabaseHandler instance for database operations
            logger (logging.Logger, optional): Logger instance
        """
        self.db = db_handler
        self.logger = logger or logging.getLogger(__name__)

    @property
    def db_handler(self):
        """
        Backward compatibility property for accessing the DatabaseHandler.

        Some code (like read_pdfs_v2.py) expects self.db_writer.db_handler,
        but DBWriter stores it as self.db. This property provides the alias.

        Returns:
            DatabaseHandler: The underlying database handler instance
        """
        return self.db

    def normalize_event_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize event data for database insertion.

        Ensures all required fields are present and properly formatted.

        Args:
            event (dict): Event data dictionary

        Returns:
            dict: Normalized event data
        """
        try:
            normalized = {
                'event_name': event.get('event_name', 'Unknown Event'),
                'dance_style': event.get('dance_style', ''),
                'description': event.get('description', ''),
                'day_of_week': event.get('day_of_week', ''),
                'start_date': event.get('start_date'),
                'end_date': event.get('end_date'),
                'start_time': event.get('start_time'),
                'end_time': event.get('end_time'),
                'location': event.get('location', ''),
                'price': event.get('price', ''),
                'source': event.get('source', 'unknown'),
                'url': event.get('url', ''),
                'event_type': event.get('event_type', ''),
                'address_id': event.get('address_id', 0),
                'time_stamp': event.get('time_stamp', datetime.now()),
            }

            # Remove None values except for optional fields
            optional_fields = {'address_id', 'end_date', 'end_time'}
            filtered = {k: v for k, v in normalized.items()
                       if v is not None or k in optional_fields}

            return filtered

        except Exception as e:
            self.logger.error(f"Failed to normalize event data: {e}")
            return normalized

    def normalize_events_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize a batch of events for database insertion.

        Args:
            events (list): List of event dictionaries

        Returns:
            list: List of normalized event dictionaries
        """
        return [self.normalize_event_data(event) for event in events]

    def normalize_dataframe_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame for database insertion.

        Ensures all required columns are present and properly formatted.

        Args:
            df (pd.DataFrame): Event DataFrame

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        try:
            # Required columns
            required_cols = [
                'event_name', 'dance_style', 'description', 'day_of_week',
                'start_date', 'end_date', 'start_time', 'end_time',
                'location', 'price', 'source', 'url', 'event_type', 'address_id', 'time_stamp'
            ]

            # Add missing columns with default values
            for col in required_cols:
                if col not in df.columns:
                    if col in ['address_id']:
                        df[col] = 0
                    elif col in ['time_stamp']:
                        df[col] = datetime.now()
                    else:
                        df[col] = ''

            # Select and order columns
            df = df[required_cols]

            # Fill NaN values
            df = df.fillna({
                'event_name': 'Unknown Event',
                'dance_style': '',
                'description': '',
                'day_of_week': '',
                'location': '',
                'price': '',
                'source': 'unknown',
                'url': '',
                'event_type': '',
                'address_id': 0,
                'time_stamp': datetime.now(),
            })

            return df

        except Exception as e:
            self.logger.error(f"Failed to normalize DataFrame: {e}")
            return df

    def write_event_to_db(self, event: Dict[str, Any]) -> bool:
        """
        Write a single event to the database.

        Args:
            event (dict): Event data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            normalized = self.normalize_event_data(event)
            df = pd.DataFrame([normalized])
            df.to_sql('events', self.db.conn, if_exists='append', index=False)
            return True

        except Exception as e:
            self.logger.error(f"Failed to write event to database: {e}")
            return False

    def write_events_to_db(self, events: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Write multiple events to the database.

        Args:
            events (list): List of event dictionaries

        Returns:
            Tuple[int, int]: (success_count, fail_count)
        """
        if not events:
            self.logger.warning("No events to write")
            return (0, 0)

        try:
            normalized = self.normalize_events_batch(events)
            df = pd.DataFrame(normalized)
            df = self.normalize_dataframe_for_insert(df)

            df.to_sql('events', self.db.conn, if_exists='append', index=False, method='multi')
            self.logger.info(f"Successfully wrote {len(df)} events to database")
            return (len(df), 0)

        except Exception as e:
            self.logger.error(f"Failed to write events batch: {e}")
            return (0, len(events))

    def write_dataframe_to_db(self, df: pd.DataFrame, table: str = 'events') -> bool:
        """
        Write a DataFrame directly to the database.

        Args:
            df (pd.DataFrame): DataFrame to write
            table (str): Target table name

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if df.empty:
                self.logger.warning(f"DataFrame is empty, skipping write to {table}")
                return False

            if table == 'events':
                df = self.normalize_dataframe_for_insert(df)

            df.to_sql(table, self.db.conn, if_exists='append', index=False, method='multi')
            self.logger.info(f"Successfully wrote {len(df)} rows to {table}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write DataFrame to {table}: {e}")
            return False

    def write_url_to_db(self, url_data: Tuple) -> bool:
        """
        Write URL tracking data to database.

        Args:
            url_data (tuple): (url, parent_url, source, keywords, success, retry_count, timestamp)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not hasattr(self.db, 'url_repo'):
                self.logger.error("URL repository not available")
                return False

            result = self.db.url_repo.write_url_to_db(url_data)
            return result

        except Exception as e:
            self.logger.error(f"Failed to write URL to database: {e}")
            return False

    def write_urls_batch(self, urls: List[Tuple]) -> Tuple[int, int]:
        """
        Write multiple URL tracking entries to database.

        Args:
            urls (list): List of URL data tuples

        Returns:
            Tuple[int, int]: (success_count, fail_count)
        """
        success_count = 0
        fail_count = 0

        for url_data in urls:
            if self.write_url_to_db(url_data):
                success_count += 1
            else:
                fail_count += 1

        return (success_count, fail_count)

    def create_event_tuple_for_insert(self, event: Dict[str, Any]) -> tuple:
        """
        Create database-ready tuple from event dictionary.

        Used for bulk inserts where tuple format is more efficient.

        Args:
            event (dict): Event data

        Returns:
            tuple: Database-ready tuple in standard order
        """
        normalized = self.normalize_event_data(event)

        return (
            normalized.get('event_name'),
            normalized.get('dance_style'),
            normalized.get('description'),
            normalized.get('day_of_week'),
            normalized.get('start_date'),
            normalized.get('end_date'),
            normalized.get('start_time'),
            normalized.get('end_time'),
            normalized.get('location'),
            normalized.get('price'),
            normalized.get('source'),
            normalized.get('url'),
            normalized.get('event_type'),
            normalized.get('address_id', 0),
            normalized.get('time_stamp'),
        )

    def get_event_insert_sql(self) -> str:
        """
        Get SQL statement for event insertion.

        Returns:
            str: SQL INSERT statement template
        """
        columns = [
            'event_name', 'dance_style', 'description', 'day_of_week',
            'start_date', 'end_date', 'start_time', 'end_time',
            'location', 'price', 'source', 'url', 'event_type', 'address_id', 'time_stamp'
        ]
        placeholders = ', '.join(['?' for _ in columns])
        columns_str = ', '.join(columns)

        return f"""
            INSERT INTO events ({columns_str})
            VALUES ({placeholders})
        """

    def validate_event_data(self, event: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate event data before insertion.

        Args:
            event (dict): Event data to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Check required fields
            if not event.get('event_name'):
                return (False, "Missing required field: event_name")

            if not event.get('start_date'):
                return (False, "Missing required field: start_date")

            # Validate date/time formats if present
            if event.get('start_date') and not isinstance(event['start_date'], (str, datetime)):
                return (False, "Invalid start_date format")

            return (True, None)

        except Exception as e:
            return (False, f"Validation error: {e}")

    def get_insert_statistics(self, success_count: int, total_count: int) -> dict:
        """
        Generate statistics for insert operation.

        Args:
            success_count (int): Number of successfully inserted rows
            total_count (int): Total number of rows attempted

        Returns:
            dict: Statistics dictionary
        """
        fail_count = total_count - success_count
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0

        return {
            'total_attempted': total_count,
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': f"{success_rate:.1f}%",
        }
