"""
Event repository for centralized event management and CRUD operations.

This module consolidates all event-related database operations previously
scattered throughout DatabaseHandler. It handles event writing, updating,
deletion, and retrieval with comprehensive data validation and processing.
"""

from typing import Optional, Dict, Any, List
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fuzzy_utils import FuzzyMatcher


class EventRepository:
    """
    Repository for managing event operations in the database.

    Consolidates event validation, writing, updating, and deletion logic
    previously scattered across DatabaseHandler.

    Key responsibilities:
    - Event writing with data validation and cleaning
    - Event updating with field overlaying
    - Event deletion (single, multiple, or by criteria)
    - Event retrieval and DataFrame management
    - Event data processing (datetime conversion, field cleaning)
    """

    def __init__(self, db_handler):
        """
        Initialize EventRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)

    def write_events_to_db(self, df: pd.DataFrame, url: str, parent_url: str,
                          source: str, keywords) -> bool:
        """
        Processes and writes event data to the 'events' table in the database.

        This method performs several data cleaning and transformation steps on the input DataFrame,
        including renaming columns, handling missing values, formatting dates and times, and removing
        outdated or incomplete events.

        Args:
            df (pd.DataFrame): DataFrame containing raw event data to be processed and stored.
            url (str): The URL from which the events data was sourced.
            parent_url (str): The parent URL, if applicable.
            source (str): The source identifier for the event data.
            keywords (str or list): Keywords or dance styles associated with the events.

        Returns:
            bool: True if events were successfully written, False otherwise.
        """
        try:
            url = '' if pd.isna(url) else str(url)
            parent_url = '' if pd.isna(parent_url) else str(parent_url)

            # Handle Google Calendar format
            if 'calendar' in url or 'calendar' in parent_url:
                df = self._rename_google_calendar_columns(df)
                df['dance_style'] = ', '.join(keywords) if isinstance(keywords, list) else keywords

            # Set source and URL
            source = source if source else (url.split('.')[-2] if url and '.' in url and len(url.split('.')) >= 2 else 'unknown')
            df['source'] = df.get('source', pd.Series([''] * len(df))).replace('', source).fillna(source)
            df['url'] = df.get('url', pd.Series([''] * len(df))).replace('', url).fillna(url)

            # Convert datetime fields
            self._convert_datetime_fields(df)

            # Ensure price column exists
            if 'price' not in df.columns:
                self.logger.warning("write_events_to_db: 'price' column is missing. Filling with empty string.")
                df['price'] = ''

            # Add timestamp
            df['time_stamp'] = datetime.now()

            # Clean day_of_week field
            df = self._clean_day_of_week_field(df)

            # Basic location cleanup
            df = self.db.clean_up_address_basic(df)

            # Resolve structured addresses
            updated_rows = []
            for i, row in df.iterrows():
                event_dict = row.to_dict()
                event_dict = self.db.address_data_repo.normalize_nulls(event_dict)
                updated_event = self.db.address_resolution_repo.process_event_address(event_dict)
                for key in ["address_id", "location"]:
                    if key in updated_event:
                        df.at[i, key] = updated_event[key]
                updated_rows.append(updated_event)

            self.logger.info(f"write_events_to_db: Address processing complete for {len(updated_rows)} events.")

            # Filter old or incomplete events
            df = self._filter_events(df)

            if df.empty:
                self.logger.info("write_events_to_db: No events remain after filtering, skipping write.")
                self.db.url_repo.write_url_to_db([url, parent_url, source, keywords, False, 1, datetime.now()])
                return False

            # Write debug CSV (only locally, not on Render)
            if os.getenv('RENDER') != 'true':
                os.makedirs('output', exist_ok=True)
                df.to_csv('output/cleaned_events.csv', index=False)

            self.logger.info(f"write_events_to_db: Number of events to write: {len(df)}")

            # Write to database
            df.to_sql('events', self.db.conn, if_exists='append', index=False, method='multi')
            self.db.url_repo.write_url_to_db([url, parent_url, source, keywords, True, 1, datetime.now()])
            self.logger.info("write_events_to_db: Events data written to the 'events' table.")
            return True

        except Exception as e:
            self.logger.error(f"write_events_to_db: Failed to write events: {e}")
            return False

    def update_event(self, event_identifier: Dict[str, Any], new_data: Dict[str, Any],
                     best_url: str) -> bool:
        """
        Update an existing event in the database by overlaying new data.

        Args:
            event_identifier (dict): Dictionary specifying criteria to identify the event.
                Example: {'event_name': ..., 'start_date': ..., 'start_time': ...}
            new_data (dict): Dictionary containing new values to update. Only non-empty
                values will overwrite existing fields.
            best_url (str): The URL to set as the event's 'url' field.

        Returns:
            bool: True if the event was found and updated, False otherwise.
        """
        try:
            select_query = """
            SELECT * FROM events
            WHERE event_name = :event_name
            AND start_date = :start_date
            AND start_time = :start_time
            """
            result = self.db.execute_query(select_query, event_identifier)
            existing_row = result[0] if result else None

            if not existing_row:
                self.logger.error("update_event: No matching event found for identifier: %s", event_identifier)
                return False

            # Overlay new data onto existing row
            updated_data = dict(existing_row)
            for col, new_val in new_data.items():
                if new_val not in [None, '', pd.NaT]:
                    updated_data[col] = new_val

            # Update URL
            updated_data['url'] = best_url

            # Build update query
            update_cols = [col for col in updated_data.keys() if col != 'event_id']
            set_clause = ", ".join([f"{col} = :{col}" for col in update_cols])
            update_query = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
            updated_data['event_id'] = existing_row['event_id']

            self.db.execute_query(update_query, updated_data)
            self.logger.info("update_event: Updated event %s", updated_data)
            return True

        except Exception as e:
            self.logger.error(f"update_event: Failed to update event: {e}")
            return False

    def delete_event(self, url: str, event_name: str, start_date) -> bool:
        """
        Delete an event from the 'events' table based on event name and start date.

        Args:
            url (str): The URL of the event (for logging purposes).
            event_name (str): The name of the event to delete.
            start_date: The start date of the event.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            self.logger.info("delete_event: Deleting event with URL: %s, Event Name: %s, Start Date: %s",
                           url, event_name, start_date)

            delete_event_query = """
                DELETE FROM events
                WHERE event_name = :event_name
                  AND start_date = :start_date;
            """
            params = {'event_name': event_name, 'start_date': start_date}
            self.db.execute_query(delete_event_query, params)
            self.logger.info("delete_event: Deleted event from 'events' table.")
            return True

        except Exception as e:
            self.logger.error("delete_event: Failed to delete event: %s", e)
            return False

    def delete_event_with_event_id(self, event_id: int) -> bool:
        """
        Delete an event from the 'events' table based on event_id.

        Args:
            event_id (int): The unique identifier of the event to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE event_id = :event_id;
            """
            params = {'event_id': event_id}
            self.db.execute_query(delete_query, params)
            self.logger.info("delete_event_with_event_id: Deleted event with event_id %d successfully.", event_id)
            return True

        except Exception as e:
            self.logger.error("delete_event_with_event_id: Failed to delete event with event_id %d: %s", event_id, e)
            return False

    def delete_multiple_events(self, event_ids: List[int]) -> bool:
        """
        Delete multiple events from the 'events' table based on a list of event IDs.

        Args:
            event_ids (list): A list of event IDs (int) to be deleted.

        Returns:
            bool: True if all specified events were successfully deleted, False otherwise.
        """
        if not event_ids:
            self.logger.warning("delete_multiple_events: No event_ids provided for deletion.")
            return False

        success_count = 0
        for event_id in event_ids:
            try:
                if self.delete_event_with_event_id(event_id):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"delete_multiple_events: Failed to delete event_id {event_id}: {e}")

        self.logger.info(f"delete_multiple_events: Successfully deleted {success_count}/{len(event_ids)} events.")
        return success_count == len(event_ids)

    def fetch_events_dataframe(self) -> pd.DataFrame:
        """
        Fetch all events from the database and return as a sorted DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all events, sorted by start_date and start_time.
        """
        try:
            query = "SELECT * FROM events"
            df = pd.read_sql(query, self.db.conn)
            df.sort_values(by=['start_date', 'start_time'], inplace=True)
            self.logger.info(f"fetch_events_dataframe: Retrieved {len(df)} events from database.")
            return df

        except Exception as e:
            self.logger.error(f"fetch_events_dataframe: Failed to fetch events: {e}")
            return pd.DataFrame()

    def _rename_google_calendar_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns of a DataFrame containing Google Calendar event data.

        Args:
            df (pd.DataFrame): The input DataFrame with Google Calendar column names.

        Returns:
            pd.DataFrame: A DataFrame with columns renamed to standardized names.
        """
        return df.rename(columns={
            'URL': 'url',
            'Type_of_Event': 'event_type',
            'Name_of_the_Event': 'event_name',
            'Day_of_Week': 'day_of_week',
            'Start_Date': 'start_date',
            'End_Date': 'end_date',
            'Start_Time': 'start_time',
            'End_Time': 'end_time',
            'Price': 'price',
            'Location': 'location',
            'Description': 'description'
        })

    def _convert_datetime_fields(self, df: pd.DataFrame) -> None:
        """
        Convert datetime-related columns to appropriate date and time types.

        Converts:
        - 'start_date' and 'end_date' to datetime.date objects
        - 'start_time' and 'end_time' to datetime.time objects

        Args:
            df (pd.DataFrame): The DataFrame to convert (modified in place).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for col in ['start_date', 'end_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            for col in ['start_time', 'end_time']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
            warnings.resetwarnings()

    def _clean_day_of_week_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the day_of_week field.

        Fixes:
        - Compound values like "Friday, Saturday" -> takes first day
        - Special values like "Daily" -> converts to empty string
        - Normalizes case and whitespace

        Args:
            df (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: DataFrame with cleaned day_of_week values.
        """
        if 'day_of_week' not in df.columns:
            return df

        changes_made = 0

        for i, row in df.iterrows():
            original_value = row.get('day_of_week', '')
            if pd.isna(original_value) or not str(original_value).strip():
                continue

            day_str = str(original_value).strip()
            cleaned_value = original_value

            # Handle compound values like "Friday, Saturday"
            if ',' in day_str:
                cleaned_value = day_str.split(',')[0].strip()
                changes_made += 1
                self.logger.info(f"_clean_day_of_week_field: Changed compound day '{original_value}' to '{cleaned_value}'")

            # Handle special values like "Daily"
            elif day_str.lower() in ['daily', 'every day', 'everyday']:
                cleaned_value = ''
                changes_made += 1
                self.logger.info(f"_clean_day_of_week_field: Changed special day '{original_value}' to empty")

            # Normalize standard day names
            else:
                valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                day_lower = day_str.lower()
                if day_lower in valid_days:
                    cleaned_value = day_lower.capitalize()
                    if cleaned_value != original_value:
                        changes_made += 1

            # Update the DataFrame if value changed
            if cleaned_value != original_value:
                df.at[i, 'day_of_week'] = cleaned_value

        self.logger.info(f"_clean_day_of_week_field: Made {changes_made} changes")
        return df

    def _filter_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a DataFrame of events by removing incomplete and old events.

        Performs:
        1. Replaces empty strings in important columns with NA
        2. Drops rows where all important columns are missing
        3. Removes events older than configured threshold

        Args:
            df (pd.DataFrame): DataFrame containing event data.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        # Replace empty strings with NA
        important_cols = ['start_date', 'end_date', 'start_time', 'end_time', 'location', 'description']
        for col in important_cols:
            if col in df.columns:
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

        # Drop rows where all important columns are missing
        df = df.dropna(subset=important_cols, how='all')

        # Remove old events
        try:
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce').dt.date
            days_threshold = int(self.db.config['clean_up']['old_events'])
            cutoff_date = pd.Timestamp.now().date() - pd.Timedelta(days=days_threshold)
            df = df[df['end_date'] >= cutoff_date]
            self.logger.info(f"_filter_events: Filtered out old events (older than {cutoff_date})")
        except Exception as e:
            self.logger.warning(f"_filter_events: Could not filter by end_date: {e}")

        return df
