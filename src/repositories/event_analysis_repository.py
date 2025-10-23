"""
Event Analysis Repository for reporting and analysis operations.

This module consolidates event analysis, reporting, and synchronization operations.
Previously scattered across DatabaseHandler, it handles:
- Location synchronization with address table
- Orphaned reference cleanup for data integrity
- Event and URL statistics collection
- Image/PDF event existence validation

This repository focuses on reporting/analysis and data integrity operations (Priority 4),
complementing EventRepository (CRUD) and EventManagementRepository (data quality).
"""

from typing import Optional, Dict, Any
import logging
import pandas as pd
from datetime import datetime


class EventAnalysisRepository:
    """
    Repository for managing event analysis, reporting, and synchronization operations.

    Consolidates operations for:
    - Syncing event locations with canonical address references
    - Cleaning orphaned references in related tables
    - Collecting event and URL statistics
    - Validating event existence for external references

    Key responsibilities:
    - Location data synchronization
    - Referential integrity maintenance
    - Statistics collection and reporting
    - Event validation and tracking
    """

    def __init__(self, db_handler):
        """
        Initialize EventAnalysisRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)

    def sync_event_locations_with_address_table(self) -> int:
        """
        Updates all events so that location matches the canonical full_address from the address table.

        Ensures consistency between event location fields and the authoritative address table.
        This prevents divergence of location data across tables.

        Returns:
            int: The number of events updated to use canonical full_address.

        Raises:
            Exception: If an error occurs during the synchronization.
        """
        try:
            sync_query = """
                UPDATE events e
                SET location = a.full_address
                FROM address a
                WHERE e.address_id = a.address_id
                AND (e.location IS DISTINCT FROM a.full_address);
            """
            affected_rows = self.db.execute_query(sync_query) or 0
            self.logger.info(
                f"sync_event_locations_with_address_table(): Updated {affected_rows} "
                "events to use canonical full_address."
            )
            return affected_rows
        except Exception as e:
            self.logger.error(
                "sync_event_locations_with_address_table(): Failed to sync locations: %s", e
            )
            raise

    def clean_orphaned_references(self) -> Dict[str, int]:
        """
        Clean up orphaned references in related tables to maintain referential integrity.

        Removes records that reference non-existent addresses:
        1. raw_locations records with invalid address_id
        2. events records with invalid address_id
        3. events_history records with invalid address_id (critical for preventing corruption)

        Returns:
            dict: Count of cleaned up records by table with keys:
                - raw_locations: Count of orphaned location records removed
                - events: Count of orphaned event records removed
                - events_history: Count of orphaned history records removed
                - total: Total records cleaned

        Raises:
            Exception: If an error occurs during the cleanup process.
        """
        cleanup_counts = {
            'raw_locations': 0,
            'events': 0,
            'events_history': 0,
            'total': 0
        }

        try:
            # Clean up orphaned raw_locations (references non-existent addresses)
            cleanup_raw_locations_sql = """
            DELETE FROM raw_locations
            WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            raw_locations_count = self.db.execute_query(cleanup_raw_locations_sql) or 0
            cleanup_counts['raw_locations'] = raw_locations_count
            self.logger.info(
                "clean_orphaned_references: Cleaned %d orphaned raw_locations records.",
                raw_locations_count
            )

            # Clean up events with non-existent address_ids (should be rare)
            cleanup_events_sql = """
            DELETE FROM events
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            events_count = self.db.execute_query(cleanup_events_sql) or 0
            cleanup_counts['events'] = events_count
            self.logger.info(
                "clean_orphaned_references: Cleaned %d orphaned events records.",
                events_count
            )

            # Clean up events_history with non-existent address_ids
            # (Critical for preventing data corruption in history table)
            cleanup_events_history_sql = """
            DELETE FROM events_history
            WHERE address_id IS NOT NULL
              AND address_id NOT IN (SELECT address_id FROM address);
            """
            events_history_count = self.db.execute_query(cleanup_events_history_sql) or 0
            cleanup_counts['events_history'] = events_history_count
            self.logger.info(
                "clean_orphaned_references: Cleaned %d orphaned events_history records.",
                events_history_count
            )

            cleanup_counts['total'] = (
                raw_locations_count + events_count + events_history_count
            )
            self.logger.info(
                "clean_orphaned_references: Total cleaned up records: %d",
                cleanup_counts['total']
            )
            return cleanup_counts

        except Exception as e:
            self.logger.error("clean_orphaned_references: Failed to clean references: %s", e)
            raise

    def count_events_urls_start(self, file_name: str) -> pd.DataFrame:
        """
        Counts the number of events and distinct URLs at the start of a process.

        Creates a baseline snapshot for comparison at process end, allowing calculation
        of how many new events/URLs were discovered during execution.

        Args:
            file_name (str): The name of the .py file initiating the count (for logging).

        Returns:
            pd.DataFrame: A DataFrame containing:
                - file_name: The originating file name
                - start_time: Timestamp when count was initiated
                - events_count_start: Count of events at start
                - urls_count_start: Count of distinct URLs at start
        """
        try:
            # Create DataFrame with source file name
            file_name_df = pd.DataFrame([[file_name]], columns=["file_name"])

            # Capture start timestamp
            start_time = datetime.now()
            start_time_df = pd.DataFrame([[start_time]], columns=["start_time"])

            # Count events in database
            events_sql = "SELECT COUNT(*) as events_count_start FROM events"
            events_count_df = pd.read_sql(events_sql, self.db.conn)

            # Count distinct URLs in database
            urls_sql = "SELECT COUNT(DISTINCT link) as urls_count_start FROM urls"
            urls_count_df = pd.read_sql(urls_sql, self.db.conn)

            # Concatenate all columns into result DataFrame
            start_df = pd.concat(
                [file_name_df, start_time_df, events_count_df, urls_count_df],
                axis=1
            )

            self.logger.info(
                "count_events_urls_start: Captured baseline - %d events, %d distinct URLs",
                start_df['events_count_start'].iloc[0],
                start_df['urls_count_start'].iloc[0]
            )
            return start_df

        except Exception as e:
            self.logger.error("count_events_urls_start: Failed to count events/URLs: %s", e)
            raise

    def count_events_urls_end(
        self, start_df: pd.DataFrame, file_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Counts events and URLs at process end, calculates delta, and writes statistics.

        Compares end counts with start counts to determine net change in events/URLs.
        Calculates elapsed time for performance tracking.

        Args:
            start_df (pd.DataFrame): DataFrame from count_events_urls_start() with baseline counts.
            file_name (str): The name of the .py file for output file naming.

        Returns:
            pd.DataFrame: A DataFrame containing all statistics:
                - file_name: Source file
                - start_time: Process start timestamp
                - events_count_start: Baseline event count
                - urls_count_start: Baseline URL count
                - events_count_end: Final event count
                - urls_count_end: Final URL count
                - new_events_in_db: Difference (end - start)
                - new_urls_in_db: Difference (end - start)
                - time_stamp: Process end timestamp
                - elapsed_time: Duration from start to end

        Raises:
            Exception: If an error occurs during counting or file writing.
        """
        try:
            # Count events at end
            events_sql = "SELECT COUNT(*) as events_count_end FROM events"
            events_count_df = pd.read_sql(events_sql, self.db.conn)

            # Count distinct URLs at end
            urls_sql = "SELECT COUNT(DISTINCT link) as urls_count_end FROM urls"
            urls_count_df = pd.read_sql(urls_sql, self.db.conn)

            # Combine all data into results DataFrame
            results_df = pd.concat([start_df, events_count_df, urls_count_df], axis=1)

            # Calculate deltas and elapsed time
            results_df['new_events_in_db'] = (
                results_df['events_count_end'] - results_df['events_count_start']
            )
            results_df['new_urls_in_db'] = (
                results_df['urls_count_end'] - results_df['urls_count_start']
            )
            results_df['time_stamp'] = datetime.now()
            results_df['elapsed_time'] = (
                results_df['time_stamp'] - results_df['start_time']
            )

            # Log summary
            self.logger.info(
                "count_events_urls_end: Process complete - "
                "%d new events, %d new URLs, elapsed: %s",
                results_df['new_events_in_db'].iloc[0],
                results_df['new_urls_in_db'].iloc[0],
                results_df['elapsed_time'].iloc[0]
            )

            # Write to CSV (only on local, not on Render ephemeral filesystem)
            import os
            if os.getenv('RENDER') != 'true':
                try:
                    output_file = self.db.config['output']['events_urls_diff']
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    if not os.path.isfile(output_file):
                        results_df.to_csv(output_file, index=False)
                    else:
                        results_df.to_csv(output_file, mode='a', header=False, index=False)

                    self.logger.info(
                        "count_events_urls_end: Wrote statistics to %s", output_file
                    )
                except Exception as write_error:
                    self.logger.warning(
                        "count_events_urls_end: Failed to write CSV: %s", write_error
                    )
            else:
                self.logger.info(
                    "count_events_urls_end: Skipping CSV write on Render (ephemeral filesystem)"
                )

            return results_df

        except Exception as e:
            self.logger.error("count_events_urls_end: Failed to count/analyze events: %s", e)
            raise

    def check_image_events_exist(self, image_url: str) -> bool:
        """
        Check if events exist for a given image URL.

        Currently disabled and always returns False to force re-scraping of images/PDFs.
        This ensures fresh, accurate data with correct address normalization on every run.

        Previous implementation had data corruption issues when copying from events_history,
        so this safety mechanism prevents that corruption.

        Args:
            image_url (str): The URL of the image/PDF to check for associated events.

        Returns:
            bool: Always returns False to force re-scraping (corruption prevention).

        Note:
            This method is intentionally simple because the previous implementation
            (checking history and copying events) caused data corruption issues with
            address_id references. The safer approach is to always re-scrape.
        """
        self.logger.info(
            "check_image_events_exist(): Forcing re-scrape for URL: %s (corruption prevention)",
            image_url
        )
        return False
