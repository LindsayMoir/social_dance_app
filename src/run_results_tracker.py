#!/usr/bin/env python3
"""
run_results_tracker.py - Run Results Tracking Utility

Tracks and records execution statistics for scraper runs, writing results to the
run_results database table. Each scraper execution writes a row containing:
- File name and execution timestamps
- Event and URL counts (start/end/delta)
- Execution elapsed time
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class RunResultsTracker:
    """Tracks and writes execution statistics for scraper runs."""

    def __init__(self, file_name: str, db_handler=None):
        """Initialize run results tracker.

        Args:
            file_name: Name of the script/file being executed
            db_handler: Database handler instance
        """
        self.logger = logging.getLogger(__name__)
        self.file_name = file_name
        self.db_handler = db_handler

        # Start tracking
        self.start_time_df = None
        self.events_count_start = 0
        self.urls_count_start = 0
        self.events_count_end = 0
        self.urls_count_end = 0

    def initialize(self, events_count: int = 0, urls_count: int = 0) -> None:
        """Initialize tracking at the start of execution.

        Args:
            events_count: Initial event count in database
            urls_count: Initial URL count in database
        """
        self.start_time_df = datetime.now().isoformat()
        self.events_count_start = events_count
        self.urls_count_start = urls_count

    def finalize(self, events_count: int = 0, urls_count: int = 0) -> None:
        """Finalize tracking at the end of execution.

        Args:
            events_count: Final event count in database
            urls_count: Final URL count in database
        """
        self.events_count_end = events_count
        self.urls_count_end = urls_count

    def write_results(self, elapsed_time: str = None) -> bool:
        """Write run results to database.

        Args:
            elapsed_time: Duration of execution as string (e.g., "0 days 00:30:45.123456")

        Returns:
            True if successfully written, False otherwise
        """
        if not self.db_handler:
            self.logger.warning("RunResultsTracker: Database handler not configured")
            return False

        try:
            # Calculate deltas
            new_events_in_db = self.events_count_end - self.events_count_start
            new_urls_in_db = self.urls_count_end - self.urls_count_start

            # Prepare data for insertion
            run_result_data = {
                'file_name': [self.file_name],
                'start_time_df': [self.start_time_df],
                'events_count_start': [self.events_count_start],
                'urls_count_start': [self.urls_count_start],
                'events_count_end': [self.events_count_end],
                'urls_count_end': [self.urls_count_end],
                'new_events_in_db': [new_events_in_db],
                'new_urls_in_db': [new_urls_in_db],
                'time_stamp': [datetime.now()],
                'elapsed_time': [elapsed_time]
            }

            # Create DataFrame and write to database
            df = pd.DataFrame(run_result_data)

            try:
                # Try using SQLAlchemy engine if available
                connection = self.db_handler.get_db_connection()
                df.to_sql('run_results', connection, if_exists='append', index=False)
                self.logger.info(
                    f"RunResultsTracker: Successfully wrote run results for {self.file_name}"
                )
                return True
            except Exception as e:
                self.logger.error(f"RunResultsTracker: Error writing to database: {e}")
                return False

        except Exception as e:
            self.logger.error(f"RunResultsTracker: Error preparing run results: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked statistics.

        Returns:
            Dictionary containing tracking summary
        """
        return {
            'file_name': self.file_name,
            'start_time': self.start_time_df,
            'events_count_start': self.events_count_start,
            'urls_count_start': self.urls_count_start,
            'events_count_end': self.events_count_end,
            'urls_count_end': self.urls_count_end,
            'new_events_in_db': self.events_count_end - self.events_count_start,
            'new_urls_in_db': self.urls_count_end - self.urls_count_start,
        }


def get_database_counts(db_handler) -> tuple:
    """Get current event and URL counts from database.

    Args:
        db_handler: Database handler instance

    Returns:
        Tuple of (events_count, urls_count)
    """
    try:
        if not db_handler:
            return 0, 0

        # Get event count
        events_query = "SELECT COUNT(*) FROM events"
        events_result = db_handler.execute_query(events_query)
        events_count = events_result[0][0] if events_result else 0

        # Get URL count
        urls_query = "SELECT COUNT(*) FROM urls"
        urls_result = db_handler.execute_query(urls_query)
        urls_count = urls_result[0][0] if urls_result else 0

        return events_count, urls_count
    except Exception as e:
        logging.error(f"get_database_counts: Error querying database: {e}")
        return 0, 0
