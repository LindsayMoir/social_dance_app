"""
Event Management Repository for data quality and maintenance operations.

This module consolidates event management, deduplication, and data quality operations.
Previously scattered across DatabaseHandler, it handles:
- Deletion of old events based on configured thresholds
- Filtering of low-quality or invalid events
- Deduplication of events based on similarity criteria
- Day-of-week date consistency checking and correction

This repository focuses on data quality and maintenance operations (Priority 3),
complementing EventRepository's CRUD operations (Priority 1-2).
"""

from typing import Optional, List
import logging
import pandas as pd
from datetime import datetime, timedelta


class EventManagementRepository:
    """
    Repository for managing event data quality and maintenance operations.

    Consolidates operations for:
    - Removing stale/old events
    - Filtering low-quality events (missing required fields, out-of-region)
    - Deduplicating similar events
    - Validating and correcting date/day-of-week consistency

    Key responsibilities:
    - Event deletion based on age threshold
    - Event filtering based on quality criteria
    - Event deduplication using similarity heuristics
    - Day-of-week date validation and correction
    """

    def __init__(self, db_handler):
        """
        Initialize EventManagementRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)

    def delete_old_events(self) -> int:
        """
        Deletes events that are older than the configured threshold.

        The number of days is retrieved from configuration under 'clean_up' -> 'old_events'.
        Events with an 'End_Date' earlier than (current_date - threshold_days) are deleted.

        Returns:
            int: The number of events deleted from the database.

        Raises:
            Exception: If an error occurs during the deletion process.
        """
        try:
            days = int(self.db.config['clean_up']['old_events'])
            delete_query = """
                DELETE FROM events
                WHERE end_date < CURRENT_DATE - INTERVAL '%s days';
            """ % days
            self.db.execute_query(delete_query)
            deleted_count = self.db.execute_query(delete_query) or 0
            self.logger.info(
                f"delete_old_events: Deleted {deleted_count} events older than {days} days."
            )
            return deleted_count
        except Exception as e:
            self.logger.error("delete_old_events: Failed to delete old events: %s", e)
            raise

    def delete_likely_dud_events(self) -> dict:
        """
        Deletes likely invalid or irrelevant events based on multiple quality criteria.

        Applies the following deletion rules:
        1. Events with empty source AND dance_style AND url, unless they have an address_id
        2. Events with address outside British Columbia (province_or_state != 'BC')
        3. Events with address outside Canada (country_id != 'CA')
        4. Events with empty dance_style AND url, event_type='other', AND null location/description

        Returns:
            dict: Summary of deletions with keys for each deletion rule applied:
                - rule_1_deleted: Count from rule 1
                - rule_2_deleted: Count from rule 2 (BC filtering)
                - rule_3_deleted: Count from rule 3 (CA filtering)
                - rule_4_deleted: Count from rule 4 (low quality)
                - total_deleted: Total across all rules
        """
        results = {
            'rule_1_deleted': 0,
            'rule_2_deleted': 0,
            'rule_3_deleted': 0,
            'rule_4_deleted': 0,
            'total_deleted': 0
        }

        try:
            # Rule 1: Delete events with no source, dance_style, url, and no address_id
            delete_query_1 = """
            DELETE FROM events
            WHERE source = :source
            AND dance_style = :dance_style
            AND url = :url
            AND address_id IS NULL
            RETURNING event_id;
            """
            params = {
                'source': '',
                'dance_style': '',
                'url': ''
            }
            deleted_events = self.db.execute_query(delete_query_1, params)
            deleted_count = len(deleted_events) if deleted_events else 0
            results['rule_1_deleted'] = deleted_count
            self.logger.info(
                "delete_likely_dud_events: Rule 1 - Deleted %d events with empty "
                "source, dance_style, url, and no address_id.",
                deleted_count
            )

            # Rule 2: Delete events outside British Columbia
            delete_query_2 = """
            DELETE FROM events
            WHERE address_id IN (
                SELECT address_id
                FROM address
                WHERE province_or_state IS NOT NULL
                    AND province_or_state != :province_or_state
            )
            RETURNING event_id;
            """
            params = {'province_or_state': 'BC'}
            deleted_events = self.db.execute_query(delete_query_2, params)
            deleted_count = len(deleted_events) if deleted_events else 0
            results['rule_2_deleted'] = deleted_count
            self.logger.info(
                "delete_likely_dud_events: Rule 2 - Deleted %d events "
                "outside of British Columbia (BC).",
                deleted_count
            )

            # Rule 3: Delete events not in Canada
            delete_query_3 = """
            DELETE FROM events
            WHERE address_id IN (
                SELECT address_id
                FROM address
                WHERE country_id IS NOT NULL
                    AND country_id != :country_id
            )
            RETURNING event_id;
            """
            params = {'country_id': 'CA'}
            deleted_events = self.db.execute_query(delete_query_3, params)
            deleted_count = len(deleted_events) if deleted_events else 0
            results['rule_3_deleted'] = deleted_count
            self.logger.info(
                "delete_likely_dud_events: Rule 3 - Deleted %d events "
                "that are not in Canada (CA).",
                deleted_count
            )

            # Rule 4: Delete low-quality events with minimal info
            delete_query_4 = """
            DELETE FROM events
            WHERE dance_style = :dance_style
                AND url = :url
                AND event_type = :event_type
                AND location IS NULL
                AND description IS NULL
            RETURNING event_id;
            """
            params = {
                'dance_style': '',
                'url': '',
                'event_type': 'other'
            }
            deleted_events = self.db.execute_query(delete_query_4, params)
            deleted_count = len(deleted_events) if deleted_events else 0
            results['rule_4_deleted'] = deleted_count
            self.logger.info(
                "delete_likely_dud_events: Rule 4 - Deleted %d events with empty "
                "dance_style, url, event_type='other', and null location/description.",
                deleted_count
            )

            results['total_deleted'] = sum([
                results['rule_1_deleted'],
                results['rule_2_deleted'],
                results['rule_3_deleted'],
                results['rule_4_deleted']
            ])
            return results

        except Exception as e:
            self.logger.error("delete_likely_dud_events: Failed: %s", e)
            raise

    def delete_events_with_nulls(self) -> int:
        """
        Deletes events with critical null/missing values.

        Removes events where either:
        - Both start_date AND start_time are NULL, OR
        - Both start_time AND end_time are NULL

        These indicate incomplete event records that lack essential timing information.

        Returns:
            int: The number of events deleted from the table.

        Raises:
            Exception: If an error occurs during the deletion process.
        """
        try:
            delete_query = """
            DELETE FROM events
            WHERE (start_date IS NULL AND start_time IS NULL) OR
            (start_time IS NULL AND end_time IS NULL);
            """
            self.db.execute_query(delete_query)
            deleted_count = self.db.execute_query(delete_query) or 0
            self.logger.info(
                "delete_events_with_nulls: Deleted %d events with critical null values.",
                deleted_count
            )
            return deleted_count
        except Exception as e:
            self.logger.error("delete_events_with_nulls: Failed: %s", e)
            raise

    def dedup(self) -> int:
        """
        Deduplicates the events table by removing similar events.

        Duplicates are identified based on matching:
        - address_id
        - start_date and end_date
        - start_time and end_time (within 900 seconds / 15 minutes tolerance)

        For each group of duplicates, only the latest entry (highest event_id) is retained.
        All older duplicates are deleted.

        Returns:
            int: The number of rows deleted from the 'events' table during deduplication.

        Raises:
            Exception: If an error occurs during the deduplication process.
        """
        try:
            dedup_events_query = """
                DELETE FROM events e1
                USING events e2
                WHERE e1.event_id < e2.event_id
                    AND e1.address_id = e2.address_id
                    AND e1.start_date = e2.start_date
                    AND e1.end_date = e2.end_date
                    AND ABS(EXTRACT(EPOCH FROM (e1.start_time - e2.start_time))) <= 900
                    AND ABS(EXTRACT(EPOCH FROM (e1.end_time - e2.end_time))) <= 900;
            """
            deleted_count = self.db.execute_query(dedup_events_query) or 0
            self.logger.info(
                "dedup: Deduplicated events table successfully. Rows deleted: %d",
                deleted_count
            )
            return deleted_count
        except Exception as e:
            self.logger.error("dedup: Failed to deduplicate events: %s", e)
            raise

    def update_dow_date(self, event_id: int, corrected_date) -> bool:
        """
        Updates the start_date and end_date for a specific event.

        Used to correct date inconsistencies when the stored date doesn't match
        the expected day-of-week.

        Args:
            event_id (int): The unique identifier of the event to update.
            corrected_date: The new date to set for both start_date and end_date.

        Returns:
            bool: True if the update operation was executed.

        Raises:
            Exception: If an error occurs during the update.
        """
        try:
            update_query = """
                UPDATE events
                   SET start_date = :corrected_date,
                       end_date   = :corrected_date
                 WHERE event_id  = :event_id
            """
            params = {
                'corrected_date': corrected_date,
                'event_id': event_id
            }
            self.db.execute_query(update_query, params)
            self.logger.info(
                "update_dow_date: Updated event_id %d to date %s",
                event_id, corrected_date
            )
            return True
        except Exception as e:
            self.logger.error(
                "update_dow_date: Failed to update event_id %d: %s", event_id, e
            )
            raise

    def check_dow_date_consistent(self) -> dict:
        """
        Validates and corrects day-of-week date consistency for all events.

        Ensures that each event's start_date matches its specified day_of_week.

        Process:
        1. Retrieve all events' event_id, start_date, and day_of_week
        2. For each event, check if start_date's weekday matches day_of_week
        3. If mismatch found, compute minimal shift (within ±3 days)
        4. Call update_dow_date() to update both start_date and end_date
        5. Log all adjustments

        Returns:
            dict: Summary with keys:
                - events_checked: Total events validated
                - events_updated: Count of dates corrected
                - errors: Count of validation errors

        Raises:
            Exception: If an error occurs during the validation process.
        """
        results = {
            'events_checked': 0,
            'events_updated': 0,
            'errors': 0
        }

        try:
            select_query = """
                SELECT event_id, start_date, day_of_week
                  FROM events
            """
            rows = self.db.execute_query(select_query)

            # Map day names to weekday numbers (0=Monday, 6=Sunday)
            name_to_wd = {
                'monday': 0,
                'tuesday': 1,
                'wednesday': 2,
                'thursday': 3,
                'friday': 4,
                'saturday': 5,
                'sunday': 6
            }

            results['events_checked'] = len(rows) if rows else 0

            if not rows:
                self.logger.info("check_dow_date_consistent: No events to check.")
                return results

            for event_id, start_date, day_of_week in rows:
                try:
                    # Parse start_date if it's a string
                    if isinstance(start_date, str):
                        start_date = pd.to_datetime(start_date).date()

                    # Skip if day_of_week is empty or None
                    if not day_of_week or pd.isna(day_of_week):
                        continue

                    # Get expected weekday
                    day_lower = str(day_of_week).lower().strip()
                    if day_lower not in name_to_wd:
                        self.logger.warning(
                            "check_dow_date_consistent: Unrecognized day_of_week '%s' "
                            "for event_id %d",
                            day_of_week, event_id
                        )
                        continue

                    expected_wd = name_to_wd[day_lower]
                    actual_wd = start_date.weekday()

                    # Check if dates match
                    if actual_wd != expected_wd:
                        # Find nearest matching date within ±3 days
                        best_date = None
                        min_diff = 4  # Start with a value > 3

                        for shift in range(-3, 4):
                            candidate_date = start_date + timedelta(days=shift)
                            candidate_wd = candidate_date.weekday()

                            if candidate_wd == expected_wd:
                                diff = abs(shift)
                                if diff < min_diff:
                                    best_date = candidate_date
                                    min_diff = diff

                        if best_date:
                            self.update_dow_date(event_id, best_date)
                            results['events_updated'] += 1
                            self.logger.info(
                                "check_dow_date_consistent: Updated event_id %d from "
                                "%s (%s) to %s (%s)",
                                event_id, start_date,
                                list(name_to_wd.keys())[actual_wd],
                                best_date,
                                day_of_week
                            )
                        else:
                            self.logger.warning(
                                "check_dow_date_consistent: Could not find matching "
                                "date within ±3 days for event_id %d",
                                event_id
                            )
                except Exception as e:
                    self.logger.error(
                        "check_dow_date_consistent: Error processing event_id %d: %s",
                        event_id, e
                    )
                    results['errors'] += 1

            self.logger.info(
                "check_dow_date_consistent: Checked %d events, updated %d, %d errors",
                results['events_checked'], results['events_updated'], results['errors']
            )
            return results

        except Exception as e:
            self.logger.error("check_dow_date_consistent: Failed: %s", e)
            raise
