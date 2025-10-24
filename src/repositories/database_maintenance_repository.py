"""
Database Maintenance Repository for administrative and maintenance operations.

This module consolidates database maintenance and administrative operations:
- SQL input execution from configuration files
- Address ID sequence reset and renumbering
- Building name updates with full address synchronization

Previously scattered in DatabaseHandler, these operations handle database
maintenance, cleanup, and administrative tasks that require careful coordination.

This repository focuses on database maintenance operations (Priority 9),
complementing the main data repositories with administrative capabilities.

⚠️  WARNING: Some operations in this repository are HIGH RISK and should only
be executed during maintenance windows with proper backups and no concurrent access.
"""

import logging
import json
from typing import Optional


class DatabaseMaintenanceRepository:
    """
    Repository for database maintenance and administrative operations.

    Consolidates operations for:
    - SQL batch execution from configuration files
    - Address ID sequence reset with full reference updates
    - Building name synchronization with address data

    Key responsibilities:
    - Execute administrative SQL operations
    - Maintain database integrity during maintenance
    - Coordinate complex multi-step maintenance procedures
    - Provide safe, auditable maintenance operations

    ⚠️  HIGH RISK OPERATIONS:
    - reset_address_id_sequence() requires database lock and no concurrent access
    - Should only be called during maintenance windows
    - Always backup database before execution
    """

    def __init__(self, db_handler):
        """
        Initialize DatabaseMaintenanceRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)

    def sql_input(self, file_path: str) -> bool:
        """
        Execute SQL queries from a JSON configuration file.

        Reads a JSON file containing a flat dictionary of SQL statements
        and executes them sequentially. Each key is a descriptive name,
        each value is a SQL statement to execute.

        Args:
            file_path (str): Path to JSON file containing SQL statements.
                            Format: {"operation_name": "SQL statement", ...}

        Returns:
            bool: True if all queries executed (success or expected failure),
                 False if file couldn't be loaded

        Example JSON file:
        {
            "fix_duplicate_addresses": "DELETE FROM address WHERE ...",
            "update_missing_cities": "UPDATE address SET city = ... WHERE ..."
        }
        """
        try:
            self.logger.info("sql_input(): Starting SQL input execution from %s", file_path)

            # Load SQL statements from JSON file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sql_dict = json.load(f)
                self.logger.info("sql_input(): Successfully loaded %d SQL entries", len(sql_dict))
            except Exception as e:
                self.logger.error("sql_input(): Failed to load or parse JSON file: %s", e)
                return False

            # Execute each SQL statement
            executed = 0
            failed = 0

            for name, query in sql_dict.items():
                try:
                    self.logger.info("sql_input(): Executing [%s]: %s", name, query[:80])
                    result = self.db.execute_query(query)

                    if result is None:
                        self.logger.warning("sql_input(): Failed to execute [%s]", name)
                        failed += 1
                    else:
                        self.logger.info("sql_input(): Successfully executed [%s]", name)
                        executed += 1

                except Exception as e:
                    self.logger.error("sql_input(): Exception executing [%s]: %s", name, e)
                    failed += 1

            self.logger.info(
                "sql_input(): Completed - Executed: %d, Failed: %d out of %d total",
                executed, failed, len(sql_dict)
            )
            return True

        except Exception as e:
            self.logger.error("sql_input(): Unexpected error: %s", e)
            return False

    def reset_address_id_sequence(self) -> Optional[int]:
        """
        Reset address_id sequence to start from 1, updating all references.

        ⚠️  HIGH RISK OPERATION ⚠️
        - Requires database lock (no concurrent access)
        - Updates address_id in ALL related tables (address, events, events_history, raw_locations)
        - Should only be called during maintenance window with backup
        - Irreversible operation - ensure data is backed up first

        This operation:
        1. Cleans up orphaned raw_locations records
        2. Creates mapping from old to new sequential IDs
        3. Updates all tables with new address_id values
        4. Resets PostgreSQL sequence for new inserts

        Returns:
            Optional[int]: Number of addresses renumbered, None if error

        Example usage (CAREFULLY):
        ```python
        # Only during maintenance window!
        backup_database()  # FIRST!
        try:
            count = repo.reset_address_id_sequence()
            if count:
                db.commit()
        except Exception as e:
            db.rollback()
            raise
        ```
        """
        try:
            self.logger.warning("reset_address_id_sequence(): ⚠️  STARTING HIGH RISK OPERATION ⚠️")
            self.logger.warning("reset_address_id_sequence(): Ensure no concurrent database access!")

            import pandas as pd

            # Step 0: Clean up orphaned references first
            cleanup_orphaned_sql = """
                DELETE FROM raw_locations
                WHERE address_id NOT IN (SELECT address_id FROM address);
            """
            orphaned_count = self.db.execute_query(cleanup_orphaned_sql) or 0
            self.logger.info("reset_address_id_sequence(): Cleaned up %d orphaned raw_locations", orphaned_count)

            # Step 1: Get all addresses and create mapping
            get_addresses_sql = """
                SELECT address_id, full_address, building_name, street_number, street_name,
                       street_type, direction, city, met_area, province_or_state,
                       postal_code, country_id, time_stamp
                FROM address
                ORDER BY address_id;
            """

            addresses_df = pd.read_sql(get_addresses_sql, self.db.conn)

            if addresses_df.empty:
                self.logger.info("reset_address_id_sequence(): No addresses found to renumber")
                return 0

            # Create mapping from old to new sequential IDs
            address_mapping = {}
            for idx, row in addresses_df.iterrows():
                old_id = row['address_id']
                new_id = idx + 1
                address_mapping[old_id] = new_id

            self.logger.info("reset_address_id_sequence(): Created mapping for %d addresses", len(address_mapping))

            # Step 2: Create temporary table with new IDs
            create_temp_sql = """
                CREATE TEMP TABLE address_temp AS
                SELECT * FROM address WHERE 1=0;
            """
            self.db.execute_query(create_temp_sql)
            self.logger.info("reset_address_id_sequence(): Created temporary table")

            # Step 3: Insert renumbered addresses into temp table
            for old_id, new_id in address_mapping.items():
                old_row = addresses_df[addresses_df['address_id'] == old_id].iloc[0]
                insert_temp_sql = """
                    INSERT INTO address_temp (address_id, full_address, building_name,
                        street_number, street_name, street_type, direction, city,
                        met_area, province_or_state, postal_code, country_id, time_stamp)
                    VALUES (:new_id, :full_address, :building_name, :street_number,
                        :street_name, :street_type, :direction, :city, :met_area,
                        :province_or_state, :postal_code, :country_id, :time_stamp)
                """
                params = {
                    'new_id': new_id,
                    'full_address': old_row['full_address'],
                    'building_name': old_row['building_name'],
                    'street_number': old_row['street_number'],
                    'street_name': old_row['street_name'],
                    'street_type': old_row['street_type'],
                    'direction': old_row['direction'],
                    'city': old_row['city'],
                    'met_area': old_row['met_area'],
                    'province_or_state': old_row['province_or_state'],
                    'postal_code': old_row['postal_code'],
                    'country_id': old_row['country_id'],
                    'time_stamp': old_row['time_stamp']
                }
                self.db.execute_query(insert_temp_sql, params)

            self.logger.info("reset_address_id_sequence(): Inserted renumbered addresses into temp table")

            # Step 4: Update all dependent tables with new address_ids
            events_updated = 0
            events_history_updated = 0
            raw_locations_updated = 0

            for old_id, new_id in address_mapping.items():
                # Update events table
                update_events_sql = """
                    UPDATE events SET address_id = :new_id WHERE address_id = :old_id;
                """
                result = self.db.execute_query(update_events_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_updated += result

                # Update events_history table
                update_events_history_sql = """
                    UPDATE events_history SET address_id = :new_id WHERE address_id = :old_id;
                """
                result = self.db.execute_query(update_events_history_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    events_history_updated += result

                # Update raw_locations table
                update_raw_locations_sql = """
                    UPDATE raw_locations SET address_id = :new_id WHERE address_id = :old_id;
                """
                result = self.db.execute_query(update_raw_locations_sql, {'new_id': new_id, 'old_id': old_id})
                if result:
                    raw_locations_updated += result

            self.logger.info("reset_address_id_sequence(): Updated events table for %d addresses", events_updated)
            self.logger.info("reset_address_id_sequence(): Updated events_history table for %d addresses", events_history_updated)
            self.logger.info("reset_address_id_sequence(): Updated raw_locations table for %d addresses", raw_locations_updated)

            # Step 5: Replace original address table with renumbered version
            self.db.execute_query("DELETE FROM address;")
            self.logger.info("reset_address_id_sequence(): Cleared original address table")

            copy_back_sql = """
                INSERT INTO address (address_id, full_address, building_name, street_number,
                    street_name, street_type, direction, city, met_area,
                    province_or_state, postal_code, country_id, time_stamp)
                SELECT address_id, full_address, building_name, street_number,
                    street_name, street_type, direction, city, met_area,
                    province_or_state, postal_code, country_id, time_stamp
                FROM address_temp;
            """
            self.db.execute_query(copy_back_sql)
            self.logger.info("reset_address_id_sequence(): Copied renumbered data back to address table")

            # Step 6: Reset the sequence
            max_id = len(address_mapping)
            reset_sequence_sql = f"""
                ALTER SEQUENCE address_address_id_seq RESTART WITH {max_id + 1};
            """
            self.db.execute_query(reset_sequence_sql)
            self.logger.info("reset_address_id_sequence(): Reset sequence to start at %d", max_id + 1)

            self.logger.warning("reset_address_id_sequence(): ✅ SUCCESSFULLY COMPLETED")
            return len(address_mapping)

        except Exception as e:
            self.logger.error("reset_address_id_sequence(): ❌ FAILED - %s", e)
            self.logger.error("reset_address_id_sequence(): PLEASE CHECK DATABASE INTEGRITY")
            raise

    def update_full_address_with_building_names(self) -> Optional[int]:
        """
        Synchronize full_address with building_name where address_id lacks full_address.

        Updates address records to use building_name as full_address when:
        - full_address is NULL or empty
        - building_name is available and non-empty

        Returns:
            Optional[int]: Number of records updated, None if error
        """
        try:
            self.logger.info("update_full_address_with_building_names(): Starting synchronization")

            update_sql = """
                UPDATE address
                SET full_address = building_name
                WHERE (full_address IS NULL OR full_address = '')
                AND building_name IS NOT NULL
                AND building_name != '';
            """

            result = self.db.execute_query(update_sql)
            updated_count = result if result else 0

            self.logger.info(
                "update_full_address_with_building_names(): Updated %d records",
                updated_count
            )
            return updated_count

        except Exception as e:
            self.logger.error("update_full_address_with_building_names(): Failed - %s", e)
            raise
