"""
Location Cache Repository for managing location-to-address mappings and lookups.

This module consolidates location caching and building name lookup operations:
- In-memory building name dictionary caching
- Location string to address_id caching
- Raw location lookup for fast address resolution
- Building name dictionary creation and management

Previously scattered in DatabaseHandler, these operations optimize address resolution
by caching frequently accessed mappings and reducing database queries.

This repository focuses on caching and lookup optimization (Priority 8),
complementing AddressRepository (resolution) and AddressResolutionRepository (LLM).
"""

from typing import Optional, Dict
from datetime import datetime
import logging


class LocationCacheRepository:
    """
    Repository for managing location caching and lookup operations.

    Consolidates operations for:
    - In-memory building name dictionary caching
    - Raw location to address_id caching
    - Fast location lookups without database overhead
    - Building name extraction and matching

    Key responsibilities:
    - Cache building names from address table
    - Store raw location to address_id mappings
    - Provide fast lookups for cached locations
    - Manage cache lifecycle and freshness
    """

    def __init__(self, db_handler):
        """
        Initialize LocationCacheRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)
        self._building_name_cache = {}  # In-memory cache

    def _get_building_name_dictionary(self) -> Dict[str, int]:
        """
        Creates and caches a dictionary mapping building names to address_ids.

        Builds an in-memory dictionary on first call, then returns cached version
        on subsequent calls. Building names are normalized to lowercase for
        case-insensitive matching.

        Returns:
            dict: Dictionary with lowercase building_name as keys and address_id as values.
                 Empty dict if no building names found.
        """
        try:
            if not self._building_name_cache:
                self.logger.info("_get_building_name_dictionary: Building building name lookup cache")

                query = """
                    SELECT address_id, building_name FROM address
                    WHERE building_name IS NOT NULL AND building_name != ''
                """
                results = self.db.execute_query(query)

                if results:
                    for address_id, building_name in results:
                        if building_name and building_name.strip():
                            # Use lowercase for case-insensitive matching
                            key = building_name.lower().strip()
                            self._building_name_cache[key] = address_id

                self.logger.info(
                    "_get_building_name_dictionary: Cached %d building names",
                    len(self._building_name_cache)
                )

            return self._building_name_cache

        except Exception as e:
            self.logger.error("_get_building_name_dictionary: Failed to build cache: %s", e)
            return {}

    def cache_raw_location(self, raw_location: str, address_id: int) -> bool:
        """
        Cache a raw location string to address_id mapping for fast future lookups.

        Uses PostgreSQL ON CONFLICT to avoid duplicate key errors when the same
        location is cached multiple times. Silently ignores conflicts.

        Args:
            raw_location (str): The raw location string to cache
            address_id (int): The address_id this location maps to

        Returns:
            bool: True if cached successfully, False if error (but doesn't raise)
        """
        try:
            # PostgreSQL syntax: INSERT ... ON CONFLICT DO NOTHING
            insert_query = """
                INSERT INTO raw_locations (raw_location, address_id, created_at)
                VALUES (:raw_location, :address_id, :created_at)
                ON CONFLICT (raw_location) DO NOTHING
            """
            result = self.db.execute_query(insert_query, {
                "raw_location": raw_location,
                "address_id": address_id,
                "created_at": datetime.now()
            })

            self.logger.info(
                "cache_raw_location: Cached '%s' → address_id=%d",
                raw_location[:50], address_id
            )
            return True

        except Exception as e:
            self.logger.warning(
                "cache_raw_location: Failed to cache '%s': %s",
                raw_location[:50], e
            )
            return False

    def lookup_raw_location(self, raw_location: str) -> Optional[int]:
        """
        Look up a raw location string in the cache to get its address_id.

        Provides fast address resolution for previously seen location strings
        without requiring expensive address parsing or LLM processing.

        Args:
            raw_location (str): The raw location string to look up

        Returns:
            Optional[int]: address_id if found in cache, None otherwise
        """
        try:
            result = self.db.execute_query(
                "SELECT address_id FROM raw_locations WHERE raw_location = :raw_location",
                {"raw_location": raw_location}
            )

            if result:
                address_id = result[0][0]
                self.logger.info(
                    "lookup_raw_location: Cache hit for '%s' → address_id=%d",
                    raw_location[:50], address_id
                )
                return address_id

            self.logger.debug("lookup_raw_location: Cache miss for '%s'", raw_location[:50])
            return None

        except Exception as e:
            self.logger.warning(
                "lookup_raw_location: Cache lookup failed for '%s': %s",
                raw_location[:50], e
            )
            return None

    def create_raw_locations_table(self) -> bool:
        """
        Create the raw_locations table for caching location string to address_id mappings.

        Also creates the address table if it doesn't exist to satisfy the foreign key
        constraint. Creates an index on raw_location for fast lookups.

        Returns:
            bool: True if tables created successfully, False if error
        """
        try:
            # First ensure address table exists for foreign key constraint
            address_table_query = """
                CREATE TABLE IF NOT EXISTS address (
                    address_id SERIAL PRIMARY KEY,
                    full_address TEXT UNIQUE,
                    building_name TEXT,
                    street_number TEXT,
                    street_name TEXT,
                    street_type TEXT,
                    direction TEXT,
                    city TEXT,
                    met_area TEXT,
                    province_or_state TEXT,
                    postal_code TEXT,
                    country_id TEXT,
                    time_stamp TIMESTAMP
                )
            """

            # PostgreSQL syntax (not SQLite)
            create_table_query = """
                CREATE TABLE IF NOT EXISTS raw_locations (
                    raw_location_id SERIAL PRIMARY KEY,
                    raw_location TEXT NOT NULL UNIQUE,
                    address_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (address_id) REFERENCES address(address_id)
                )
            """

            # Create address table first
            self.db.execute_query(address_table_query)
            self.logger.info("create_raw_locations_table: Address table creation/verification completed")

            # Then create raw_locations table with foreign key
            self.db.execute_query(create_table_query)
            self.logger.info("create_raw_locations_table: Raw locations table creation/verification completed")

            # Create index for faster lookups
            index_query = "CREATE INDEX IF NOT EXISTS idx_raw_location ON raw_locations(raw_location)"
            self.db.execute_query(index_query)
            self.logger.info("create_raw_locations_table: Index creation/verification completed")

            return True

        except Exception as e:
            self.logger.error("create_raw_locations_table: Failed to create table: %s", e)
            return False

    def clear_building_cache(self) -> None:
        """
        Clear the in-memory building name cache.

        Useful when address data has been updated and cache needs to be refreshed.
        The cache will be rebuilt on next call to _get_building_name_dictionary().
        """
        try:
            cache_size = len(self._building_name_cache)
            self._building_name_cache.clear()
            self.logger.info("clear_building_cache: Cleared %d cached building names", cache_size)
        except Exception as e:
            self.logger.error("clear_building_cache: Error clearing cache: %s", e)
