"""
Address Resolution Repository for LLM-based address processing and resolution.

This module consolidates address resolution logic, focusing on:
- Multi-level fallback strategy for address resolution
- LLM-based address parsing and extraction
- Cache management for resolved addresses
- Minimal address creation for missing location data

Previously scattered in DatabaseHandler, it handles the complex orchestration
of finding, parsing, and standardizing event addresses through multiple
resolution strategies (cache, quick lookup, LLM processing, fallback creation).

This repository focuses on resolving event addresses (Priority 6), complementing
AddressRepository (address lookup) and LocationCacheRepository (caching).
"""

from typing import Optional, Dict, Any
import logging
import pandas as pd
from fuzzywuzzy import fuzz


class AddressResolutionRepository:
    """
    Repository for orchestrating complex address resolution strategies.

    Consolidates operations for:
    - Coordinating address resolution from multiple sources (cache, quick lookup, LLM)
    - Managing multi-level fallback strategy
    - Extracting addresses from event details (building names)
    - Creating minimal fallback addresses

    Key responsibilities:
    - Orchestrate cache-first lookup strategy
    - Coordinate LLM-based address parsing
    - Handle fallback address creation
    - Normalize and store resolved addresses
    - Manage event location standardization
    """

    def __init__(self, db_handler, llm_handler=None):
        """
        Initialize AddressResolutionRepository with database and LLM handlers.

        Args:
            db_handler: DatabaseHandler instance for database operations
            llm_handler: Optional LLMHandler instance for AI-based resolution
        """
        self.db = db_handler
        self.llm = llm_handler
        self.logger = logging.getLogger(__name__)

    def process_event_address(self, event: dict) -> dict:
        """
        Main orchestration method for address resolution with multi-level fallback.

        Implements a 4-level fallback strategy:
        1. Cache lookup (fastest - exact match in raw_locations)
        2. Quick address lookup (regex/fuzzy matching without LLM)
        3. LLM processing (AI-based address parsing)
        4. Minimal fallback address (when all else fails)

        Args:
            event (dict): Event dictionary with keys:
                - location: str (raw location string)
                - event_name: str (event name)
                - source: str (source identifier)
                - url: str (source URL)
                - description: str (optional event description)

        Returns:
            dict: Event with resolved 'address_id' and standardized 'location'
        """
        try:
            location = event.get("location", None)
            event_name = event.get("event_name", "Unknown Event")
            source = event.get("source", "Unknown Source")

            # Normalize location string
            if location is not None and isinstance(location, str):
                location = location.strip()

            # Handle missing/invalid location (None, NaN, empty, too short, 'Unknown')
            if (location is None or pd.isna(location) or not isinstance(location, str) or
                len(location) < 5 or 'Unknown' in str(location)):
                self.logger.info(
                    "process_event_address: Location missing/invalid for event '%s' from %s, "
                    "attempting building name extraction",
                    event_name, source
                )
                return self._resolve_missing_location(event)

            # STEP 1: Check cache first (fastest path)
            cached_addr_id = self.db.lookup_raw_location(location)
            if cached_addr_id:
                return self._finalize_event_with_address(event, cached_addr_id, location, "cache")

            # STEP 2: Try quick lookup (regex/fuzzy matching, no LLM)
            quick_addr_id = self.db.quick_address_lookup(location)
            if quick_addr_id:
                # Cache for future use
                self.db.cache_raw_location(location, quick_addr_id)
                return self._finalize_event_with_address(event, quick_addr_id, location, "quick_lookup")

            # STEP 3: LLM processing (if available)
            if self.llm:
                address_id = self._resolve_via_llm(event, location)
                if address_id:
                    self.db.cache_raw_location(location, address_id)
                    return self._finalize_event_with_address(event, address_id, location, "llm")

            # STEP 4: Create minimal fallback address
            return self._create_fallback_address(event, location)

        except Exception as e:
            self.logger.error("process_event_address: Unexpected error during address resolution: %s", e)
            # Return event with minimal defaults instead of crashing
            event["address_id"] = 0
            event["location"] = event.get("source", "Location unavailable")
            return event

    def _resolve_missing_location(self, event: dict) -> dict:
        """
        Handle events with missing or invalid location data.

        Attempts multiple strategies:
        1. Extract building name from event details
        2. Deduplication check against source/event_name
        3. Create minimal address entry

        Args:
            event (dict): Event with missing/invalid location

        Returns:
            dict: Event with resolved address_id and location
        """
        # Try extracting building name from event details
        extracted_address_id = self._extract_address_from_event_details(event)
        if extracted_address_id:
            event["address_id"] = extracted_address_id
            full_address = self.db.get_full_address_from_id(extracted_address_id)
            if full_address:
                event["location"] = full_address
            self.logger.info(
                "process_event_address: Found existing address via building name extraction: "
                "address_id=%d", extracted_address_id
            )
            return event

        # Try deduplication check against source/event_name
        source = event.get("source", "Unknown")
        dedup_addr_id = self.db.find_address_by_building_name(source, threshold=75)
        if dedup_addr_id:
            event["address_id"] = dedup_addr_id
            full_address = self.db.get_full_address_from_id(dedup_addr_id)
            if full_address:
                event["location"] = full_address
            self.logger.info(
                "process_event_address: Found existing address via deduplication check: "
                "source='%s' → address_id=%d", source, dedup_addr_id
            )
            return event

        # Create minimal address entry
        return self._create_minimal_address(event)

    def _resolve_via_llm(self, event: dict, location: str) -> Optional[int]:
        """
        Use LLM to parse and resolve address from location string.

        This is the most expensive resolution strategy and should only be used
        after cache and quick lookup fail.

        Args:
            event (dict): Event dictionary with context
            location (str): Location string to parse

        Returns:
            Optional[int]: address_id if successful, None otherwise
        """
        try:
            # Generate LLM prompt
            prompt, schema_type = self.llm.generate_prompt(
                event.get("url", "address_fix"),
                location,
                "address_internet_fix"
            )

            # Query LLM
            llm_response = self.llm.query_llm(
                event.get("url", "").strip(),
                prompt,
                schema_type
            )

            # Parse LLM response
            parsed_results = self.llm.extract_and_parse_json(
                llm_response,
                "address_fix",
                schema_type
            )

            # Validate parsed results
            if not parsed_results or not isinstance(parsed_results, list) or \
               not isinstance(parsed_results[0], dict):
                self.logger.warning(
                    "process_event_address: Could not parse address from LLM response for '%s'",
                    location
                )
                return None

            # Normalize nulls and resolve address
            parsed_address = self.db.normalize_nulls(parsed_results[0])
            address_id = self.db.resolve_or_insert_address(parsed_address)

            if address_id:
                self.logger.info(
                    "process_event_address: LLM resolved address for '%s' → address_id=%d",
                    location, address_id
                )
                return address_id

            return None

        except Exception as e:
            self.logger.warning("process_event_address: LLM resolution failed: %s", e)
            return None

    def _extract_address_from_event_details(self, event: Dict[str, Any]) -> Optional[int]:
        """
        Extract building names from event_name and description, then match against database.

        Attempts to identify known venue/building names in the event details
        using both exact and fuzzy matching.

        Args:
            event (dict): Event dictionary with event_name and description

        Returns:
            Optional[int]: address_id if match found, None otherwise
        """
        try:
            building_dict = self.db._get_building_name_dictionary()
            if not building_dict:
                return None

            # Collect text to search
            search_texts = []
            event_name = event.get("event_name", "")
            description = event.get("description", "")

            if event_name:
                search_texts.append(str(event_name))
            if description:
                search_texts.append(str(description))

            if not search_texts:
                return None

            combined_text = " ".join(search_texts).lower()

            # First try exact matches
            for building_name, address_id in building_dict.items():
                if building_name in combined_text:
                    self.logger.info(
                        "_extract_address_from_event_details: Found exact match '%s' → address_id=%d",
                        building_name, address_id
                    )
                    return address_id

            # Then try fuzzy matching for partial matches
            best_match = None
            best_score = 0

            for building_name, address_id in building_dict.items():
                # Skip very short building names to avoid false positives
                if len(building_name) < 6:
                    continue

                score = fuzz.partial_ratio(building_name, combined_text)
                if score > 80 and score > best_score:
                    best_score = score
                    best_match = (building_name, address_id)

            if best_match:
                building_name, address_id = best_match
                self.logger.info(
                    "_extract_address_from_event_details: Found fuzzy match '%s' "
                    "(score: %d) → address_id=%d",
                    building_name, best_score, address_id
                )
                return address_id

            return None

        except Exception as e:
            self.logger.warning("_extract_address_from_event_details: Error during extraction: %s", e)
            return None

    def _create_minimal_address(self, event: dict) -> dict:
        """
        Create a minimal but valid address when location data is completely missing.

        Creates a basic address entry using available event information,
        falls back to reasonable defaults if needed.

        Args:
            event (dict): Event with missing location

        Returns:
            dict: Event with created address_id and location
        """
        event_name = event.get("event_name", "Unknown Event")
        source = event.get("source", "Unknown Source")

        minimal_address = {
            "address_id": 0,
            "full_address": f"Location details unavailable - {source}",
            "building_name": str(event_name)[:50],
            "street_number": "",
            "street_name": "",
            "street_type": "",
            "direction": None,
            "city": "Unknown",
            "met_area": None,
            "province_or_state": "BC",
            "postal_code": None,
            "country_id": "CA",
            "time_stamp": None
        }

        address_id = self.db.resolve_or_insert_address(minimal_address)
        if address_id:
            event["address_id"] = address_id
            full_address = self.db.get_full_address_from_id(address_id)
            if full_address:
                event["location"] = full_address
            else:
                event["location"] = minimal_address["full_address"]
            self.logger.info(
                "process_event_address: Created minimal address entry with address_id=%d",
                address_id
            )
            return event
        else:
            # Final fallback
            self.logger.error("process_event_address: Failed to create minimal address entry")
            event["address_id"] = 0
            event["location"] = f"Location unavailable - {source}"
            return event

    def _create_fallback_address(self, event: dict, location: str) -> dict:
        """
        Create fallback address when all resolution strategies fail.

        This is the last resort - attempts to create a minimal address
        and gracefully handles failure by setting reasonable defaults.

        Args:
            event (dict): Event dictionary
            location (str): Location string that couldn't be resolved

        Returns:
            dict: Event with fallback address or defaults
        """
        event_name = event.get("event_name", "Event")
        source = event.get("source", "Unknown")

        minimal_address = {
            "building_name": str(event_name)[:50],
            "street_name": location[:50] if location else "Unknown Location",
            "city": "Unknown",
            "province_or_state": "BC",
            "country_id": "Canada"
        }

        address_id = self.db.resolve_or_insert_address(minimal_address)
        if address_id:
            event["address_id"] = address_id
            full_address = self.db.get_full_address_from_id(address_id)
            if full_address:
                event["location"] = full_address
            self.logger.info(
                "process_event_address: Created fallback address for '%s' → address_id=%d",
                location, address_id
            )
            return event

        # Ultimate fallback
        self.logger.error(
            "process_event_address: All address resolution attempts failed for '%s'",
            location
        )
        event["address_id"] = 0
        event["location"] = f"Location unavailable - {source}"
        return event

    def _finalize_event_with_address(
        self,
        event: dict,
        address_id: int,
        location: str,
        source: str
    ) -> dict:
        """
        Finalize event with resolved address information.

        Updates event with address_id and retrieves canonical full_address
        from address table for consistency.

        Args:
            event (dict): Event to update
            address_id (int): Resolved address_id
            location (str): Original location string
            source (str): Resolution source (cache/quick_lookup/llm)

        Returns:
            dict: Event with updated address_id and location
        """
        event["address_id"] = address_id
        full_address = self.db.get_full_address_from_id(address_id)
        if full_address:
            event["location"] = full_address

        self.logger.info(
            "process_event_address: Resolved via %s for '%s' → address_id=%d",
            source, location, address_id
        )
        return event
