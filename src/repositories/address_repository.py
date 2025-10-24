"""
Address repository for centralized address management and resolution.

This module consolidates all address-related database operations previously
scattered throughout DatabaseHandler. It uses fuzzy matching for deduplication
and provides a clean interface for address CRUD operations.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fuzzy_utils import FuzzyMatcher
from rapidfuzz import fuzz


class AddressRepository:
    """
    Repository for managing address operations in the database.

    Consolidates address resolution, deduplication, and formatting logic
    previously scattered across DatabaseHandler.

    Key responsibilities:
    - Address resolution with fuzzy matching
    - Address insertion and deduplication
    - Address formatting and validation
    - Building name lookups and matching
    """

    def __init__(self, db_handler):
        """
        Initialize AddressRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.fuzzy = FuzzyMatcher()
        self.logger = logging.getLogger(__name__)

    def resolve_or_insert_address(self, parsed_address: dict) -> Optional[int]:
        """
        Resolves an address by checking multiple matching strategies in order of specificity.
        Uses improved fuzzy matching to prevent duplicate addresses.

        Args:
            parsed_address (dict): Dictionary of parsed address fields.
                Expected keys: building_name, street_number, street_name, postal_code,
                              city, province_or_state, country_id, street_type

        Returns:
            int or None: The address_id of the matched or newly inserted address.
        """
        if not parsed_address:
            self.logger.info("resolve_or_insert_address: No parsed address provided.")
            return None

        building_name = (parsed_address.get("building_name") or "").strip()
        street_number = (parsed_address.get("street_number") or "").strip()
        street_name = (parsed_address.get("street_name") or "").strip()
        postal_code = (parsed_address.get("postal_code") or "").strip()
        city = (parsed_address.get("city") or "").strip()
        country_id = (parsed_address.get("country_id") or "").strip()

        # Step 1: Exact match on postal code + street number (most specific)
        if postal_code and street_number:
            self.logger.debug(
                f"resolve_or_insert_address: Trying postal_code + street_number "
                f"match: {postal_code}, {street_number}"
            )
            postal_match_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(postal_code) = LOWER(:postal_code)
                AND LOWER(street_number) = LOWER(:street_number)
            """
            postal_matches = self.db.execute_query(postal_match_query, {
                "postal_code": postal_code,
                "street_number": street_number
            })

            for addr_id, b_name, s_num, s_name, p_code in postal_matches or []:
                if building_name and b_name:
                    # Use FuzzyMatcher for consistent fuzzy matching
                    if self.fuzzy.compare(building_name, b_name, threshold=85, algorithm='token_set'):
                        self.logger.debug(
                            f"Postal+street+fuzzy building match → address_id={addr_id}"
                        )
                        return addr_id
                else:
                    # Same postal code + street number is very likely the same location
                    self.logger.debug(
                        f"Postal+street match (no building comparison) → address_id={addr_id}"
                    )
                    return addr_id

        # Step 2: Street number + street name match with improved building name fuzzy matching
        if street_number and street_name:
            select_query = """
                SELECT address_id, building_name, street_number, street_name, postal_code
                FROM address
                WHERE LOWER(street_number) = LOWER(:street_number)
                AND (LOWER(street_name) = LOWER(:street_name) OR LOWER(street_name) = LOWER(:street_name_alt))
            """
            # Handle common street name variations (Niagra vs Niagara)
            street_name_alt = street_name.replace('Niagra', 'Niagara').replace('Niagara', 'Niagra')

            street_matches = self.db.execute_query(select_query, {
                "street_number": street_number,
                "street_name": street_name,
                "street_name_alt": street_name_alt
            })

            for addr_id, b_name, s_num, s_name, p_code in street_matches or []:
                if building_name and b_name:
                    # Use FuzzyMatcher for consistent fuzzy matching
                    if self.fuzzy.compare(building_name, b_name, threshold=75, algorithm='token_set'):
                        self.logger.debug(
                            f"Street+fuzzy building match → address_id={addr_id}"
                        )
                        return addr_id
                else:
                    self.logger.debug(
                        f"Street match (no building name) → address_id={addr_id}"
                    )
                    return addr_id
        else:
            self.logger.debug(
                "resolve_or_insert_address: Missing street_number or street_name; "
                "skipping street match"
            )

        # Step 3: City + building name fuzzy match (broader search)
        if city and building_name:
            self.logger.debug(
                f"resolve_or_insert_address: Trying city + building_name match: "
                f"{city}, {building_name}"
            )
            city_building_query = """
                SELECT address_id, building_name, city, postal_code
                FROM address
                WHERE LOWER(city) = LOWER(:city) AND building_name IS NOT NULL
            """
            city_matches = self.db.execute_query(city_building_query, {"city": city})

            for addr_id, b_name, addr_city, p_code in city_matches or []:
                if b_name:
                    if self.fuzzy.compare(building_name, b_name, threshold=90, algorithm='token_set'):
                        self.logger.debug(
                            f"City+building fuzzy match → address_id={addr_id}"
                        )
                        return addr_id

        # Step 4: Legacy building name-only fuzzy match (least reliable)
        if building_name:
            self.logger.debug(
                f"resolve_or_insert_address: Trying building_name-only fuzzy match: "
                f"{building_name}"
            )
            query = "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            candidates = self.db.execute_query(query)

            for addr_id, existing_name in candidates or []:
                if existing_name:
                    # Use FuzzyMatcher for consistent scoring
                    score = self.fuzzy.get_score(building_name, existing_name, algorithm='token_set')
                    if score >= 95:
                        self.logger.debug(
                            f"Building-name-only fuzzy match → address_id={addr_id} "
                            f"(score={score})"
                        )
                        return addr_id

        # Step 5: Normalize null values and prepare required fields for insert
        parsed_address = self.db.normalize_nulls(parsed_address)

        # Ensure ALL fields expected by INSERT query are present (set to None if missing)
        required_fields = [
            "building_name", "street_number", "street_name", "city",
            "province_or_state", "postal_code", "country_id"
        ]

        for field in required_fields:
            if field not in parsed_address:
                parsed_address[field] = None

        # Set specific fields from extracted values
        parsed_address["building_name"] = building_name or parsed_address.get("building_name")
        parsed_address["street_number"] = street_number or parsed_address.get("street_number")
        parsed_address["street_name"] = street_name or parsed_address.get("street_name")
        parsed_address["country_id"] = country_id or parsed_address.get("country_id")

        # Build standardized full_address from components
        standardized_full_address = self.build_full_address(
            building_name=parsed_address.get("building_name"),
            street_number=parsed_address.get("street_number"),
            street_name=parsed_address.get("street_name"),
            street_type=parsed_address.get("street_type"),
            city=parsed_address.get("city"),
            province_or_state=parsed_address.get("province_or_state"),
            postal_code=parsed_address.get("postal_code"),
            country_id=parsed_address.get("country_id")
        )
        parsed_address["full_address"] = standardized_full_address

        # Set time_stamp for the new address
        parsed_address["time_stamp"] = datetime.now().isoformat()

        # FINAL DEDUPLICATION CHECK: Before inserting, check if building_name already exists
        building_name = parsed_address.get("building_name") or ""
        if isinstance(building_name, str):
            building_name = building_name.strip()
        else:
            building_name = ""
        if building_name and len(building_name) > 2:
            # Try to find existing address with same building name
            existing_addr_id = self.find_address_by_building_name(building_name, threshold=80)
            if existing_addr_id:
                self.logger.info(
                    f"resolve_or_insert_address: Found existing address (dedup) with "
                    f"building_name='{building_name}' → address_id={existing_addr_id}"
                )
                return existing_addr_id

        insert_query = """
            INSERT INTO address (
                building_name, street_number, street_name, city,
                province_or_state, postal_code, country_id, full_address, time_stamp
            ) VALUES (
                :building_name, :street_number, :street_name, :city,
                :province_or_state, :postal_code, :country_id, :full_address, :time_stamp
            )
            RETURNING address_id;
        """

        result = self.db.execute_query(insert_query, parsed_address)
        if result:
            address_id = result[0][0]
            self.logger.info(f"Inserted new address with address_id: {address_id}")
            return address_id
        else:
            # If insert failed (likely due to unique constraint), try to find existing address
            full_address = parsed_address.get("full_address")
            if full_address:
                lookup_query = "SELECT address_id FROM address WHERE full_address = :full_address"
                lookup_result = self.db.execute_query(lookup_query, {"full_address": full_address})
                if lookup_result:
                    address_id = lookup_result[0][0]
                    self.logger.info(f"Found existing address with address_id: {address_id}")
                    return address_id

            self.logger.error("resolve_or_insert_address: Failed to insert or find existing address")
            return None

    def build_full_address(
        self,
        building_name: str = None,
        street_number: str = None,
        street_name: str = None,
        street_type: str = None,
        city: str = None,
        province_or_state: str = None,
        postal_code: str = None,
        country_id: str = None
    ) -> str:
        """
        Builds a standardized full_address string from address components.

        Format: "building_name, street_number street_name street_type, city,
                 province_or_state postal_code, country_id"

        Args:
            building_name: Building or venue name (optional)
            street_number: Street number
            street_name: Street name
            street_type: Street type (St, Ave, Rd, etc.)
            city: City name
            province_or_state: Province or state
            postal_code: Postal code
            country_id: Country code

        Returns:
            str: Formatted full address
        """
        address_parts = []

        # Add building name first if present
        if building_name and building_name.strip():
            address_parts.append(building_name.strip())

        # Build street address
        street_parts = []
        if street_number and street_number.strip():
            street_parts.append(street_number.strip())
        if street_name and street_name.strip():
            street_parts.append(street_name.strip())
        if street_type and street_type.strip():
            street_parts.append(street_type.strip())

        if street_parts:
            address_parts.append(' '.join(street_parts))

        # Add city
        if city and city.strip():
            address_parts.append(city.strip())

        # Add province/state and postal code
        if province_or_state and province_or_state.strip():
            if postal_code and postal_code.strip():
                address_parts.append(f"{province_or_state.strip()} {postal_code.strip()}")
            else:
                address_parts.append(province_or_state.strip())
        elif postal_code and postal_code.strip():
            address_parts.append(postal_code.strip())

        # Add country
        if country_id and country_id.strip():
            address_parts.append(country_id.strip())

        return ', '.join(address_parts)

    def get_full_address_from_id(self, address_id: int) -> Optional[str]:
        """
        Returns the full_address from the address table for the given address_id.

        Args:
            address_id: The ID of the address to look up

        Returns:
            str: The full address string, or None if not found
        """
        query = "SELECT full_address FROM address WHERE address_id = :address_id"
        result = self.db.execute_query(query, {"address_id": address_id})
        return result[0][0] if result else None

    def find_address_by_building_name(
        self,
        building_name: str,
        threshold: int = 75
    ) -> Optional[int]:
        """
        Find an existing address by fuzzy matching on building_name.
        Prevents creation of duplicate addresses with the same venue name.

        Args:
            building_name (str): The venue/building name to search for
            threshold (int): Fuzzy match score threshold (0-100)

        Returns:
            address_id if found, None otherwise
        """
        if not building_name or not isinstance(building_name, str):
            return None

        building_name = building_name.strip()

        try:
            # Query all addresses with building names
            building_matches = self.db.execute_query(
                "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL"
            )

            best_score = 0
            best_addr_id = None

            for addr_id, existing_building in building_matches or []:
                if existing_building and existing_building.strip():
                    # Use FuzzyMatcher for consistent scoring
                    score = self.fuzzy.get_score(
                        building_name,
                        existing_building,
                        algorithm='partial'
                    )

                    if score >= threshold and score > best_score:
                        best_score = score
                        best_addr_id = addr_id
                        self.logger.debug(
                            f"find_address_by_building_name: '{building_name}' vs "
                            f"'{existing_building}' = {score}"
                        )

            if best_addr_id:
                self.logger.info(
                    f"find_address_by_building_name: Found address_id={best_addr_id} "
                    f"for '{building_name}' (score={best_score})"
                )
                return best_addr_id

            self.logger.debug(f"find_address_by_building_name: No match found for '{building_name}'")
            return None

        except Exception as e:
            self.logger.warning(f"find_address_by_building_name: Error looking up '{building_name}': {e}")
            return None

    def quick_address_lookup(self, location: str) -> Optional[int]:
        """
        Attempts to find an existing address without using LLM by:
        1. Exact string match on full_address
        2. Regex parsing to extract street_number + street_name for exact match
        3. Fuzzy matching on building names for the same street

        Args:
            location: The location string to look up

        Returns:
            address_id if found, None if LLM is needed
        """
        # Step 1: Exact string match (already implemented)
        exact_match = self.db.execute_query(
            "SELECT address_id FROM address WHERE LOWER(full_address) = LOWER(:location)",
            {"location": location}
        )
        if exact_match:
            self.logger.info(f"quick_address_lookup: Exact match → address_id={exact_match[0][0]}")
            return exact_match[0][0]

        # Step 2: Parse basic components with regex
        street_pattern = (
            r'(\d+)\s+([A-Za-z\s]+?)(?:,|\s+(?:Street|St|Avenue|Ave|Road|Rd|'
            r'Drive|Dr|Way|Lane|Ln|Boulevard|Blvd))'
        )
        street_match = re.search(street_pattern, location, re.IGNORECASE)

        if street_match:
            street_number = street_match.group(1).strip()
            street_name_raw = street_match.group(2).strip()

            # Clean street name (remove common suffixes if they got included)
            street_name = re.sub(
                r'\b(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Way|Lane|Ln|Boulevard|Blvd)\b',
                '',
                street_name_raw,
                flags=re.IGNORECASE
            ).strip()

            if street_number and street_name:
                # Try exact match on street number + street name
                exact_street_match = self.db.execute_query(
                    """SELECT address_id FROM address
                       WHERE LOWER(street_number) = LOWER(:street_number)
                       AND LOWER(street_name) = LOWER(:street_name)""",
                    {"street_number": street_number, "street_name": street_name}
                )

                if exact_street_match:
                    self.logger.info(
                        f"quick_address_lookup: Street match → address_id={exact_street_match[0][0]}"
                    )
                    return exact_street_match[0][0]

                # Step 3: Try fuzzy building name match for addresses on the same street
                building_query = """
                    SELECT address_id, building_name FROM address
                    WHERE LOWER(street_number) = LOWER(:street_number)
                    AND LOWER(street_name) = LOWER(:street_name)
                    AND building_name IS NOT NULL
                """
                building_candidates = self.db.execute_query(
                    building_query,
                    {"street_number": street_number, "street_name": street_name}
                )

                # Try to extract building name from location
                building_pattern = r'^([^,]+?)(?:,|\s+\d+)'
                building_match = re.search(building_pattern, location)

                if building_match and building_candidates:
                    potential_building = building_match.group(1).strip()

                    for addr_id, existing_building in building_candidates:
                        if self.fuzzy.compare(
                            potential_building,
                            existing_building,
                            threshold=80,
                            algorithm='token_set'
                        ):
                            self.logger.info(
                                f"quick_address_lookup: Building fuzzy match → "
                                f"address_id={addr_id}"
                            )
                            return addr_id

        return None

    def format_address_from_db_row(self, db_row) -> str:
        """
        Constructs a formatted address string from a database row.

        This method takes a database row object containing address components and constructs
        a single formatted address string. The formatted address includes the street address,
        city (municipality), province abbreviation, postal code, and country ("CA").
        Missing components are omitted gracefully.

        Args:
            db_row: An object representing a database row with address fields. Expected attributes are:
                - civic_no
                - civic_no_suffix
                - official_street_name
                - official_street_type
                - official_street_dir
                - mail_mun_name (city/municipality)
                - mail_prov_abvn (province abbreviation)
                - mail_postal_code

        Returns:
            str: A formatted address string in the form:
                "<street address>, <city>, <province abbreviation>, <postal code>, CA"
            Any missing components are omitted from the output.
        """
        # Build the street portion
        parts = [
            str(db_row.civic_no) if db_row.civic_no else "",
            str(db_row.civic_no_suffix) if db_row.civic_no_suffix else "",
            db_row.official_street_name or "",
            db_row.official_street_type or "",
            db_row.official_street_dir or ""
        ]
        street_address = " ".join(part for part in parts if part).strip()

        # Insert the city if available
        city = db_row.mail_mun_name or ""

        # Construct final location string
        formatted = (
            f"{street_address}, "
            f"{city}, "
            f"{db_row.mail_prov_abvn or ''}, "
            f"{db_row.mail_postal_code or ''}, CA"
        )

        # Clean up spacing
        formatted = re.sub(r'\s+,', ',', formatted)
        formatted = re.sub(r',\s+,', ',', formatted)
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        return formatted
