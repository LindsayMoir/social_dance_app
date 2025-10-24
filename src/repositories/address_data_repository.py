"""
Address Data Repository for data transformation and validation operations.

This module consolidates address data cleaning, transformation, and validation logic:
- Null value normalization across address records
- Canadian postal code extraction and validation
- Postal code standardization
- String cleaning in address fields
- Database row formatting

Previously scattered in DatabaseHandler, these utility methods handle consistent
data transformation, ensuring address data quality and consistency.

This repository focuses on data transformation and validation (Priority 7),
complementing AddressRepository (lookup/resolution) and AddressResolutionRepository (LLM processing).
"""

from typing import Optional, Dict, Any
import logging
import re
import pandas as pd
import numpy as np


class AddressDataRepository:
    """
    Repository for address data transformation, cleaning, and validation.

    Consolidates operations for:
    - Null value normalization (string nulls to Python None)
    - Canadian postal code extraction and validation
    - Postal code standardization to V8N 1S3 format
    - Null string cleanup in address fields
    - Database row formatting

    Key responsibilities:
    - Normalize various null representations
    - Validate and extract postal codes
    - Standardize address field formatting
    - Clean and transform raw address data
    """

    def __init__(self, db_handler):
        """
        Initialize AddressDataRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)

    def normalize_nulls(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replaces various null representations with Python None.

        Handles multiple null representations:
        - String literals: 'null', 'none', 'nan', 'n/a', 'na', 'nil', 'undefined'
        - Empty strings and whitespace
        - Pandas/NumPy NaN values
        - Other falsy values (except numeric 0)

        Args:
            record (dict): Dictionary with potentially null values

        Returns:
            dict: Dictionary with normalized None values
        """
        try:
            cleaned = {}
            null_strings = {"null", "none", "nan", "", "n/a", "na", "nil", "undefined"}

            for key, value in record.items():
                # Handle pandas/numpy NaN
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    cleaned[key] = None
                # Handle string nulls (case insensitive, strip whitespace)
                elif isinstance(value, str) and value.strip().lower() in null_strings:
                    cleaned[key] = None
                # Handle numeric 0 that shouldn't be null (keep as is)
                elif value == 0:
                    cleaned[key] = value
                # Handle other falsy values that should be None
                elif not value and value != 0:
                    cleaned[key] = None
                else:
                    cleaned[key] = value

            self.logger.info("normalize_nulls: Normalized %d key-value pairs", len(record))
            return cleaned

        except Exception as e:
            self.logger.error("normalize_nulls: Error during normalization: %s", e)
            raise

    def is_canadian_postal_code(self, postal_code: str) -> bool:
        """
        Validates Canadian postal code format.

        A valid Canadian postal code follows pattern: A1A 1A1
        Where 'A' is a letter and '1' is a digit. Space is optional.

        Args:
            postal_code (str): Postal code string to validate

        Returns:
            bool: True if valid Canadian format, False otherwise
        """
        try:
            if not postal_code or not isinstance(postal_code, str):
                return False

            pattern = r'^[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d$'
            result = bool(re.match(pattern, postal_code.strip()))

            if result:
                self.logger.debug("is_canadian_postal_code: Validated postal code '%s'", postal_code)
            else:
                self.logger.debug("is_canadian_postal_code: Invalid postal code '%s'", postal_code)

            return result

        except Exception as e:
            self.logger.warning("is_canadian_postal_code: Error validating '%s': %s", postal_code, e)
            return False

    def extract_canadian_postal_code(self, location_str: str) -> Optional[str]:
        """
        Extracts and validates Canadian postal code from location string.

        Searches for postal code pattern in string, validates it, and returns
        cleaned version without spaces.

        Args:
            location_str (str): Location string potentially containing postal code

        Returns:
            Optional[str]: Valid postal code without spaces, or None if not found
        """
        try:
            if not location_str or not isinstance(location_str, str):
                return None

            # Search for postal code pattern
            match = re.search(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d', location_str)
            if not match:
                return None

            # Extract and clean the match
            possible_pc = match.group().replace(' ', '').upper()

            # Validate the extracted postal code
            if self.is_canadian_postal_code(possible_pc):
                self.logger.info(
                    "extract_canadian_postal_code: Extracted valid postal code '%s' from '%s'",
                    possible_pc, location_str[:50]
                )
                return possible_pc

            return None

        except Exception as e:
            self.logger.warning("extract_canadian_postal_code: Error extracting from '%s': %s", location_str, e)
            return None

    def standardize_postal_codes(self) -> Optional[int]:
        """
        Standardizes Canadian postal codes in database to V8N 1S3 format.

        Converts postal codes to uppercase with space between 3rd and 4th characters.
        Only processes codes matching Canadian postal code pattern.

        Returns:
            Optional[int]: Number of records updated, or None if error
        """
        try:
            query = """
                UPDATE address
                SET postal_code = UPPER(
                    CASE
                        WHEN LENGTH(REPLACE(postal_code, ' ', '')) = 6
                        THEN SUBSTRING(REPLACE(postal_code, ' ', ''), 1, 3) || ' ' ||
                             SUBSTRING(REPLACE(postal_code, ' ', ''), 4, 3)
                        ELSE postal_code
                    END
                )
                WHERE postal_code IS NOT NULL
                AND postal_code ~ '^[A-Za-z][0-9][A-Za-z][ ]?[0-9][A-Za-z][0-9]$'
            """

            result = self.db.execute_query(query)
            affected_rows = result if result else 0

            self.logger.info("standardize_postal_codes: Standardized %d postal code records", affected_rows)
            return affected_rows

        except Exception as e:
            self.logger.error("standardize_postal_codes: Failed to standardize postal codes: %s", e)
            raise

    def clean_null_strings_in_address(self) -> None:
        """
        Replaces string null representations with SQL NULL in address table.

        Replaces string literals like 'null', 'none', 'nan', etc. with actual NULL
        values in all address fields.

        Raises:
            Exception: If database operation fails
        """
        try:
            fields = [
                "full_address", "building_name", "street_number", "street_name", "direction",
                "city", "met_area", "province_or_state", "postal_code", "country_id"
            ]

            total_updated = 0

            for field in fields:
                query = f"""
                    UPDATE address
                    SET {field} = NULL
                    WHERE TRIM(LOWER({field})) IN ('null', 'none', 'nan', '', '[null]',
                                                    '(null)', 'n/a', 'na');
                """
                result = self.db.execute_query(query)
                field_count = result if result else 0
                total_updated += field_count

                self.logger.debug("clean_null_strings_in_address: Cleaned %d records in %s", field_count, field)

            self.logger.info("clean_null_strings_in_address: Total %d null strings replaced", total_updated)

        except Exception as e:
            self.logger.error("clean_null_strings_in_address: Failed to clean null strings: %s", e)
            raise

    def format_address_from_db_row(self, db_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats database address row into standardized address dictionary.

        Converts database column names to standardized address format and
        ensures all fields are present with appropriate defaults.

        Args:
            db_row (dict): Row from address table with database column names

        Returns:
            dict: Standardized address dictionary with all expected fields
        """
        try:
            formatted = {
                'address_id': db_row.get('address_id'),
                'full_address': db_row.get('full_address'),
                'building_name': db_row.get('building_name'),
                'street_number': db_row.get('street_number'),
                'street_name': db_row.get('street_name'),
                'street_type': db_row.get('street_type'),
                'direction': db_row.get('direction'),
                'city': db_row.get('city'),
                'met_area': db_row.get('met_area'),
                'province_or_state': db_row.get('province_or_state'),
                'postal_code': db_row.get('postal_code'),
                'country_id': db_row.get('country_id'),
                'time_stamp': db_row.get('time_stamp')
            }

            self.logger.debug("format_address_from_db_row: Formatted address_id=%s", db_row.get('address_id'))
            return formatted

        except Exception as e:
            self.logger.error("format_address_from_db_row: Error formatting row: %s", e)
            raise
