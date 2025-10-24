"""Tests for AddressDataRepository."""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.repositories.address_data_repository import AddressDataRepository


class TestAddressDataRepository:
    """Test suite for AddressDataRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.execute_query = Mock(return_value=0)
        return mock_db

    @pytest.fixture
    def repo(self, mock_db_handler):
        """Create an AddressDataRepository instance."""
        return AddressDataRepository(mock_db_handler)

    # =====================================================================
    # NORMALIZE NULLS TESTS
    # =====================================================================

    def test_initialization(self, mock_db_handler):
        """Test AddressDataRepository initialization."""
        repo = AddressDataRepository(mock_db_handler)
        assert repo.db is mock_db_handler

    def test_normalize_nulls_string_nulls(self, repo):
        """Test normalizing various string null representations."""
        record = {
            'name': 'null',
            'city': 'NONE',
            'postal': 'nan',
            'street': 'n/a',
            'building': 'nil',
            'direction': 'undefined'
        }

        result = repo.normalize_nulls(record)

        assert result['name'] is None
        assert result['city'] is None
        assert result['postal'] is None
        assert result['street'] is None
        assert result['building'] is None
        assert result['direction'] is None

    def test_normalize_nulls_empty_strings(self, repo):
        """Test normalizing empty strings."""
        record = {
            'field1': '',
            'field2': '   ',
            'field3': 'value'
        }

        result = repo.normalize_nulls(record)

        assert result['field1'] is None
        assert result['field2'] is None
        assert result['field3'] == 'value'

    def test_normalize_nulls_pandas_nan(self, repo):
        """Test normalizing pandas NaN values."""
        record = {
            'field1': float('nan'),
            'field2': pd.NA,
            'field3': np.nan
        }

        result = repo.normalize_nulls(record)

        assert result['field1'] is None
        assert result['field2'] is None
        assert result['field3'] is None

    def test_normalize_nulls_numeric_zero(self, repo):
        """Test that numeric 0 is preserved."""
        record = {
            'number': 0,
            'float_zero': 0.0,
            'negative': -5
        }

        result = repo.normalize_nulls(record)

        assert result['number'] == 0
        assert result['float_zero'] == 0.0
        assert result['negative'] == -5

    def test_normalize_nulls_preserves_values(self, repo):
        """Test that valid values are preserved."""
        record = {
            'street': '123 Main St',
            'city': 'Vancouver',
            'postal': 'V8N 1S3',
            'count': 42,
            'flag': True
        }

        result = repo.normalize_nulls(record)

        assert result['street'] == '123 Main St'
        assert result['city'] == 'Vancouver'
        assert result['postal'] == 'V8N 1S3'
        assert result['count'] == 42
        assert result['flag'] is True

    def test_normalize_nulls_mixed_record(self, repo):
        """Test normalization with mixed null and valid values."""
        record = {
            'full_address': 'null',
            'street_number': '456',
            'street_name': 'n/a',
            'city': 'Victoria',
            'postal_code': None,
            'country': 'Canada'
        }

        result = repo.normalize_nulls(record)

        assert result['full_address'] is None
        assert result['street_number'] == '456'
        assert result['street_name'] is None
        assert result['city'] == 'Victoria'
        assert result['postal_code'] is None
        assert result['country'] == 'Canada'

    # =====================================================================
    # CANADIAN POSTAL CODE VALIDATION TESTS
    # =====================================================================

    def test_is_canadian_postal_code_valid_with_space(self, repo):
        """Test validation of valid Canadian postal code with space."""
        assert repo.is_canadian_postal_code('V8N 1S3') is True

    def test_is_canadian_postal_code_valid_without_space(self, repo):
        """Test validation of valid Canadian postal code without space."""
        assert repo.is_canadian_postal_code('V8N1S3') is True

    def test_is_canadian_postal_code_lowercase(self, repo):
        """Test validation with lowercase postal code."""
        assert repo.is_canadian_postal_code('v8n 1s3') is True

    def test_is_canadian_postal_code_mixed_case(self, repo):
        """Test validation with mixed case."""
        assert repo.is_canadian_postal_code('V8n 1s3') is True

    def test_is_canadian_postal_code_invalid_pattern(self, repo):
        """Test validation of invalid postal code pattern."""
        assert repo.is_canadian_postal_code('12345') is False
        assert repo.is_canadian_postal_code('ABCDEF') is False
        assert repo.is_canadian_postal_code('V8N1S') is False
        assert repo.is_canadian_postal_code('8VN 1S3') is False

    def test_is_canadian_postal_code_none(self, repo):
        """Test validation with None value."""
        assert repo.is_canadian_postal_code(None) is False

    def test_is_canadian_postal_code_empty_string(self, repo):
        """Test validation with empty string."""
        assert repo.is_canadian_postal_code('') is False

    def test_is_canadian_postal_code_whitespace_only(self, repo):
        """Test validation with whitespace only."""
        assert repo.is_canadian_postal_code('   ') is False

    def test_is_canadian_postal_code_with_leading_trailing_space(self, repo):
        """Test validation with leading/trailing spaces."""
        assert repo.is_canadian_postal_code('  V8N 1S3  ') is True

    # =====================================================================
    # EXTRACT CANADIAN POSTAL CODE TESTS
    # =====================================================================

    def test_extract_canadian_postal_code_simple(self, repo):
        """Test extracting postal code from simple string."""
        result = repo.extract_canadian_postal_code('123 Main St V8N 1S3')

        assert result == 'V8N1S3'

    def test_extract_canadian_postal_code_with_spaces(self, repo):
        """Test extracting postal code with spaces."""
        result = repo.extract_canadian_postal_code('Address: V8N 1S3 Victoria')

        assert result == 'V8N1S3'

    def test_extract_canadian_postal_code_no_space(self, repo):
        """Test extracting postal code without internal space."""
        result = repo.extract_canadian_postal_code('Location V8N1S3 here')

        assert result == 'V8N1S3'

    def test_extract_canadian_postal_code_full_address(self, repo):
        """Test extracting from full address string."""
        location = '456 Elm Street, Victoria BC V6R 2Z8 Canada'
        result = repo.extract_canadian_postal_code(location)

        assert result == 'V6R2Z8'

    def test_extract_canadian_postal_code_not_found(self, repo):
        """Test when no postal code in string."""
        result = repo.extract_canadian_postal_code('123 Main Street, City')

        assert result is None

    def test_extract_canadian_postal_code_invalid_pattern(self, repo):
        """Test when string contains invalid postal pattern."""
        result = repo.extract_canadian_postal_code('Address 12345 or ABC DEF')

        assert result is None

    def test_extract_canadian_postal_code_none(self, repo):
        """Test with None input."""
        result = repo.extract_canadian_postal_code(None)

        assert result is None

    def test_extract_canadian_postal_code_empty_string(self, repo):
        """Test with empty string."""
        result = repo.extract_canadian_postal_code('')

        assert result is None

    def test_extract_canadian_postal_code_multiple_codes(self, repo):
        """Test extracting first postal code when multiple present."""
        result = repo.extract_canadian_postal_code('V8N 1S3 and V6R 2Z8')

        # Should extract the first match
        assert result == 'V8N1S3'

    # =====================================================================
    # STANDARDIZE POSTAL CODES TESTS
    # =====================================================================

    def test_standardize_postal_codes_success(self, repo, mock_db_handler):
        """Test successful postal code standardization."""
        mock_db_handler.execute_query.return_value = 15

        result = repo.standardize_postal_codes()

        assert result == 15
        mock_db_handler.execute_query.assert_called_once()

    def test_standardize_postal_codes_no_updates(self, repo, mock_db_handler):
        """Test when no postal codes need standardization."""
        mock_db_handler.execute_query.return_value = 0

        result = repo.standardize_postal_codes()

        assert result == 0

    def test_standardize_postal_codes_failure(self, repo, mock_db_handler):
        """Test failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            repo.standardize_postal_codes()

    # =====================================================================
    # CLEAN NULL STRINGS TESTS
    # =====================================================================

    def test_clean_null_strings_in_address_success(self, repo, mock_db_handler):
        """Test successful cleanup of null strings."""
        mock_db_handler.execute_query.return_value = 5

        repo.clean_null_strings_in_address()

        # Should call execute_query for each field (10 fields)
        assert mock_db_handler.execute_query.call_count == 10

    def test_clean_null_strings_in_address_no_nulls(self, repo, mock_db_handler):
        """Test when no null strings to clean."""
        mock_db_handler.execute_query.return_value = 0

        repo.clean_null_strings_in_address()

        # Should still call execute_query for each field
        assert mock_db_handler.execute_query.call_count == 10

    def test_clean_null_strings_in_address_failure(self, repo, mock_db_handler):
        """Test failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            repo.clean_null_strings_in_address()

    # =====================================================================
    # FORMAT ADDRESS FROM DB ROW TESTS
    # =====================================================================

    def test_format_address_from_db_row_complete(self, repo):
        """Test formatting complete database row."""
        db_row = {
            'address_id': 42,
            'full_address': '123 Main St, Vancouver BC V8N 1S3',
            'building_name': 'City Hall',
            'street_number': '123',
            'street_name': 'Main',
            'street_type': 'St',
            'direction': None,
            'city': 'Vancouver',
            'met_area': 'Greater Vancouver',
            'province_or_state': 'BC',
            'postal_code': 'V8N 1S3',
            'country_id': 'CA',
            'time_stamp': '2025-10-23'
        }

        result = repo.format_address_from_db_row(db_row)

        assert result['address_id'] == 42
        assert result['full_address'] == '123 Main St, Vancouver BC V8N 1S3'
        assert result['building_name'] == 'City Hall'
        assert result['city'] == 'Vancouver'
        assert result['postal_code'] == 'V8N 1S3'
        assert result['country_id'] == 'CA'

    def test_format_address_from_db_row_partial(self, repo):
        """Test formatting partial database row."""
        db_row = {
            'address_id': 10,
            'full_address': 'Unknown Address',
            'city': 'Victoria'
        }

        result = repo.format_address_from_db_row(db_row)

        assert result['address_id'] == 10
        assert result['full_address'] == 'Unknown Address'
        assert result['city'] == 'Victoria'
        assert result['building_name'] is None
        assert result['street_name'] is None

    def test_format_address_from_db_row_empty(self, repo):
        """Test formatting empty row."""
        result = repo.format_address_from_db_row({})

        assert len(result) == 13  # All expected fields
        assert all(v is None for v in result.values())

    def test_format_address_from_db_row_extra_fields(self, repo):
        """Test that extra fields are ignored."""
        db_row = {
            'address_id': 5,
            'city': 'Vancouver',
            'extra_field': 'should be ignored',
            'another_extra': 123
        }

        result = repo.format_address_from_db_row(db_row)

        assert 'extra_field' not in result
        assert 'another_extra' not in result
        assert len(result) == 13

    # =====================================================================
    # ERROR HANDLING TESTS
    # =====================================================================

    def test_normalize_nulls_error_handling(self, repo):
        """Test error handling in normalize_nulls."""
        # Create a mock with problematic iteration
        record = {'key': 'value'}

        # Should not raise, just log error
        result = repo.normalize_nulls(record)
        assert result['key'] == 'value'

    def test_is_canadian_postal_code_error_handling(self, repo):
        """Test error handling in postal code validation."""
        # Test with non-string types
        assert repo.is_canadian_postal_code(12345) is False
        assert repo.is_canadian_postal_code([]) is False

    def test_extract_postal_code_error_handling(self, repo):
        """Test error handling in extraction."""
        # Test with non-string types
        result = repo.extract_canadian_postal_code(12345)
        assert result is None

    def test_format_address_error_handling(self, repo):
        """Test error handling in formatting."""
        # Should handle None gracefully
        with pytest.raises(Exception):
            repo.format_address_from_db_row(None)

    # =====================================================================
    # EDGE CASE TESTS
    # =====================================================================

    def test_normalize_nulls_unicode(self, repo):
        """Test normalization with unicode characters."""
        record = {
            'name': 'Café',
            'city': 'Montréal',
            'null_value': 'null'
        }

        result = repo.normalize_nulls(record)

        assert result['name'] == 'Café'
        assert result['city'] == 'Montréal'
        assert result['null_value'] is None

    def test_extract_postal_code_unicode(self, repo):
        """Test extraction from unicode string."""
        result = repo.extract_canadian_postal_code('Rue Main, Montréal QC H1A 1A1')

        assert result == 'H1A1A1'

    def test_postal_code_case_variations(self, repo):
        """Test various case combinations."""
        test_cases = [
            ('V8n 1s3', True),
            ('v8N 1S3', True),
            ('V8N1s3', True),
            ('v8n1s3', True)
        ]

        for postal, expected in test_cases:
            assert repo.is_canadian_postal_code(postal) == expected
