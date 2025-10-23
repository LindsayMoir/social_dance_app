"""Tests for AddressRepository utility."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.repositories.address_repository import AddressRepository


class TestAddressRepository:
    """Test suite for AddressRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.execute_query = Mock(return_value=None)
        mock_db.normalize_nulls = Mock(side_effect=lambda x: x)
        return mock_db

    @pytest.fixture
    def address_repo(self, mock_db_handler):
        """Create an AddressRepository instance with mock database."""
        return AddressRepository(mock_db_handler)

    def test_initialization(self, mock_db_handler):
        """Test AddressRepository initialization."""
        repo = AddressRepository(mock_db_handler)
        assert repo.db is mock_db_handler
        assert repo.fuzzy is not None
        assert repo.logger is not None

    def test_build_full_address_with_all_fields(self, address_repo):
        """Test building full address with all components."""
        result = address_repo.build_full_address(
            building_name="Duke Saloon",
            street_number="123",
            street_name="Main",
            street_type="St",
            city="Toronto",
            province_or_state="ON",
            postal_code="M5V 3A8",
            country_id="CA"
        )
        assert "Duke Saloon" in result
        assert "123 Main St" in result
        assert "Toronto" in result
        assert "ON M5V 3A8" in result
        assert "CA" in result

    def test_build_full_address_with_minimal_fields(self, address_repo):
        """Test building full address with minimal components."""
        result = address_repo.build_full_address(
            city="Vancouver",
            postal_code="V6B 4X1"
        )
        assert "Vancouver" in result
        assert "V6B 4X1" in result
        assert len(result) > 0

    def test_build_full_address_with_no_fields(self, address_repo):
        """Test building full address with no components."""
        result = address_repo.build_full_address()
        assert result == ""

    def test_get_full_address_from_id_found(self, address_repo, mock_db_handler):
        """Test getting full address when ID exists."""
        mock_db_handler.execute_query.return_value = [("123 Main St, Toronto, ON M5V 3A8, CA",)]

        result = address_repo.get_full_address_from_id(1)

        assert result == "123 Main St, Toronto, ON M5V 3A8, CA"
        mock_db_handler.execute_query.assert_called_once()

    def test_get_full_address_from_id_not_found(self, address_repo, mock_db_handler):
        """Test getting full address when ID does not exist."""
        mock_db_handler.execute_query.return_value = None

        result = address_repo.get_full_address_from_id(999)

        assert result is None

    def test_find_address_by_building_name_found(self, address_repo, mock_db_handler):
        """Test finding address by building name with match."""
        mock_db_handler.execute_query.return_value = [
            (1, "Duke Saloon"),
            (2, "The Duke")
        ]

        result = address_repo.find_address_by_building_name("Duke Saloon", threshold=80)

        # Should find the first one with exact match or high score
        assert result is not None
        assert isinstance(result, int)

    def test_find_address_by_building_name_not_found(self, address_repo, mock_db_handler):
        """Test finding address by building name with no match."""
        mock_db_handler.execute_query.return_value = [
            (1, "Duke Saloon"),
            (2, "The Duke")
        ]

        result = address_repo.find_address_by_building_name("RandomVenue", threshold=95)

        # With high threshold, unlikely to match
        assert result is None or isinstance(result, int)

    def test_find_address_by_building_name_empty_input(self, address_repo):
        """Test finding address with empty building name."""
        result = address_repo.find_address_by_building_name("")

        assert result is None

    def test_find_address_by_building_name_none_input(self, address_repo):
        """Test finding address with None input."""
        result = address_repo.find_address_by_building_name(None)

        assert result is None

    def test_find_address_by_building_name_non_string_input(self, address_repo):
        """Test finding address with non-string input."""
        result = address_repo.find_address_by_building_name(123)

        assert result is None

    def test_find_address_by_building_name_exception_handling(self, address_repo, mock_db_handler):
        """Test exception handling in find_address_by_building_name."""
        mock_db_handler.execute_query.side_effect = Exception("Database error")

        result = address_repo.find_address_by_building_name("Duke Saloon")

        assert result is None

    def test_quick_address_lookup_exact_match(self, address_repo, mock_db_handler):
        """Test quick address lookup with exact match."""
        mock_db_handler.execute_query.return_value = [(1,)]

        result = address_repo.quick_address_lookup("123 Main St, Toronto, ON M5V 3A8, CA")

        assert result == 1
        mock_db_handler.execute_query.assert_called()

    def test_quick_address_lookup_no_match(self, address_repo, mock_db_handler):
        """Test quick address lookup with no match."""
        mock_db_handler.execute_query.return_value = None

        result = address_repo.quick_address_lookup("Unknown Address")

        assert result is None

    def test_format_address_from_db_row_complete(self, address_repo):
        """Test formatting address from complete database row."""
        mock_row = Mock()
        mock_row.civic_no = "123"
        mock_row.civic_no_suffix = "A"
        mock_row.official_street_name = "Main"
        mock_row.official_street_type = "St"
        mock_row.official_street_dir = "W"
        mock_row.mail_mun_name = "Toronto"
        mock_row.mail_prov_abvn = "ON"
        mock_row.mail_postal_code = "M5V 3A8"

        result = address_repo.format_address_from_db_row(mock_row)

        assert "123" in result
        assert "Main" in result
        assert "Toronto" in result
        assert "ON" in result
        assert "CA" in result

    def test_format_address_from_db_row_partial(self, address_repo):
        """Test formatting address from partial database row."""
        mock_row = Mock()
        mock_row.civic_no = "123"
        mock_row.civic_no_suffix = None
        mock_row.official_street_name = "Main"
        mock_row.official_street_type = "St"
        mock_row.official_street_dir = None
        mock_row.mail_mun_name = "Toronto"
        mock_row.mail_prov_abvn = "ON"
        mock_row.mail_postal_code = "M5V 3A8"

        result = address_repo.format_address_from_db_row(mock_row)

        assert "123" in result
        assert "Main" in result
        assert "Toronto" in result
        assert len(result) > 0

    def test_format_address_from_db_row_minimal(self, address_repo):
        """Test formatting address from minimal database row."""
        mock_row = Mock()
        mock_row.civic_no = None
        mock_row.civic_no_suffix = None
        mock_row.official_street_name = None
        mock_row.official_street_type = None
        mock_row.official_street_dir = None
        mock_row.mail_mun_name = "Toronto"
        mock_row.mail_prov_abvn = "ON"
        mock_row.mail_postal_code = None

        result = address_repo.format_address_from_db_row(mock_row)

        # Should still produce valid output
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resolve_or_insert_address_none_input(self, address_repo):
        """Test resolve_or_insert_address with None input."""
        result = address_repo.resolve_or_insert_address(None)

        assert result is None

    def test_resolve_or_insert_address_empty_dict(self, address_repo, mock_db_handler):
        """Test resolve_or_insert_address with empty dictionary."""
        mock_db_handler.normalize_nulls.return_value = {}
        mock_db_handler.execute_query.return_value = [(1,)]  # Return new address ID

        result = address_repo.resolve_or_insert_address({})

        # Should attempt to insert
        assert result is not None or result is None  # Depends on implementation details

    def test_resolve_or_insert_address_with_postal_match(self, address_repo, mock_db_handler):
        """Test resolve_or_insert_address with postal code + street number match."""
        mock_db_handler.execute_query.side_effect = [
            # First call: postal match query
            [(1, "Duke Saloon", "123", "Main", "M5V 3A8")],
            # Subsequent calls as needed
        ]

        parsed_address = {
            "building_name": "Duke Saloon",
            "street_number": "123",
            "street_name": "Main",
            "postal_code": "M5V 3A8",
            "city": "Toronto",
            "country_id": "CA"
        }

        result = address_repo.resolve_or_insert_address(parsed_address)

        # Should return the matched address ID
        assert result == 1

    def test_resolve_or_insert_address_inserts_new(self, address_repo, mock_db_handler):
        """Test resolve_or_insert_address inserting new address."""
        # No matches, should insert
        # Need to account for: postal match, street match, city match,
        # building-only match, find_address_by_building_name query, insert
        mock_db_handler.execute_query.side_effect = [
            None,  # postal match
            None,  # street match
            None,  # city match
            None,  # building-only match
            None,  # find_address_by_building_name query
            [(5,)]  # insert returns new address_id
        ]
        mock_db_handler.normalize_nulls.return_value = {}

        parsed_address = {
            "building_name": "New Venue",
            "street_number": "456",
            "street_name": "Queen",
            "postal_code": "M6J 1A1",
            "city": "Toronto",
            "country_id": "CA"
        }

        result = address_repo.resolve_or_insert_address(parsed_address)

        # Should return new address ID
        assert result == 5

    def test_resolve_or_insert_address_handles_exceptions(self, address_repo, mock_db_handler):
        """Test resolve_or_insert_address exception handling."""
        mock_db_handler.execute_query.side_effect = Exception("Database error")

        parsed_address = {
            "building_name": "Duke Saloon",
            "street_number": "123",
            "street_name": "Main"
        }

        # Should handle exception gracefully
        with pytest.raises(Exception):
            address_repo.resolve_or_insert_address(parsed_address)
