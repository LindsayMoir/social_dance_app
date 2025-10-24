"""Tests for LocationCacheRepository."""
import pytest
from unittest.mock import Mock
from datetime import datetime
from src.repositories.location_cache_repository import LocationCacheRepository


class TestLocationCacheRepository:
    """Test suite for LocationCacheRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.execute_query = Mock(return_value=None)
        return mock_db

    @pytest.fixture
    def repo(self, mock_db_handler):
        """Create a LocationCacheRepository instance."""
        return LocationCacheRepository(mock_db_handler)

    # =====================================================================
    # INITIALIZATION & BASIC TESTS
    # =====================================================================

    def test_initialization(self, mock_db_handler):
        """Test LocationCacheRepository initialization."""
        repo = LocationCacheRepository(mock_db_handler)
        assert repo.db is mock_db_handler
        assert repo._building_name_cache == {}

    # =====================================================================
    # BUILDING NAME DICTIONARY TESTS
    # =====================================================================

    def test_get_building_name_dictionary_success(self, repo, mock_db_handler):
        """Test successful building name dictionary creation."""
        mock_db_handler.execute_query.return_value = [
            (1, "Duke Saloon"),
            (2, "The Ritz"),
            (3, "City Hall")
        ]

        result = repo._get_building_name_dictionary()

        assert len(result) == 3
        assert result["duke saloon"] == 1
        assert result["the ritz"] == 2
        assert result["city hall"] == 3

    def test_get_building_name_dictionary_empty(self, repo, mock_db_handler):
        """Test building dictionary when no results."""
        mock_db_handler.execute_query.return_value = None

        result = repo._get_building_name_dictionary()

        assert result == {}

    def test_get_building_name_dictionary_caching(self, repo, mock_db_handler):
        """Test that building dictionary is cached after first call."""
        mock_db_handler.execute_query.return_value = [
            (1, "Venue A"),
            (2, "Venue B")
        ]

        # First call
        result1 = repo._get_building_name_dictionary()
        assert mock_db_handler.execute_query.call_count == 1

        # Second call should return cached version without querying DB
        result2 = repo._get_building_name_dictionary()
        assert mock_db_handler.execute_query.call_count == 1  # No additional call

        assert result1 == result2

    def test_get_building_name_dictionary_case_insensitive(self, repo, mock_db_handler):
        """Test that building names are normalized to lowercase."""
        mock_db_handler.execute_query.return_value = [
            (1, "Duke Saloon"),
            (2, "THE RITZ"),
            (3, "City HALL")
        ]

        result = repo._get_building_name_dictionary()

        assert "duke saloon" in result
        assert "the ritz" in result
        assert "city hall" in result

    def test_get_building_name_dictionary_whitespace_handling(self, repo, mock_db_handler):
        """Test that whitespace is trimmed from building names."""
        mock_db_handler.execute_query.return_value = [
            (1, "  Duke Saloon  "),
            (2, "\tThe Ritz\n")
        ]

        result = repo._get_building_name_dictionary()

        assert "duke saloon" in result
        assert "the ritz" in result

    def test_get_building_name_dictionary_skips_empty(self, repo, mock_db_handler):
        """Test that empty building names are skipped."""
        mock_db_handler.execute_query.return_value = [
            (1, "Venue A"),
            (2, ""),
            (3, "   "),
            (4, "Venue B")
        ]

        result = repo._get_building_name_dictionary()

        assert len(result) == 2
        assert "venue a" in result
        assert "venue b" in result

    def test_get_building_name_dictionary_error_handling(self, repo, mock_db_handler):
        """Test error handling in building dictionary creation."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = repo._get_building_name_dictionary()

        assert result == {}

    # =====================================================================
    # CACHE RAW LOCATION TESTS
    # =====================================================================

    def test_cache_raw_location_success(self, repo, mock_db_handler):
        """Test successful caching of raw location."""
        mock_db_handler.execute_query.return_value = True

        result = repo.cache_raw_location("Main Street, Vancouver", 42)

        assert result is True
        mock_db_handler.execute_query.assert_called_once()

    def test_cache_raw_location_conflict_ignored(self, repo, mock_db_handler):
        """Test that duplicate caching (ON CONFLICT DO NOTHING) returns True."""
        mock_db_handler.execute_query.return_value = 0  # No rows inserted due to conflict

        result = repo.cache_raw_location("Main Street, Vancouver", 42)

        assert result is True  # Still returns True, conflict is handled gracefully

    def test_cache_raw_location_error_handling(self, repo, mock_db_handler):
        """Test error handling in location caching."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = repo.cache_raw_location("Main Street, Vancouver", 42)

        assert result is False

    def test_cache_raw_location_long_string(self, repo, mock_db_handler):
        """Test caching with very long location strings."""
        mock_db_handler.execute_query.return_value = True
        long_location = "A" * 1000

        result = repo.cache_raw_location(long_location, 1)

        assert result is True
        mock_db_handler.execute_query.assert_called_once()

    def test_cache_raw_location_special_characters(self, repo, mock_db_handler):
        """Test caching with special characters."""
        mock_db_handler.execute_query.return_value = True
        location = "123 Main St, Vancouver BC V8N 1S3 (Near Park)"

        result = repo.cache_raw_location(location, 10)

        assert result is True

    # =====================================================================
    # LOOKUP RAW LOCATION TESTS
    # =====================================================================

    def test_lookup_raw_location_found(self, repo, mock_db_handler):
        """Test successful lookup of cached location."""
        mock_db_handler.execute_query.return_value = [(42,)]

        result = repo.lookup_raw_location("Main Street, Vancouver")

        assert result == 42

    def test_lookup_raw_location_not_found(self, repo, mock_db_handler):
        """Test lookup when location not in cache."""
        mock_db_handler.execute_query.return_value = None

        result = repo.lookup_raw_location("Unknown Location")

        assert result is None

    def test_lookup_raw_location_empty_result(self, repo, mock_db_handler):
        """Test lookup with empty result list."""
        mock_db_handler.execute_query.return_value = []

        result = repo.lookup_raw_location("Location")

        assert result is None

    def test_lookup_raw_location_error_handling(self, repo, mock_db_handler):
        """Test error handling in location lookup."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = repo.lookup_raw_location("Main Street, Vancouver")

        assert result is None

    def test_lookup_raw_location_multiple_results(self, repo, mock_db_handler):
        """Test lookup when multiple results returned (should use first)."""
        mock_db_handler.execute_query.return_value = [(42,), (99,)]

        result = repo.lookup_raw_location("Main Street")

        assert result == 42  # Returns first result

    # =====================================================================
    # CREATE RAW LOCATIONS TABLE TESTS
    # =====================================================================

    def test_create_raw_locations_table_success(self, repo, mock_db_handler):
        """Test successful creation of raw_locations table."""
        mock_db_handler.execute_query.return_value = True

        result = repo.create_raw_locations_table()

        assert result is True
        # Should be called 3 times: address table, raw_locations table, index
        assert mock_db_handler.execute_query.call_count == 3

    def test_create_raw_locations_table_already_exists(self, repo, mock_db_handler):
        """Test creation when table already exists (IF NOT EXISTS)."""
        mock_db_handler.execute_query.return_value = None

        result = repo.create_raw_locations_table()

        assert result is True

    def test_create_raw_locations_table_failure(self, repo, mock_db_handler):
        """Test error handling in table creation."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = repo.create_raw_locations_table()

        assert result is False

    def test_create_raw_locations_table_partial_failure(self, repo, mock_db_handler):
        """Test when table creation fails partway through."""
        # First two calls succeed, third (index) fails
        mock_db_handler.execute_query.side_effect = [
            True,  # address table
            True,  # raw_locations table
            Exception("Index creation failed")
        ]

        result = repo.create_raw_locations_table()

        assert result is False

    # =====================================================================
    # CLEAR BUILDING CACHE TESTS
    # =====================================================================

    def test_clear_building_cache_empty(self, repo):
        """Test clearing empty cache."""
        repo._building_name_cache = {}

        repo.clear_building_cache()

        assert repo._building_name_cache == {}

    def test_clear_building_cache_with_data(self, repo):
        """Test clearing cache with data."""
        repo._building_name_cache = {
            "venue 1": 1,
            "venue 2": 2,
            "venue 3": 3
        }

        repo.clear_building_cache()

        assert repo._building_name_cache == {}

    def test_clear_building_cache_allows_rebuild(self, repo, mock_db_handler):
        """Test that cache can be rebuilt after clearing."""
        # Set initial cache
        repo._building_name_cache = {"old": 1}

        # Clear it
        repo.clear_building_cache()
        assert repo._building_name_cache == {}

        # Mock DB for rebuilding
        mock_db_handler.execute_query.return_value = [
            (10, "New Venue")
        ]

        # Rebuild
        result = repo._get_building_name_dictionary()
        assert "new venue" in result
        assert result["new venue"] == 10

    def test_clear_building_cache_error_handling(self, repo):
        """Test error handling in cache clearing."""
        # This shouldn't raise, just log
        repo.clear_building_cache()
        assert repo._building_name_cache == {}

    # =====================================================================
    # INTEGRATION TESTS
    # =====================================================================

    def test_cache_and_lookup_flow(self, repo, mock_db_handler):
        """Test complete cache and lookup flow."""
        # Mock cache operation
        mock_db_handler.execute_query.return_value = True
        repo.cache_raw_location("Test Location", 99)

        # Mock lookup operation
        mock_db_handler.execute_query.return_value = [(99,)]
        result = repo.lookup_raw_location("Test Location")

        assert result == 99

    def test_building_dictionary_and_lookup(self, repo, mock_db_handler):
        """Test building name dictionary with case-insensitive lookup."""
        mock_db_handler.execute_query.return_value = [
            (1, "Duke Saloon"),
            (2, "The Ritz Hotel")
        ]

        building_dict = repo._get_building_name_dictionary()

        # Test case-insensitive access
        assert building_dict.get("DUKE SALOON".lower()) == 1
        assert building_dict.get("the ritz hotel".lower()) == 2

    def test_multiple_cache_operations(self, repo, mock_db_handler):
        """Test multiple cache and lookup operations."""
        mock_db_handler.execute_query.return_value = True

        # Cache multiple locations
        assert repo.cache_raw_location("Location 1", 1) is True
        assert repo.cache_raw_location("Location 2", 2) is True
        assert repo.cache_raw_location("Location 3", 3) is True

        assert mock_db_handler.execute_query.call_count == 3

    # =====================================================================
    # EDGE CASE TESTS
    # =====================================================================

    def test_cache_with_unicode(self, repo, mock_db_handler):
        """Test caching with unicode characters."""
        mock_db_handler.execute_query.return_value = True
        location = "Rue Main, Montréal QC"

        result = repo.cache_raw_location(location, 1)

        assert result is True

    def test_lookup_with_unicode(self, repo, mock_db_handler):
        """Test lookup with unicode characters."""
        mock_db_handler.execute_query.return_value = [(5,)]
        location = "Café Downtown, Vancouver"

        result = repo.lookup_raw_location(location)

        assert result == 5

    def test_building_name_with_special_characters(self, repo, mock_db_handler):
        """Test building names with special characters."""
        mock_db_handler.execute_query.return_value = [
            (1, "O'Reilly's Pub"),
            (2, "Joe's Bar & Grill"),
            (3, "The Duke's Arms")
        ]

        result = repo._get_building_name_dictionary()

        assert len(result) == 3
        assert "o'reilly's pub" in result
        assert "joe's bar & grill" in result
        assert "the duke's arms" in result

    def test_zero_address_id(self, repo, mock_db_handler):
        """Test caching and lookup with address_id of 0."""
        mock_db_handler.execute_query.return_value = True
        repo.cache_raw_location("Location", 0)

        mock_db_handler.execute_query.return_value = [(0,)]
        result = repo.lookup_raw_location("Location")

        assert result == 0  # Zero address_id is valid

    def test_negative_address_id(self, repo, mock_db_handler):
        """Test handling of invalid negative address_id."""
        mock_db_handler.execute_query.return_value = True

        result = repo.cache_raw_location("Location", -1)

        assert result is True  # Method doesn't validate, just caches
