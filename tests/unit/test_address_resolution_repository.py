"""Tests for AddressResolutionRepository."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd
from src.repositories.address_resolution_repository import AddressResolutionRepository


class TestAddressResolutionRepository:
    """Test suite for AddressResolutionRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_db.lookup_raw_location = Mock(return_value=None)
        mock_db.quick_address_lookup = Mock(return_value=None)
        mock_db.cache_raw_location = Mock()
        mock_db.get_full_address_from_id = Mock(return_value="123 Main St, Vancouver, BC")
        mock_db.find_address_by_building_name = Mock(return_value=None)
        mock_db._get_building_name_dictionary = Mock(return_value={})
        mock_db.resolve_or_insert_address = Mock(return_value=1)
        mock_db.normalize_nulls = Mock(side_effect=lambda x: x)
        return mock_db

    @pytest.fixture
    def mock_llm_handler(self):
        """Create a mock LLMHandler for testing."""
        mock_llm = Mock()
        mock_llm.generate_prompt = Mock(return_value=("prompt", "schema"))
        mock_llm.query_llm = Mock(return_value='{"street_name": "Main St"}')
        mock_llm.extract_and_parse_json = Mock(return_value=[{"street_name": "Main St"}])
        return mock_llm

    @pytest.fixture
    def repo_with_llm(self, mock_db_handler, mock_llm_handler):
        """Create AddressResolutionRepository with LLM handler."""
        return AddressResolutionRepository(mock_db_handler, mock_llm_handler)

    @pytest.fixture
    def repo_without_llm(self, mock_db_handler):
        """Create AddressResolutionRepository without LLM handler."""
        return AddressResolutionRepository(mock_db_handler)

    # =====================================================================
    # MAIN ORCHESTRATION METHOD TESTS
    # =====================================================================

    def test_initialization(self, mock_db_handler):
        """Test AddressResolutionRepository initialization."""
        repo = AddressResolutionRepository(mock_db_handler)
        assert repo.db is mock_db_handler
        assert repo.llm is None

    def test_initialization_with_llm(self, mock_db_handler, mock_llm_handler):
        """Test initialization with LLM handler."""
        repo = AddressResolutionRepository(mock_db_handler, mock_llm_handler)
        assert repo.llm is mock_llm_handler

    # =====================================================================
    # CACHE HIT TESTS
    # =====================================================================

    def test_process_event_address_cache_hit(self, repo_without_llm, mock_db_handler):
        """Test successful cache hit in process_event_address."""
        mock_db_handler.lookup_raw_location.return_value = 42
        mock_db_handler.get_full_address_from_id.return_value = "123 Main St, Vancouver"

        event = {
            "location": "Main Street, Vancouver",
            "event_name": "Dance Night",
            "source": "instagram"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 42
        assert result["location"] == "123 Main St, Vancouver"
        mock_db_handler.lookup_raw_location.assert_called_once_with("Main Street, Vancouver")
        # Quick lookup should not be called
        mock_db_handler.quick_address_lookup.assert_not_called()

    # =====================================================================
    # QUICK LOOKUP TESTS
    # =====================================================================

    def test_process_event_address_quick_lookup(self, repo_without_llm, mock_db_handler):
        """Test quick lookup when cache misses."""
        mock_db_handler.lookup_raw_location.return_value = None
        mock_db_handler.quick_address_lookup.return_value = 15
        mock_db_handler.get_full_address_from_id.return_value = "456 Elm St, Victoria"

        event = {
            "location": "Elm Street, Victoria",
            "event_name": "Jazz Night",
            "source": "facebook"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 15
        assert result["location"] == "456 Elm St, Victoria"
        mock_db_handler.cache_raw_location.assert_called_once_with("Elm Street, Victoria", 15)

    # =====================================================================
    # LLM RESOLUTION TESTS
    # =====================================================================

    def test_process_event_address_llm_resolution(self, repo_with_llm, mock_db_handler, mock_llm_handler):
        """Test LLM resolution when cache and quick lookup fail."""
        mock_db_handler.lookup_raw_location.return_value = None
        mock_db_handler.quick_address_lookup.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 99
        mock_db_handler.get_full_address_from_id.return_value = "789 Oak Ave, Seattle"

        event = {
            "location": "Oak Avenue, Seattle",
            "event_name": "Tango Event",
            "source": "website",
            "url": "https://example.com/event"
        }

        result = repo_with_llm.process_event_address(event)

        assert result["address_id"] == 99
        assert result["location"] == "789 Oak Ave, Seattle"
        mock_llm_handler.generate_prompt.assert_called_once()
        mock_llm_handler.query_llm.assert_called_once()
        mock_db_handler.cache_raw_location.assert_called_once_with("Oak Avenue, Seattle", 99)

    def test_process_event_address_llm_parse_failure(self, repo_with_llm, mock_db_handler, mock_llm_handler):
        """Test fallback when LLM parsing fails."""
        mock_db_handler.lookup_raw_location.return_value = None
        mock_db_handler.quick_address_lookup.return_value = None
        mock_llm_handler.extract_and_parse_json.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 50

        event = {
            "location": "Unclear Location Text",
            "event_name": "Event",
            "source": "source"
        }

        result = repo_with_llm.process_event_address(event)

        # Should create fallback address
        assert result["address_id"] == 50

    # =====================================================================
    # MISSING LOCATION TESTS
    # =====================================================================

    def test_process_event_address_missing_location(self, repo_without_llm, mock_db_handler):
        """Test handling of missing/None location."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 25

        event = {
            "location": None,
            "event_name": "Dance Class",
            "source": "eventbrite"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 25
        # Should attempt building name extraction
        mock_db_handler._get_building_name_dictionary.assert_called()

    def test_process_event_address_empty_location(self, repo_without_llm, mock_db_handler):
        """Test handling of empty location string."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 25

        event = {
            "location": "",
            "event_name": "Workshop",
            "source": "meetup"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 25

    def test_process_event_address_unknown_location(self, repo_without_llm, mock_db_handler):
        """Test handling of 'Unknown' location."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 25

        event = {
            "location": "Unknown",
            "event_name": "Performance",
            "source": "other"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 25

    def test_process_event_address_nan_location(self, repo_without_llm, mock_db_handler):
        """Test handling of NaN location."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 25

        event = {
            "location": float('nan'),
            "event_name": "Concert",
            "source": "ticketmaster"
        }

        result = repo_without_llm.process_event_address(event)

        assert result["address_id"] == 25

    # =====================================================================
    # BUILDING NAME EXTRACTION TESTS
    # =====================================================================

    def test_extract_address_from_event_details_exact_match(self, repo_without_llm, mock_db_handler):
        """Test exact match of building name in event details."""
        building_dict = {"Duke Saloon": 10, "The Ritz": 20}
        mock_db_handler._get_building_name_dictionary.return_value = building_dict

        event = {
            "event_name": "Dance at Duke Saloon",
            "description": "Evening dance"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        assert result == 10

    def test_extract_address_from_event_details_fuzzy_match(self, repo_without_llm, mock_db_handler):
        """Test fuzzy matching of building name."""
        building_dict = {"Grand Hotel": 30, "Small Cafe": 5}
        mock_db_handler._get_building_name_dictionary.return_value = building_dict

        event = {
            "event_name": "Evening at the Grand Hotel downtown",
            "description": "Special event"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        assert result == 30

    def test_extract_address_from_event_details_no_match(self, repo_without_llm, mock_db_handler):
        """Test when building name doesn't match."""
        building_dict = {"Duke Saloon": 10}
        mock_db_handler._get_building_name_dictionary.return_value = building_dict

        event = {
            "event_name": "Dance Night",
            "description": "Somewhere else"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        assert result is None

    def test_extract_address_from_event_details_empty_dict(self, repo_without_llm, mock_db_handler):
        """Test with empty building dictionary."""
        mock_db_handler._get_building_name_dictionary.return_value = {}

        event = {
            "event_name": "Dance Event",
            "description": "Details"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        assert result is None

    def test_extract_address_from_event_details_missing_event_name(self, repo_without_llm, mock_db_handler):
        """Test with missing event_name."""
        building_dict = {"Duke Saloon": 10}
        mock_db_handler._get_building_name_dictionary.return_value = building_dict

        event = {
            "description": "At Duke Saloon tonight"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        assert result == 10

    # =====================================================================
    # DEDUPLICATION CHECK TESTS
    # =====================================================================

    def test_resolve_missing_location_via_dedup(self, repo_without_llm, mock_db_handler):
        """Test deduplication check when building extraction fails."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = 55
        mock_db_handler.get_full_address_from_id.return_value = "999 Hall St, Portland"

        event = {
            "location": None,
            "event_name": "Unknown Event",
            "source": "my_dance_venue"
        }

        result = repo_without_llm._resolve_missing_location(event)

        assert result["address_id"] == 55
        mock_db_handler.find_address_by_building_name.assert_called_once_with(
            "my_dance_venue", threshold=75
        )

    # =====================================================================
    # LOCATION NORMALIZATION TESTS
    # =====================================================================

    def test_process_event_address_location_normalization(self, repo_without_llm, mock_db_handler):
        """Test that location is normalized (whitespace trimmed)."""
        mock_db_handler.lookup_raw_location.return_value = None
        mock_db_handler.quick_address_lookup.return_value = 77
        mock_db_handler.get_full_address_from_id.return_value = "Address"

        event = {
            "location": "   Main Street   ",
            "event_name": "Event",
            "source": "source"
        }

        repo_without_llm.process_event_address(event)

        # Verify that trimmed location was used
        mock_db_handler.lookup_raw_location.assert_called_once_with("Main Street")

    # =====================================================================
    # FALLBACK ADDRESS CREATION TESTS
    # =====================================================================

    def test_create_minimal_address_success(self, repo_without_llm, mock_db_handler):
        """Test successful creation of minimal address."""
        mock_db_handler.resolve_or_insert_address.return_value = 88
        mock_db_handler.get_full_address_from_id.return_value = "Fallback Address"

        event = {
            "location": None,
            "event_name": "Test Event",
            "source": "test_source"
        }

        result = repo_without_llm._create_minimal_address(event)

        assert result["address_id"] == 88
        assert result["location"] == "Fallback Address"
        mock_db_handler.resolve_or_insert_address.assert_called_once()

    def test_create_minimal_address_failure(self, repo_without_llm, mock_db_handler):
        """Test fallback when minimal address creation fails."""
        mock_db_handler.resolve_or_insert_address.return_value = None

        event = {
            "location": None,
            "event_name": "Test Event",
            "source": "test_source"
        }

        result = repo_without_llm._create_minimal_address(event)

        assert result["address_id"] == 0
        assert "Location unavailable" in result["location"]

    def test_create_fallback_address_success(self, repo_without_llm, mock_db_handler):
        """Test successful creation of fallback address."""
        mock_db_handler.resolve_or_insert_address.return_value = 66
        mock_db_handler.get_full_address_from_id.return_value = "Resolved Address"

        event = {
            "event_name": "Event",
            "source": "source"
        }

        result = repo_without_llm._create_fallback_address(event, "Unresolvable Location")

        assert result["address_id"] == 66
        mock_db_handler.resolve_or_insert_address.assert_called_once()

    def test_create_fallback_address_ultimate_failure(self, repo_without_llm, mock_db_handler):
        """Test ultimate fallback when all address creation fails."""
        mock_db_handler.resolve_or_insert_address.return_value = None

        event = {
            "event_name": "Event",
            "source": "source"
        }

        result = repo_without_llm._create_fallback_address(event, "Location")

        assert result["address_id"] == 0
        assert "Location unavailable" in result["location"]

    # =====================================================================
    # FINALIZATION TESTS
    # =====================================================================

    def test_finalize_event_with_address(self, repo_without_llm, mock_db_handler):
        """Test event finalization with address."""
        mock_db_handler.get_full_address_from_id.return_value = "456 Main St"

        event = {"event_name": "Event"}

        result = repo_without_llm._finalize_event_with_address(event, 42, "Main St", "cache")

        assert result["address_id"] == 42
        assert result["location"] == "456 Main St"

    def test_finalize_event_with_address_no_full_address(self, repo_without_llm, mock_db_handler):
        """Test finalization when full_address lookup returns None."""
        mock_db_handler.get_full_address_from_id.return_value = None

        event = {"event_name": "Event"}

        result = repo_without_llm._finalize_event_with_address(event, 42, "Main St", "cache")

        assert result["address_id"] == 42
        # Location should not be updated if get_full_address_from_id returns None
        assert "location" not in result or result.get("location") is None

    # =====================================================================
    # ERROR HANDLING TESTS
    # =====================================================================

    def test_process_event_address_unexpected_error(self, repo_without_llm, mock_db_handler):
        """Test exception handling in process_event_address."""
        mock_db_handler.lookup_raw_location.side_effect = Exception("DB Error")

        event = {
            "location": "Location",
            "event_name": "Event",
            "source": "test_source"
        }

        result = repo_without_llm.process_event_address(event)

        # Should return event with fallback values
        assert result["address_id"] == 0
        # Fallback uses source as the location value
        assert result["location"] == "test_source"

    def test_extract_address_from_event_details_error(self, repo_without_llm, mock_db_handler):
        """Test error handling in building extraction."""
        mock_db_handler._get_building_name_dictionary.side_effect = Exception("Error")

        event = {
            "event_name": "Event",
            "description": "Details"
        }

        result = repo_without_llm._extract_address_from_event_details(event)

        # Should return None instead of raising
        assert result is None

    def test_resolve_via_llm_error_handling(self, repo_with_llm, mock_db_handler, mock_llm_handler):
        """Test error handling in LLM resolution."""
        mock_llm_handler.generate_prompt.side_effect = Exception("LLM Error")

        event = {
            "location": "Location",
            "event_name": "Event",
            "source": "source"
        }

        result = repo_with_llm._resolve_via_llm(event, "Location")

        # Should return None instead of raising
        assert result is None

    # =====================================================================
    # EDGE CASE TESTS
    # =====================================================================

    def test_process_event_address_location_too_short(self, repo_without_llm, mock_db_handler):
        """Test with location string that's too short."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 12

        event = {
            "location": "ABC",  # Less than 5 characters
            "event_name": "Event",
            "source": "source"
        }

        result = repo_without_llm.process_event_address(event)

        # Should treat as missing location
        assert result["address_id"] == 12

    def test_process_event_address_whitespace_only(self, repo_without_llm, mock_db_handler):
        """Test with whitespace-only location."""
        mock_db_handler._get_building_name_dictionary.return_value = {}
        mock_db_handler.find_address_by_building_name.return_value = None
        mock_db_handler.resolve_or_insert_address.return_value = 13

        event = {
            "location": "     ",
            "event_name": "Event",
            "source": "source"
        }

        result = repo_without_llm.process_event_address(event)

        # Should treat as missing location
        assert result["address_id"] == 13
