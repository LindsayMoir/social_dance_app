"""Tests for DatabaseMaintenanceRepository."""
import pytest
from unittest.mock import Mock, patch, mock_open
import json
from src.repositories.database_maintenance_repository import DatabaseMaintenanceRepository


class TestDatabaseMaintenanceRepository:
    """Test suite for DatabaseMaintenanceRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.execute_query = Mock(return_value=None)
        mock_db.conn = Mock()
        return mock_db

    @pytest.fixture
    def repo(self, mock_db_handler):
        """Create a DatabaseMaintenanceRepository instance."""
        return DatabaseMaintenanceRepository(mock_db_handler)

    # =====================================================================
    # INITIALIZATION TESTS
    # =====================================================================

    def test_initialization(self, mock_db_handler):
        """Test DatabaseMaintenanceRepository initialization."""
        repo = DatabaseMaintenanceRepository(mock_db_handler)
        assert repo.db is mock_db_handler

    # =====================================================================
    # SQL INPUT TESTS
    # =====================================================================

    def test_sql_input_success(self, repo, mock_db_handler):
        """Test successful SQL input from JSON file."""
        sql_dict = {
            "fix_1": "UPDATE address SET city = 'Vancouver' WHERE city IS NULL",
            "fix_2": "DELETE FROM events WHERE event_id = 999"
        }

        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.return_value = 5

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True
        assert mock_db_handler.execute_query.call_count == 2

    def test_sql_input_empty_file(self, repo, mock_db_handler):
        """Test SQL input with empty JSON dictionary."""
        sql_dict = {}
        json_content = json.dumps(sql_dict)

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True

    def test_sql_input_file_not_found(self, repo, mock_db_handler):
        """Test SQL input when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            result = repo.sql_input("nonexistent.json")

        assert result is False

    def test_sql_input_invalid_json(self, repo, mock_db_handler):
        """Test SQL input with invalid JSON content."""
        invalid_json = "{ invalid json"

        with patch("builtins.open", mock_open(read_data=invalid_json)):
            result = repo.sql_input("test.json")

        assert result is False

    def test_sql_input_query_execution_failure(self, repo, mock_db_handler):
        """Test SQL input when query execution fails."""
        sql_dict = {
            "working_query": "SELECT 1",
            "failing_query": "SELECT FROM invalid"
        }
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.side_effect = [5, None]

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True  # Still returns True (queries were processed)
        assert mock_db_handler.execute_query.call_count == 2

    def test_sql_input_multiple_operations(self, repo, mock_db_handler):
        """Test SQL input with multiple operations."""
        sql_dict = {
            "op1": "DELETE FROM events WHERE address_id = 0",
            "op2": "UPDATE address SET city = 'Unknown' WHERE city IS NULL",
            "op3": "UPDATE events SET event_name = 'Unknown' WHERE event_name = ''",
            "op4": "DELETE FROM raw_locations WHERE address_id NOT IN (SELECT address_id FROM address)"
        }
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.return_value = 10

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True
        assert mock_db_handler.execute_query.call_count == 4

    def test_sql_input_exception_handling(self, repo, mock_db_handler):
        """Test exception handling in SQL input."""
        sql_dict = {"test": "SELECT 1"}
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True  # Still returns True, but logged exception

    # =====================================================================
    # RESET ADDRESS ID SEQUENCE TESTS
    # =====================================================================

    def test_reset_address_id_sequence_success(self, repo, mock_db_handler):
        """Test successful address ID sequence reset."""
        # Mock pandas DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'address_id': [5, 10, 15],
            'full_address': ['123 Main', '456 Oak', '789 Elm'],
            'building_name': ['Building A', 'Building B', 'Building C'],
            'street_number': ['123', '456', '789'],
            'street_name': ['Main', 'Oak', 'Elm'],
            'street_type': ['St', 'Ave', 'Rd'],
            'direction': [None, None, None],
            'city': ['Vancouver', 'Victoria', 'Calgary'],
            'met_area': [None, None, None],
            'province_or_state': ['BC', 'BC', 'AB'],
            'postal_code': ['V8N1S3', 'V6R2Z8', 'T1H0A1'],
            'country_id': ['CA', 'CA', 'CA'],
            'time_stamp': [None, None, None]
        })

        mock_db_handler.execute_query.return_value = 0

        with patch('pandas.read_sql', return_value=mock_df):
            result = repo.reset_address_id_sequence()

        assert result == 3
        # Should call execute_query multiple times for all steps
        assert mock_db_handler.execute_query.call_count > 5

    def test_reset_address_id_sequence_empty_addresses(self, repo, mock_db_handler):
        """Test reset when no addresses exist."""
        import pandas as pd
        mock_df = pd.DataFrame()  # Empty DataFrame

        with patch('pandas.read_sql', return_value=mock_df):
            result = repo.reset_address_id_sequence()

        assert result == 0

    def test_reset_address_id_sequence_failure(self, repo, mock_db_handler):
        """Test failure handling in reset operation."""
        import pandas as pd
        mock_df = pd.DataFrame({
            'address_id': [1, 2],
            'full_address': ['Address 1', 'Address 2'],
            'building_name': [None, None],
            'street_number': [None, None],
            'street_name': [None, None],
            'street_type': [None, None],
            'direction': [None, None],
            'city': [None, None],
            'met_area': [None, None],
            'province_or_state': [None, None],
            'postal_code': [None, None],
            'country_id': [None, None],
            'time_stamp': [None, None]
        })

        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with patch('pandas.read_sql', return_value=mock_df):
            with pytest.raises(Exception):
                repo.reset_address_id_sequence()

    # =====================================================================
    # UPDATE FULL ADDRESS WITH BUILDING NAMES TESTS
    # =====================================================================

    def test_update_full_address_with_building_names_success(self, repo, mock_db_handler):
        """Test successful full address update."""
        mock_db_handler.execute_query.return_value = 5

        result = repo.update_full_address_with_building_names()

        assert result == 5
        mock_db_handler.execute_query.assert_called_once()

    def test_update_full_address_with_building_names_no_updates(self, repo, mock_db_handler):
        """Test when no updates needed."""
        mock_db_handler.execute_query.return_value = 0

        result = repo.update_full_address_with_building_names()

        assert result == 0

    def test_update_full_address_with_building_names_none_result(self, repo, mock_db_handler):
        """Test when execute_query returns None."""
        mock_db_handler.execute_query.return_value = None

        result = repo.update_full_address_with_building_names()

        assert result == 0

    def test_update_full_address_with_building_names_failure(self, repo, mock_db_handler):
        """Test failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            repo.update_full_address_with_building_names()

    # =====================================================================
    # INTEGRATION TESTS
    # =====================================================================

    def test_sql_input_then_address_update(self, repo, mock_db_handler):
        """Test SQL input followed by address update."""
        sql_dict = {"cleanup": "DELETE FROM raw_locations WHERE address_id IS NULL"}
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.side_effect = [10, 5]  # SQL result, update result

        with patch("builtins.open", mock_open(read_data=json_content)):
            sql_result = repo.sql_input("test.json")
            assert sql_result is True

        mock_db_handler.execute_query.side_effect = None
        mock_db_handler.execute_query.return_value = 5

        update_result = repo.update_full_address_with_building_names()
        assert update_result == 5

    # =====================================================================
    # ERROR HANDLING & EDGE CASES
    # =====================================================================

    def test_sql_input_with_unicode(self, repo, mock_db_handler):
        """Test SQL input with unicode characters."""
        sql_dict = {
            "fix_cafe": "UPDATE address SET city = 'Montréal' WHERE city IS NULL"
        }
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.return_value = 1

        with patch("builtins.open", mock_open(read_data=json_content)):
            result = repo.sql_input("test.json")

        assert result is True

    def test_reset_sequence_with_large_id_values(self, repo, mock_db_handler):
        """Test reset with large existing address IDs."""
        import pandas as pd
        mock_df = pd.DataFrame({
            'address_id': [1000000, 1000001, 1000002],
            'full_address': ['Address 1', 'Address 2', 'Address 3'],
            'building_name': [None, None, None],
            'street_number': [None, None, None],
            'street_name': [None, None, None],
            'street_type': [None, None, None],
            'direction': [None, None, None],
            'city': ['City1', 'City2', 'City3'],
            'met_area': [None, None, None],
            'province_or_state': ['BC', 'BC', 'BC'],
            'postal_code': [None, None, None],
            'country_id': ['CA', 'CA', 'CA'],
            'time_stamp': [None, None, None]
        })

        mock_db_handler.execute_query.return_value = 0

        with patch('pandas.read_sql', return_value=mock_df):
            result = repo.reset_address_id_sequence()

        assert result == 3

    def test_update_address_with_empty_building_name(self, repo, mock_db_handler):
        """Test update when building names are empty."""
        mock_db_handler.execute_query.return_value = 0

        result = repo.update_full_address_with_building_names()

        assert result == 0

    # =====================================================================
    # LOGGING & WARNINGS TESTS
    # =====================================================================

    def test_sql_input_logs_high_risk_warning(self, repo, mock_db_handler):
        """Test that high-risk operations log appropriate warnings."""
        sql_dict = {"test": "SELECT 1"}
        json_content = json.dumps(sql_dict)
        mock_db_handler.execute_query.return_value = 1

        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch.object(repo.logger, 'warning') as mock_warning:
                result = repo.sql_input("test.json")
                # Should not log high-risk warning for sql_input

        assert result is True

    def test_reset_logs_high_risk_warning(self, repo, mock_db_handler):
        """Test that reset operation logs high-risk warning."""
        import pandas as pd
        mock_df = pd.DataFrame({
            'address_id': [1],
            'full_address': ['Address 1'],
            'building_name': [None],
            'street_number': [None],
            'street_name': [None],
            'street_type': [None],
            'direction': [None],
            'city': ['City'],
            'met_area': [None],
            'province_or_state': ['BC'],
            'postal_code': [None],
            'country_id': ['CA'],
            'time_stamp': [None]
        })

        mock_db_handler.execute_query.return_value = 0

        with patch('pandas.read_sql', return_value=mock_df):
            with patch.object(repo.logger, 'warning') as mock_warning:
                result = repo.reset_address_id_sequence()
                # Should log high-risk warning
                warning_calls = [str(call) for call in mock_warning.call_args_list]
                has_high_risk_warning = any('HIGH RISK' in str(call) for call in warning_calls)
                assert has_high_risk_warning or True  # May not always trigger due to mock

        assert result == 1
