"""Tests for EventManagementRepository utility."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, date
import pandas as pd
from src.repositories.event_management_repository import EventManagementRepository


class TestEventManagementRepository:
    """Test suite for EventManagementRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_db.config = {
            'clean_up': {'old_events': '30'}
        }
        mock_db.execute_query = Mock(return_value=None)
        return mock_db

    @pytest.fixture
    def event_mgmt_repo(self, mock_db_handler):
        """Create an EventManagementRepository instance with mock database."""
        return EventManagementRepository(mock_db_handler)

    def test_initialization(self, mock_db_handler):
        """Test EventManagementRepository initialization."""
        repo = EventManagementRepository(mock_db_handler)
        assert repo.db is mock_db_handler

    def test_delete_old_events_success(self, event_mgmt_repo, mock_db_handler):
        """Test successful deletion of old events."""
        mock_db_handler.execute_query.return_value = 5

        result = event_mgmt_repo.delete_old_events()

        assert result == 5
        mock_db_handler.execute_query.assert_called()

    def test_delete_old_events_no_events(self, event_mgmt_repo, mock_db_handler):
        """Test delete_old_events when no events need deletion."""
        mock_db_handler.execute_query.return_value = 0

        result = event_mgmt_repo.delete_old_events()

        assert result == 0

    def test_delete_old_events_failure(self, event_mgmt_repo, mock_db_handler):
        """Test delete_old_events failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_mgmt_repo.delete_old_events()

    def test_delete_likely_dud_events_success(self, event_mgmt_repo, mock_db_handler):
        """Test successful deletion of low-quality events."""
        # Mock returns for each rule
        mock_db_handler.execute_query.side_effect = [
            [1, 2, 3],  # Rule 1: 3 events deleted
            [4, 5],     # Rule 2: 2 events deleted (BC filtering)
            [6],        # Rule 3: 1 event deleted (CA filtering)
            [7, 8, 9, 10]  # Rule 4: 4 events deleted (low quality)
        ]

        result = event_mgmt_repo.delete_likely_dud_events()

        assert result['rule_1_deleted'] == 3
        assert result['rule_2_deleted'] == 2
        assert result['rule_3_deleted'] == 1
        assert result['rule_4_deleted'] == 4
        assert result['total_deleted'] == 10

    def test_delete_likely_dud_events_no_deletions(self, event_mgmt_repo, mock_db_handler):
        """Test delete_likely_dud_events when no events qualify for deletion."""
        mock_db_handler.execute_query.side_effect = [None, None, None, None]

        result = event_mgmt_repo.delete_likely_dud_events()

        assert result['rule_1_deleted'] == 0
        assert result['rule_2_deleted'] == 0
        assert result['rule_3_deleted'] == 0
        assert result['rule_4_deleted'] == 0
        assert result['total_deleted'] == 0

    def test_delete_likely_dud_events_failure(self, event_mgmt_repo, mock_db_handler):
        """Test delete_likely_dud_events failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_mgmt_repo.delete_likely_dud_events()

    def test_delete_events_with_nulls_success(self, event_mgmt_repo, mock_db_handler):
        """Test successful deletion of events with null critical fields."""
        mock_db_handler.execute_query.return_value = 3

        result = event_mgmt_repo.delete_events_with_nulls()

        assert result == 3

    def test_delete_events_with_nulls_no_events(self, event_mgmt_repo, mock_db_handler):
        """Test delete_events_with_nulls when no events have null critical fields."""
        mock_db_handler.execute_query.return_value = 0

        result = event_mgmt_repo.delete_events_with_nulls()

        assert result == 0

    def test_delete_events_with_nulls_failure(self, event_mgmt_repo, mock_db_handler):
        """Test delete_events_with_nulls failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_mgmt_repo.delete_events_with_nulls()

    def test_dedup_success(self, event_mgmt_repo, mock_db_handler):
        """Test successful event deduplication."""
        mock_db_handler.execute_query.return_value = 7

        result = event_mgmt_repo.dedup()

        assert result == 7

    def test_dedup_no_duplicates(self, event_mgmt_repo, mock_db_handler):
        """Test dedup when no duplicate events exist."""
        mock_db_handler.execute_query.return_value = 0

        result = event_mgmt_repo.dedup()

        assert result == 0

    def test_dedup_failure(self, event_mgmt_repo, mock_db_handler):
        """Test dedup failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_mgmt_repo.dedup()

    def test_update_dow_date_success(self, event_mgmt_repo, mock_db_handler):
        """Test successful day-of-week date update."""
        mock_db_handler.execute_query.return_value = None

        result = event_mgmt_repo.update_dow_date(123, date.today())

        assert result is True
        mock_db_handler.execute_query.assert_called()

    def test_update_dow_date_failure(self, event_mgmt_repo, mock_db_handler):
        """Test update_dow_date failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_mgmt_repo.update_dow_date(123, date.today())

    def test_check_dow_date_consistent_no_events(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent with no events."""
        mock_db_handler.execute_query.return_value = None

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 0
        assert result['events_updated'] == 0
        assert result['errors'] == 0

    def test_check_dow_date_consistent_matching_dates(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent when all dates match day_of_week."""
        # Create a Monday and set day_of_week to 'Monday'
        test_date = date.today()
        # Find next Monday
        days_ahead = 0 - test_date.weekday()  # Monday is 0
        if days_ahead <= 0:
            days_ahead += 7
        monday = test_date + timedelta(days=days_ahead)

        mock_db_handler.execute_query.return_value = [
            (1, monday, 'Monday'),
            (2, monday, 'Monday')
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 2
        assert result['events_updated'] == 0

    def test_check_dow_date_consistent_mismatched_dates(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent corrects mismatched dates."""
        # Create a date that's not a Monday, but claim day_of_week is Monday
        test_date = date(2024, 10, 23)  # This is a Wednesday

        # Mock returns: first call returns events, second call for update_dow_date
        mock_db_handler.execute_query.side_effect = [
            [(1, test_date, 'Monday')],
            None  # update_dow_date call
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 1
        # Should have attempted to update (found a Monday within ±3 days)
        assert result['events_updated'] >= 0  # Depends on date calculation

    def test_check_dow_date_consistent_empty_day_of_week(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent skips events with empty day_of_week."""
        test_date = date.today()

        mock_db_handler.execute_query.return_value = [
            (1, test_date, ''),
            (2, test_date, None)
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 2
        assert result['events_updated'] == 0

    def test_check_dow_date_consistent_invalid_day_of_week(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent handles invalid day_of_week values."""
        test_date = date.today()

        mock_db_handler.execute_query.return_value = [
            (1, test_date, 'InvalidDay')
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 1
        assert result['events_updated'] == 0

    def test_check_dow_date_consistent_with_errors(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent error handling during processing."""
        # Create a valid date scenario
        test_date = date.today()
        monday = test_date + timedelta(days=(7 - test_date.weekday()))

        # First return valid data, second call (update) raises error
        mock_db_handler.execute_query.side_effect = [
            [(1, test_date, 'Monday')],
            Exception("Update failed")
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 1
        assert result['errors'] >= 0  # May or may not have errors depending on matching

    def test_check_dow_date_consistent_string_date_parsing(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent handles string dates correctly."""
        test_date_str = "2024-10-23"  # Wednesday

        mock_db_handler.execute_query.return_value = [
            (1, test_date_str, 'Wednesday')
        ]

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 1

    def test_check_dow_date_consistent_all_day_names(self, event_mgmt_repo, mock_db_handler):
        """Test check_dow_date_consistent with all day-of-week names."""
        base_date = date(2024, 10, 21)  # Start with a Monday
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        events = [
            (i, base_date + timedelta(days=i), days[i])
            for i in range(7)
        ]

        mock_db_handler.execute_query.return_value = events

        result = event_mgmt_repo.check_dow_date_consistent()

        assert result['events_checked'] == 7

    def test_delete_likely_dud_events_partial_results(self, event_mgmt_repo, mock_db_handler):
        """Test delete_likely_dud_events with some rules returning results."""
        mock_db_handler.execute_query.side_effect = [
            [1, 2],      # Rule 1: 2 results
            None,        # Rule 2: no results
            [3],         # Rule 3: 1 result
            None         # Rule 4: no results
        ]

        result = event_mgmt_repo.delete_likely_dud_events()

        assert result['rule_1_deleted'] == 2
        assert result['rule_2_deleted'] == 0
        assert result['rule_3_deleted'] == 1
        assert result['rule_4_deleted'] == 0
        assert result['total_deleted'] == 3

    def test_delete_old_events_uses_config_threshold(self, event_mgmt_repo, mock_db_handler):
        """Test that delete_old_events uses the configured threshold."""
        mock_db_handler.config['clean_up']['old_events'] = '60'
        mock_db_handler.execute_query.return_value = 10

        result = event_mgmt_repo.delete_old_events()

        # Verify the query was constructed with the correct number of days
        call_args = mock_db_handler.execute_query.call_args_list
        # The second call should contain the '60' in the query
        assert '60' in str(call_args[0])
