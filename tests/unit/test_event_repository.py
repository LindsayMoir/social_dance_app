"""Tests for EventRepository utility."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta, date, time
import pandas as pd
import numpy as np
from src.repositories.event_repository import EventRepository


class TestEventRepository:
    """Test suite for EventRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_db.config = {
            'clean_up': {'old_events': '30'}
        }
        mock_db.execute_query = Mock(return_value=None)
        mock_db.clean_up_address_basic = Mock(side_effect=lambda df: df)
        mock_db.normalize_nulls = Mock(side_effect=lambda x: x)
        mock_db.process_event_address = Mock(side_effect=lambda x: x)
        mock_db.write_url_to_db = Mock(return_value=True)
        return mock_db

    @pytest.fixture
    def event_repo(self, mock_db_handler):
        """Create an EventRepository instance with mock database."""
        return EventRepository(mock_db_handler)

    def test_initialization(self, mock_db_handler):
        """Test EventRepository initialization."""
        repo = EventRepository(mock_db_handler)
        assert repo.db is mock_db_handler

    def test_write_events_to_db_success(self, event_repo):
        """Test successful event write to database."""
        df = pd.DataFrame({
            'event_name': ['Dance Night'],
            'start_date': [date.today()],
            'start_time': [time(19, 0)],
            'end_date': [date.today()],
            'end_time': [time(21, 0)],
            'location': ['Community Hall'],
            'description': ['A fun dance event']
        })

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = event_repo.write_events_to_db(df, 'https://example.com', '', 'web', 'dance')
            assert result is True
            mock_to_sql.assert_called_once()

    def test_write_events_to_db_empty_after_filtering(self, event_repo):
        """Test write_events_to_db when all events are filtered out."""
        df = pd.DataFrame({
            'event_name': [None],
            'start_date': [None],
            'start_time': [None],
            'location': [None],
            'description': [None]
        })

        result = event_repo.write_events_to_db(df, 'https://example.com', '', 'web', 'dance')
        assert result is False

    def test_write_events_to_db_with_google_calendar(self, event_repo):
        """Test write_events_to_db with Google Calendar format."""
        df = pd.DataFrame({
            'URL': ['https://calendar.example.com'],
            'Name_of_the_Event': ['Event'],
            'Start_Date': [date.today()],
            'Start_Time': [time(19, 0)],
            'End_Date': [date.today()],
            'End_Time': [time(21, 0)],
            'Location': ['Hall'],
            'Description': ['Description']
        })

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = event_repo.write_events_to_db(df, 'https://calendar.example.com', '', 'web', ['dance', 'music'])
            assert result is True

    def test_write_events_to_db_failure(self, event_repo):
        """Test write_events_to_db failure handling."""
        df = pd.DataFrame({
            'event_name': ['Event'],
            'start_date': [date.today()],
            'start_time': [time(19, 0)],
            'location': ['Hall'],
            'description': ['Desc']
        })

        with patch.object(pd.DataFrame, 'to_sql', side_effect=Exception("DB Error")):
            result = event_repo.write_events_to_db(df, 'https://example.com', '', 'web', 'dance')
            assert result is False

    def test_update_event_success(self, event_repo, mock_db_handler):
        """Test successful event update."""
        mock_db_handler.execute_query.return_value = [
            {
                'event_id': 1,
                'event_name': 'Old Name',
                'start_date': date.today(),
                'start_time': time(19, 0),
                'location': 'Old Location',
                'url': 'old_url'
            }
        ]

        identifier = {'event_name': 'Old Name', 'start_date': date.today(), 'start_time': time(19, 0)}
        new_data = {'location': 'New Location', 'description': 'New Description'}

        result = event_repo.update_event(identifier, new_data, 'https://new.com')
        assert result is True
        mock_db_handler.execute_query.assert_called()

    def test_update_event_not_found(self, event_repo, mock_db_handler):
        """Test update_event when event not found."""
        mock_db_handler.execute_query.return_value = None

        identifier = {'event_name': 'Nonexistent', 'start_date': date.today(), 'start_time': time(19, 0)}
        new_data = {'location': 'New Location'}

        result = event_repo.update_event(identifier, new_data, 'https://new.com')
        assert result is False

    def test_update_event_empty_list_result(self, event_repo, mock_db_handler):
        """Test update_event when query returns empty list."""
        mock_db_handler.execute_query.return_value = []

        identifier = {'event_name': 'Event', 'start_date': date.today(), 'start_time': time(19, 0)}
        new_data = {'location': 'New'}

        result = event_repo.update_event(identifier, new_data, 'https://new.com')
        assert result is False

    def test_delete_event_success(self, event_repo):
        """Test successful event deletion."""
        result = event_repo.delete_event('https://example.com', 'Event Name', date.today())
        assert result is True

    def test_delete_event_failure(self, event_repo, mock_db_handler):
        """Test delete_event failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = event_repo.delete_event('https://example.com', 'Event Name', date.today())
        assert result is False

    def test_delete_event_with_event_id_success(self, event_repo):
        """Test successful deletion by event_id."""
        result = event_repo.delete_event_with_event_id(123)
        assert result is True

    def test_delete_event_with_event_id_failure(self, event_repo, mock_db_handler):
        """Test delete_event_with_event_id failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        result = event_repo.delete_event_with_event_id(123)
        assert result is False

    def test_delete_multiple_events_success(self, event_repo):
        """Test successful deletion of multiple events."""
        result = event_repo.delete_multiple_events([1, 2, 3])
        assert result is True

    def test_delete_multiple_events_partial_failure(self, event_repo, mock_db_handler):
        """Test delete_multiple_events with partial failures."""
        call_count = [0]

        def mock_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("DB Error")
            return None

        mock_db_handler.execute_query.side_effect = mock_execute

        result = event_repo.delete_multiple_events([1, 2, 3])
        assert result is False

    def test_delete_multiple_events_empty_list(self, event_repo):
        """Test delete_multiple_events with empty list."""
        result = event_repo.delete_multiple_events([])
        assert result is False

    def test_fetch_events_dataframe_success(self, event_repo):
        """Test successful fetch of events DataFrame."""
        test_df = pd.DataFrame({
            'event_id': [1, 2],
            'event_name': ['Event 1', 'Event 2'],
            'start_date': [date.today(), date.today()],
            'start_time': [time(19, 0), time(20, 0)]
        })

        with patch('pandas.read_sql', return_value=test_df):
            result = event_repo.fetch_events_dataframe()
            assert len(result) == 2
            assert result['event_name'].tolist() == ['Event 1', 'Event 2']

    def test_fetch_events_dataframe_sorted(self, event_repo):
        """Test that fetch_events_dataframe returns sorted DataFrame."""
        test_df = pd.DataFrame({
            'event_id': [1, 2, 3],
            'event_name': ['Event 1', 'Event 2', 'Event 3'],
            'start_date': [date.today(), date.today() - timedelta(days=1), date.today()],
            'start_time': [time(19, 0), time(20, 0), time(18, 0)]
        })

        with patch('pandas.read_sql', return_value=test_df):
            result = event_repo.fetch_events_dataframe()
            # Check that it's sorted (earlier dates first, then earlier times)
            assert result.iloc[0]['start_date'] <= result.iloc[1]['start_date']

    def test_fetch_events_dataframe_failure(self, event_repo, mock_db_handler):
        """Test fetch_events_dataframe failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with patch('pandas.read_sql', side_effect=Exception("DB Error")):
            result = event_repo.fetch_events_dataframe()
            assert result.empty

    def test_rename_google_calendar_columns(self, event_repo):
        """Test Google Calendar column renaming."""
        df = pd.DataFrame({
            'URL': ['https://example.com'],
            'Name_of_the_Event': ['Event Name'],
            'Start_Date': [date.today()],
            'Day_of_Week': ['Friday']
        })

        result = event_repo._rename_google_calendar_columns(df)
        assert 'url' in result.columns
        assert 'event_name' in result.columns
        assert 'start_date' in result.columns
        assert 'day_of_week' in result.columns

    def test_convert_datetime_fields(self, event_repo):
        """Test datetime field conversion."""
        df = pd.DataFrame({
            'start_date': ['2024-01-15', '2024-02-20'],
            'end_date': ['2024-01-20', '2024-02-25'],
            'start_time': ['19:00', '20:30'],
            'end_time': ['21:00', '22:45']
        })

        event_repo._convert_datetime_fields(df)

        assert df['start_date'].dtype == 'object'  # date object
        assert df['start_time'].dtype == 'object'  # time object

    def test_convert_datetime_fields_with_invalid(self, event_repo):
        """Test datetime field conversion with invalid values."""
        df = pd.DataFrame({
            'start_date': ['2024-01-15', 'invalid_date'],
            'end_date': ['2024-01-20', 'invalid_date'],
            'start_time': ['19:00', 'invalid_time'],
            'end_time': ['21:00', 'invalid_time']
        })

        event_repo._convert_datetime_fields(df)

        # Should handle errors gracefully
        assert len(df) == 2

    def test_clean_day_of_week_field_compound_values(self, event_repo):
        """Test cleaning compound day_of_week values."""
        df = pd.DataFrame({
            'day_of_week': ['Friday, Saturday', 'Monday', 'Daily']
        })

        result = event_repo._clean_day_of_week_field(df)

        assert result['day_of_week'].iloc[0] == 'Friday'
        assert result['day_of_week'].iloc[1] == 'Monday'
        assert result['day_of_week'].iloc[2] == ''

    def test_clean_day_of_week_field_normalization(self, event_repo):
        """Test day_of_week normalization."""
        df = pd.DataFrame({
            'day_of_week': ['FRIDAY', 'monday', 'tUeSdAy']
        })

        result = event_repo._clean_day_of_week_field(df)

        assert result['day_of_week'].iloc[0] == 'Friday'
        assert result['day_of_week'].iloc[1] == 'Monday'
        assert result['day_of_week'].iloc[2] == 'Tuesday'

    def test_clean_day_of_week_field_missing_column(self, event_repo):
        """Test cleaning when day_of_week column is missing."""
        df = pd.DataFrame({
            'event_name': ['Event'],
            'location': ['Hall']
        })

        result = event_repo._clean_day_of_week_field(df)

        # Should return unchanged
        assert len(result) == 1

    def test_filter_events_removes_old_events(self, event_repo):
        """Test that _filter_events removes old events."""
        old_date = date.today() - timedelta(days=40)
        recent_date = date.today() - timedelta(days=10)

        df = pd.DataFrame({
            'event_name': ['Old Event', 'Recent Event'],
            'start_date': [date.today() - timedelta(days=40), date.today() - timedelta(days=10)],
            'end_date': [old_date, recent_date],
            'start_time': [time(19, 0), time(19, 0)],
            'end_time': [time(21, 0), time(21, 0)],
            'location': ['Hall', 'Hall'],
            'description': ['Old', 'Recent']
        })

        result = event_repo._filter_events(df)

        # Old event should be filtered out
        assert len(result) == 1
        assert result['event_name'].iloc[0] == 'Recent Event'

    def test_filter_events_removes_incomplete_events(self, event_repo):
        """Test that _filter_events removes events with all important columns empty."""
        df = pd.DataFrame({
            'event_name': ['Complete Event', 'Incomplete Event'],
            'start_date': [date.today(), None],
            'end_date': [date.today(), None],
            'start_time': [time(19, 0), None],
            'end_time': [time(21, 0), None],
            'location': ['Hall', None],
            'description': ['Desc', None]
        })

        result = event_repo._filter_events(df)

        # Incomplete event should be filtered out
        assert len(result) == 1
        assert result['event_name'].iloc[0] == 'Complete Event'

    def test_filter_events_keeps_partial_events(self, event_repo):
        """Test that _filter_events keeps events with some data."""
        df = pd.DataFrame({
            'event_name': ['Partial Event'],
            'start_date': [date.today()],
            'end_date': [None],
            'start_time': [time(19, 0)],
            'end_time': [None],
            'location': [None],
            'description': ['Has description']
        })

        result = event_repo._filter_events(df)

        # Event should be kept (not all important columns are empty)
        assert len(result) == 1
