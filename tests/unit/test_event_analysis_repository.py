"""Tests for EventAnalysisRepository utility."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
from src.repositories.event_analysis_repository import EventAnalysisRepository


class TestEventAnalysisRepository:
    """Test suite for EventAnalysisRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_db.config = {
            'output': {'events_urls_diff': 'output/events_urls_diff.csv'}
        }
        mock_db.execute_query = Mock(return_value=None)
        return mock_db

    @pytest.fixture
    def event_analysis_repo(self, mock_db_handler):
        """Create an EventAnalysisRepository instance with mock database."""
        return EventAnalysisRepository(mock_db_handler)

    def test_initialization(self, mock_db_handler):
        """Test EventAnalysisRepository initialization."""
        repo = EventAnalysisRepository(mock_db_handler)
        assert repo.db is mock_db_handler

    def test_sync_event_locations_success(self, event_analysis_repo, mock_db_handler):
        """Test successful synchronization of event locations."""
        mock_db_handler.execute_query.return_value = 5

        result = event_analysis_repo.sync_event_locations_with_address_table()

        assert result == 5
        mock_db_handler.execute_query.assert_called()

    def test_sync_event_locations_no_updates(self, event_analysis_repo, mock_db_handler):
        """Test sync_event_locations when all are already in sync."""
        mock_db_handler.execute_query.return_value = 0

        result = event_analysis_repo.sync_event_locations_with_address_table()

        assert result == 0

    def test_sync_event_locations_failure(self, event_analysis_repo, mock_db_handler):
        """Test sync_event_locations failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_analysis_repo.sync_event_locations_with_address_table()

    def test_clean_orphaned_references_success(self, event_analysis_repo, mock_db_handler):
        """Test successful cleanup of orphaned references."""
        mock_db_handler.execute_query.side_effect = [5, 2, 3]

        result = event_analysis_repo.clean_orphaned_references()

        assert result['raw_locations'] == 5
        assert result['events'] == 2
        assert result['events_history'] == 3
        assert result['total'] == 10

    def test_clean_orphaned_references_no_orphans(self, event_analysis_repo, mock_db_handler):
        """Test cleanup when no orphaned records exist."""
        mock_db_handler.execute_query.side_effect = [0, 0, 0]

        result = event_analysis_repo.clean_orphaned_references()

        assert result['raw_locations'] == 0
        assert result['events'] == 0
        assert result['events_history'] == 0
        assert result['total'] == 0

    def test_clean_orphaned_references_partial(self, event_analysis_repo, mock_db_handler):
        """Test cleanup with orphans in some tables only."""
        mock_db_handler.execute_query.side_effect = [3, 0, 1]

        result = event_analysis_repo.clean_orphaned_references()

        assert result['raw_locations'] == 3
        assert result['events'] == 0
        assert result['events_history'] == 1
        assert result['total'] == 4

    def test_clean_orphaned_references_failure(self, event_analysis_repo, mock_db_handler):
        """Test clean_orphaned_references failure handling."""
        mock_db_handler.execute_query.side_effect = Exception("DB Error")

        with pytest.raises(Exception):
            event_analysis_repo.clean_orphaned_references()

    def test_count_events_urls_start_success(self, event_analysis_repo, mock_db_handler):
        """Test successful baseline count of events and URLs."""
        test_df = pd.DataFrame({
            'file_name': ['test_crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_start': [100]}),
            pd.DataFrame({'urls_count_start': [50]})
        ]):
            result = event_analysis_repo.count_events_urls_start('test_crawler.py')

            assert 'file_name' in result.columns
            assert 'start_time' in result.columns
            assert 'events_count_start' in result.columns
            assert 'urls_count_start' in result.columns

    def test_count_events_urls_start_zero_counts(self, event_analysis_repo, mock_db_handler):
        """Test count_events_urls_start with empty database."""
        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_start': [0]}),
            pd.DataFrame({'urls_count_start': [0]})
        ]):
            result = event_analysis_repo.count_events_urls_start('crawler.py')

            assert result['events_count_start'].iloc[0] == 0
            assert result['urls_count_start'].iloc[0] == 0

    def test_count_events_urls_start_failure(self, event_analysis_repo, mock_db_handler):
        """Test count_events_urls_start failure handling."""
        with patch('pandas.read_sql', side_effect=Exception("DB Error")):
            with pytest.raises(Exception):
                event_analysis_repo.count_events_urls_start('crawler.py')

    def test_count_events_urls_end_success(self, event_analysis_repo, mock_db_handler):
        """Test successful end count and statistics calculation."""
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [120]}),
            pd.DataFrame({'urls_count_end': [60]})
        ]):
            result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

            assert result['new_events_in_db'].iloc[0] == 20
            assert result['new_urls_in_db'].iloc[0] == 10
            assert 'time_stamp' in result.columns
            assert 'elapsed_time' in result.columns

    def test_count_events_urls_end_no_change(self, event_analysis_repo, mock_db_handler):
        """Test end count when no new events/URLs were added."""
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [100]}),
            pd.DataFrame({'urls_count_end': [50]})
        ]):
            result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

            assert result['new_events_in_db'].iloc[0] == 0
            assert result['new_urls_in_db'].iloc[0] == 0

    def test_count_events_urls_end_decreases(self, event_analysis_repo, mock_db_handler):
        """Test end count when events/URLs decreased (from deletions)."""
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [85]}),
            pd.DataFrame({'urls_count_end': [45]})
        ]):
            result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

            assert result['new_events_in_db'].iloc[0] == -15
            assert result['new_urls_in_db'].iloc[0] == -5

    @patch('os.path.isfile', return_value=False)
    @patch('os.makedirs')
    def test_count_events_urls_end_csv_write_new_file(
        self, mock_makedirs, mock_isfile, event_analysis_repo, mock_db_handler
    ):
        """Test writing statistics to new CSV file."""
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [110]}),
            pd.DataFrame({'urls_count_end': [55]})
        ]):
            with patch('os.getenv', return_value=None):
                with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                    result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

                    # Should be called with index=False for new file
                    mock_to_csv.assert_called_once()

    def test_count_events_urls_end_no_csv_on_render(
        self, event_analysis_repo, mock_db_handler, monkeypatch
    ):
        """Test that CSV is not written on Render (ephemeral filesystem)."""
        # Patch IS_RENDER to True in the event_analysis_repository module
        from src.repositories import event_analysis_repository
        monkeypatch.setattr(event_analysis_repository, 'IS_RENDER', True)

        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [105]}),
            pd.DataFrame({'urls_count_end': [52]})
        ]):
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

                # Should not write CSV on Render
                mock_to_csv.assert_not_called()

    def test_count_events_urls_end_failure(self, event_analysis_repo, mock_db_handler):
        """Test count_events_urls_end failure handling."""
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [datetime.now()],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=Exception("DB Error")):
            with pytest.raises(Exception):
                event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

    def test_check_image_events_exist_always_false(self, event_analysis_repo):
        """Test that check_image_events_exist always returns False (corruption prevention)."""
        result1 = event_analysis_repo.check_image_events_exist("https://example.com/image1.jpg")
        result2 = event_analysis_repo.check_image_events_exist("https://example.com/image2.jpg")
        result3 = event_analysis_repo.check_image_events_exist("invalid_url")

        # All should return False to force re-scraping
        assert result1 is False
        assert result2 is False
        assert result3 is False

    def test_count_events_urls_elapsed_time_calculation(self, event_analysis_repo, mock_db_handler):
        """Test that elapsed_time is correctly calculated."""
        start_time = datetime.now() - timedelta(minutes=5)
        start_df = pd.DataFrame({
            'file_name': ['crawler.py'],
            'start_time': [start_time],
            'events_count_start': [100],
            'urls_count_start': [50]
        })

        with patch('pandas.read_sql', side_effect=[
            pd.DataFrame({'events_count_end': [105]}),
            pd.DataFrame({'urls_count_end': [52]})
        ]):
            result = event_analysis_repo.count_events_urls_end(start_df, 'crawler.py')

            elapsed = result['elapsed_time'].iloc[0]
            # Elapsed time should be approximately 5 minutes
            assert elapsed.total_seconds() > 0
            assert elapsed.total_seconds() >= 300  # At least 5 minutes
