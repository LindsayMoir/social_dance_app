"""Tests for URLRepository utility."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
from src.repositories.url_repository import URLRepository


class TestURLRepository:
    """Test suite for URLRepository class."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock DatabaseHandler for testing."""
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_db.config = {
            'constants': {'black_list_domains': 'fake_path.csv'},
            'clean_up': {'old_events': '30'}
        }
        mock_db.execute_query = Mock(return_value=None)
        return mock_db

    @pytest.fixture
    def url_repo(self, mock_db_handler):
        """Create a URLRepository instance with mock database."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'Domain': ['spam.com', 'blocked.net']})
            return URLRepository(mock_db_handler)

    def test_initialization(self, mock_db_handler):
        """Test URLRepository initialization."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'Domain': ['spam.com', 'blocked.net']})
            repo = URLRepository(mock_db_handler)
            assert repo.db is mock_db_handler
            assert len(repo.blacklisted_domains) == 2

    def test_load_blacklist_success(self, mock_db_handler):
        """Test successful blacklist loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'Domain': ['spam.com', 'BLOCKED.NET', '  ads.org  ']})
            repo = URLRepository(mock_db_handler)
            assert 'spam.com' in repo.blacklisted_domains
            assert 'blocked.net' in repo.blacklisted_domains
            assert 'ads.org' in repo.blacklisted_domains

    def test_load_blacklist_failure(self, mock_db_handler):
        """Test blacklist loading with file not found."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = FileNotFoundError("File not found")
            repo = URLRepository(mock_db_handler)
            assert len(repo.blacklisted_domains) == 0

    def test_is_blacklisted_true(self, url_repo):
        """Test is_blacklisted returns True for blacklisted domain."""
        assert url_repo.is_blacklisted("https://spam.com/page")
        assert url_repo.is_blacklisted("http://www.blocked.net")

    def test_is_blacklisted_false(self, url_repo):
        """Test is_blacklisted returns False for non-blacklisted domain."""
        assert not url_repo.is_blacklisted("https://google.com/page")
        assert not url_repo.is_blacklisted("http://www.linkedin.com")

    def test_is_blacklisted_case_insensitive(self, url_repo):
        """Test is_blacklisted is case insensitive."""
        assert url_repo.is_blacklisted("https://SPAM.COM/page")
        assert url_repo.is_blacklisted("https://Blocked.Net")

    def test_is_blacklisted_empty_url(self, url_repo):
        """Test is_blacklisted with empty URL."""
        assert not url_repo.is_blacklisted("")
        assert not url_repo.is_blacklisted(None)

    def test_write_url_to_db_success(self, url_repo):
        """Test successful URL write to database."""
        url_row = (
            "https://example.com",
            "https://parent.com",
            "web_crawler",
            ["dance", "music"],
            True,
            1,
            datetime.now().isoformat()
        )

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = url_repo.write_url_to_db(url_row)
            assert result is True
            mock_to_sql.assert_called_once()

    def test_write_url_to_db_keyword_normalization(self, url_repo):
        """Test keyword normalization in URL write."""
        url_row = (
            "https://example.com",
            "https://parent.com",
            "web_crawler",
            ['dance', 'music', 'swing'],
            True,
            1,
            datetime.now().isoformat()
        )

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = url_repo.write_url_to_db(url_row)
            assert result is True

    def test_write_url_to_db_failure(self, url_repo):
        """Test URL write failure handling."""
        url_row = (
            "https://example.com",
            "https://parent.com",
            "web_crawler",
            "dance",
            True,
            1,
            datetime.now().isoformat()
        )

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            mock_to_sql.side_effect = Exception("Database error")
            result = url_repo.write_url_to_db(url_row)
            assert result is False

    def test_stale_date_no_events(self, url_repo, mock_db_handler):
        """Test stale_date with no events for URL."""
        mock_db_handler.execute_query.return_value = None

        assert url_repo.stale_date("https://example.com") is True

    def test_stale_date_recent_events(self, url_repo, mock_db_handler):
        """Test stale_date with recent events."""
        recent_date = datetime.now().date()
        mock_db_handler.execute_query.return_value = [(recent_date,)]

        assert url_repo.stale_date("https://example.com") is False

    def test_stale_date_old_events(self, url_repo, mock_db_handler):
        """Test stale_date with old events."""
        old_date = (datetime.now() - timedelta(days=45)).date()
        mock_db_handler.execute_query.return_value = [(old_date,)]

        assert url_repo.stale_date("https://example.com") is True

    def test_stale_date_null_date(self, url_repo, mock_db_handler):
        """Test stale_date with NULL start_date."""
        mock_db_handler.execute_query.return_value = [(None,)]

        assert url_repo.stale_date("https://example.com") is True

    def test_stale_date_exception_handling(self, url_repo, mock_db_handler):
        """Test stale_date exception handling."""
        mock_db_handler.execute_query.side_effect = Exception("Database error")

        assert url_repo.stale_date("https://example.com") is True

    def test_normalize_url_instagram_cdn(self, url_repo):
        """Test URL normalization for Instagram CDN URLs."""
        url = "https://scontent.cdninstagram.com/image?_nc_gid=123&_nc_ohc=456&oh=789"
        normalized = url_repo.normalize_url(url)

        assert "_nc_gid" not in normalized
        assert "_nc_ohc" not in normalized
        assert "oh" not in normalized

    def test_normalize_url_facebook_cdn(self, url_repo):
        """Test URL normalization for Facebook CDN URLs."""
        url = "https://scontent-xyz.fbcdn.net/v/image?_nc_gid=123&oe=456"
        normalized = url_repo.normalize_url(url)

        assert "_nc_gid" not in normalized
        assert "oe" not in normalized

    def test_normalize_url_non_cdn(self, url_repo):
        """Test URL normalization for non-CDN URLs."""
        url = "https://example.com/page?param1=value1&param2=value2"
        normalized = url_repo.normalize_url(url)

        # Non-CDN URLs should remain unchanged
        assert normalized == url

    def test_normalize_url_empty(self, url_repo):
        """Test URL normalization with empty URL."""
        assert url_repo.normalize_url("") == ""
        assert url_repo.normalize_url(None) is None

    def test_should_process_url_blacklisted(self, url_repo):
        """Test should_process_url for blacklisted URL."""
        assert url_repo.should_process_url("https://spam.com/page") is False

    def test_should_process_url_non_blacklisted(self, url_repo, mock_db_handler):
        """Test should_process_url for non-blacklisted URL."""
        mock_db_handler.execute_query.return_value = None  # No events = stale
        assert url_repo.should_process_url("https://example.com") is True

    def test_should_process_url_empty(self, url_repo):
        """Test should_process_url with empty URL."""
        assert url_repo.should_process_url("") is False
        assert url_repo.should_process_url(None) is False
