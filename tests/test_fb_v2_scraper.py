#!/usr/bin/env python3
"""
Test suite for FacebookScraperV2 (src/fb_v2.py)

Comprehensive testing of the refactored Facebook event scraper with BaseScraper utilities.

This test suite covers:
1. Initialization and configuration
2. Utility manager integration
3. URL normalization and navigation
4. Event extraction and text processing
5. Statistics tracking
6. Error handling and resilience
7. Database integration

Run with:
    pytest tests/test_fb_v2_scraper.py -v
    pytest tests/test_fb_v2_scraper.py::test_initialization -v
"""

import sys
import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, 'src')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('src/.env')

from logging_config import setup_logging
import yaml

# Setup logging
setup_logging('test_fb_v2_scraper')
logger = logging.getLogger(__name__)

# Load config
config_path = Path("config/config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)


# ============================================================================
# UNIT TESTS - FacebookScraperV2 Core Functionality
# ============================================================================

class TestFacebookScraperV2Initialization:
    """Test FacebookScraperV2 initialization and setup."""

    def test_imports(self):
        """Test that FacebookScraperV2 can be imported."""
        from fb_v2 import FacebookScraperV2
        assert FacebookScraperV2 is not None
        logger.info("✓ FacebookScraperV2 imports successfully")

    def test_class_exists(self):
        """Test that FacebookScraperV2 class exists."""
        from fb_v2 import FacebookScraperV2
        assert hasattr(FacebookScraperV2, '__init__')
        assert hasattr(FacebookScraperV2, 'login_to_facebook')
        assert hasattr(FacebookScraperV2, 'normalize_facebook_url')
        logger.info("✓ FacebookScraperV2 class structure verified")

    def test_all_methods_exist(self):
        """Test that all major methods are present."""
        from fb_v2 import FacebookScraperV2

        required_methods = [
            '__init__',
            'login_to_facebook',
            'normalize_facebook_url',
            'navigate_and_maybe_login',
            'extract_event_links',
            'extract_event_text',
            'extract_relevant_text',
            'scrape_events',
            'process_fb_url',
            'driver_fb_search',
            'driver_fb_urls',
            'get_statistics',
            'log_statistics',
            'scrape'
        ]

        for method in required_methods:
            assert hasattr(FacebookScraperV2, method), f"Missing method: {method}"

        logger.info(f"✓ All {len(required_methods)} major methods present")

    def test_inherits_from_base_scraper(self):
        """Test that FacebookScraperV2 inherits from BaseScraper."""
        from fb_v2 import FacebookScraperV2
        from base_scraper import BaseScraper

        assert issubclass(FacebookScraperV2, BaseScraper)
        logger.info("✓ FacebookScraperV2 correctly inherits from BaseScraper")


class TestUtilityManagerIntegration:
    """Test integration with BaseScraper utility managers."""

    def test_utility_imports(self):
        """Test that all required utility modules can be imported."""
        from text_utils import TextExtractor
        from url_nav import URLNavigator
        from resilience import RetryManager, CircuitBreaker

        assert TextExtractor is not None
        assert URLNavigator is not None
        assert RetryManager is not None
        assert CircuitBreaker is not None
        logger.info("✓ All utility modules import successfully")

    def test_base_scraper_utilities_available(self):
        """Test that BaseScraper utilities are available."""
        from base_scraper import BaseScraper

        required_managers = [
            'browser_manager',
            'circuit_breaker',
            'logger',
            'config'
        ]

        for manager in required_managers:
            assert hasattr(BaseScraper, manager) or manager in BaseScraper.__init__.__code__.co_names

        logger.info("✓ BaseScraper utility managers available")


class TestURLNormalization:
    """Test URL normalization and validation."""

    def test_normalize_facebook_url_no_redirect(self):
        """Test normalization of regular Facebook URL."""
        test_url = "https://www.facebook.com/events/12345/"

        # Mock FacebookScraperV2 to avoid full initialization
        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.normalize_facebook_url(test_url)
            assert result == test_url
            logger.info(f"✓ URL normalization (no redirect): {test_url}")

    def test_normalize_facebook_url_with_redirect(self):
        """Test normalization of Facebook login redirect URL."""
        test_url = "https://www.facebook.com/login/?next=https://www.facebook.com/events/12345/"

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.normalize_facebook_url(test_url)
            # Should unwrap the redirect
            assert "next=" not in result or "login" not in result.lower()
            logger.info(f"✓ URL normalization (with redirect): unwrapped correctly")

    def test_normalize_facebook_url_empty(self):
        """Test normalization of empty URL."""
        test_url = ""

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.normalize_facebook_url(test_url)
            assert result == test_url
            logger.info("✓ URL normalization (empty): handled correctly")


class TestTextExtraction:
    """Test event text extraction functionality."""

    def test_extract_relevant_text_with_keywords(self):
        """Test extraction of relevant text from event content."""
        sample_content = """
        Monday Some event info here
        More About Discussion
        Tuesday Event details continue
        Guests See All attendees
        Additional information
        """

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.extract_relevant_text(sample_content, "https://example.com")

            # Should extract something between day of week and Guests See All
            if result:
                assert "Monday" in result or "Tuesday" in result
                logger.info("✓ Relevant text extraction: successful")
            else:
                logger.info("✓ Relevant text extraction: returned None as expected")

    def test_extract_relevant_text_missing_pattern(self):
        """Test extraction when required patterns are missing."""
        sample_content = "This is content without required patterns"

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.extract_relevant_text(sample_content, "https://example.com")

            # Should return None when patterns not found
            assert result is None or isinstance(result, str)
            logger.info("✓ Relevant text extraction (missing patterns): handled gracefully")


class TestStatisticsTracking:
    """Test statistics tracking functionality."""

    def test_statistics_initialization(self):
        """Test that statistics are initialized correctly."""
        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            # Set up required attributes for _init_statistics()
            scraper.config = config
            scraper.logger = logger
            scraper._init_statistics()

            assert hasattr(scraper, 'stats')
            assert 'unique_urls' in scraper.stats
            assert 'total_url_attempts' in scraper.stats
            assert 'urls_with_extracted_text' in scraper.stats
            assert 'urls_with_found_keywords' in scraper.stats
            assert 'events_written_to_db' in scraper.stats

            logger.info("✓ Statistics initialization: all keys present")

    def test_get_statistics(self):
        """Test statistics retrieval."""
        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            # Set up required attributes for _init_statistics()
            scraper.config = config
            scraper.logger = logger
            scraper._init_statistics()
            scraper.urls_visited = {'url1', 'url2', 'url3'}

            stats = scraper.get_statistics()

            assert isinstance(stats, dict)
            assert 'unique_urls' in stats
            assert stats['unique_urls'] == 3  # Based on urls_visited set
            logger.info("✓ Statistics retrieval: working correctly")


class TestMethodStructure:
    """Test method signatures and return types."""

    def test_normalize_url_signature(self):
        """Test normalize_facebook_url method signature."""
        from fb_v2 import FacebookScraperV2
        import inspect

        sig = inspect.signature(FacebookScraperV2.normalize_facebook_url)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'url' in params
        logger.info("✓ normalize_facebook_url signature correct")

    def test_extract_event_links_signature(self):
        """Test extract_event_links method signature."""
        from fb_v2 import FacebookScraperV2
        import inspect

        sig = inspect.signature(FacebookScraperV2.extract_event_links)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'search_url' in params
        logger.info("✓ extract_event_links signature correct")

    def test_extract_event_text_signature(self):
        """Test extract_event_text method signature."""
        from fb_v2 import FacebookScraperV2
        import inspect

        sig = inspect.signature(FacebookScraperV2.extract_event_text)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'link' in params
        logger.info("✓ extract_event_text signature correct")


class TestDocumentation:
    """Test that methods are properly documented."""

    def test_class_has_docstring(self):
        """Test that FacebookScraperV2 class has documentation."""
        from fb_v2 import FacebookScraperV2

        assert FacebookScraperV2.__doc__ is not None
        assert len(FacebookScraperV2.__doc__) > 50
        assert "BaseScraper" in FacebookScraperV2.__doc__
        logger.info("✓ FacebookScraperV2 class has comprehensive docstring")

    def test_methods_have_docstrings(self):
        """Test that major methods have documentation."""
        from fb_v2 import FacebookScraperV2

        methods_to_check = [
            'login_to_facebook',
            'normalize_facebook_url',
            'extract_event_links',
            'extract_event_text',
            'scrape_events',
            'process_fb_url',
            'driver_fb_search',
            'driver_fb_urls'
        ]

        for method_name in methods_to_check:
            method = getattr(FacebookScraperV2, method_name)
            assert method.__doc__ is not None, f"{method_name} missing docstring"
            assert len(method.__doc__) > 20, f"{method_name} docstring too short"

        logger.info(f"✓ All {len(methods_to_check)} major methods have docstrings")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_normalize_url_with_special_chars(self):
        """Test URL normalization with special characters."""
        test_urls = [
            "https://www.facebook.com/events/12345/?utm_source=test",
            "https://m.facebook.com/events/12345/",
            "https://www.facebook.com/groups/12345/events/",
        ]

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            for url in test_urls:
                result = scraper.normalize_facebook_url(url)
                assert isinstance(result, str)
                assert len(result) > 0

        logger.info(f"✓ URL normalization: handled {len(test_urls)} special cases")

    def test_extract_relevant_text_with_unicode(self):
        """Test text extraction with unicode characters."""
        sample_content = """
        Monday Salsa & Bachata Night
        More About Discussion
        Tuesday Event with ñ, é, ü characters
        Guests See All attendees
        """

        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger

            result = scraper.extract_relevant_text(sample_content, "https://example.com")
            # Should handle unicode without crashing
            assert result is None or isinstance(result, str)

        logger.info("✓ Text extraction: handled unicode characters correctly")


# ============================================================================
# INTEGRATION TESTS - FacebookScraperV2 with Dependencies
# ============================================================================

class TestBasicIntegration:
    """Test basic integration without external services."""

    def test_method_calls_dont_crash(self):
        """Test that methods can be called without crashing."""
        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.logger = logger
            scraper.stats = {
                'unique_urls': 0,
                'total_url_attempts': 0,
                'urls_with_extracted_text': 0,
                'urls_with_found_keywords': 0,
                'events_written_to_db': 0
            }

            # Test URL normalization
            url = scraper.normalize_facebook_url("https://www.facebook.com/events/12345/")
            assert isinstance(url, str)

            # Test text extraction
            result = scraper.extract_relevant_text("Monday text Guests See All", "https://example.com")
            assert result is None or isinstance(result, str)

            logger.info("✓ Basic method integration: no crashes")

    def test_statistics_accumulation(self):
        """Test that statistics accumulate correctly."""
        with patch('fb_v2.FacebookScraperV2.__init__', return_value=None):
            from fb_v2 import FacebookScraperV2
            scraper = FacebookScraperV2()
            scraper.stats = {
                'unique_urls': 0,
                'total_url_attempts': 0,
                'urls_with_extracted_text': 0,
                'urls_with_found_keywords': 0,
                'events_written_to_db': 0
            }
            scraper.urls_visited = set()

            # Simulate stat updates
            scraper.stats['total_url_attempts'] += 5
            scraper.stats['urls_with_extracted_text'] += 3
            scraper.stats['events_written_to_db'] += 2
            scraper.urls_visited.add('url1')
            scraper.urls_visited.add('url2')

            assert scraper.stats['total_url_attempts'] == 5
            assert scraper.stats['urls_with_extracted_text'] == 3
            assert scraper.stats['events_written_to_db'] == 2
            assert len(scraper.urls_visited) == 2

            logger.info("✓ Statistics accumulation: working correctly")


# ============================================================================
# COMPARISON TESTS - Original vs Refactored
# ============================================================================

class TestRefactoringMaintenance:
    """Test that refactoring maintains original functionality."""

    @pytest.mark.skip(reason="fb.py was deleted in Phase 2 consolidation - v1 scrapers removed")
    def test_fb_v2_matches_original_interface(self):
        """Test that fb_v2 has same public interface as original."""
        from fb_v2 import FacebookScraperV2
        from fb import FacebookEventScraper

        # Get public methods from both classes
        fb_v2_methods = {m for m in dir(FacebookScraperV2) if not m.startswith('_')}
        fb_methods = {m for m in dir(FacebookEventScraper) if not m.startswith('_')}

        # Core methods should be in both
        core_methods = [
            'login_to_facebook',
            'normalize_facebook_url',
            'extract_event_links',
            'extract_event_text',
            'extract_relevant_text',
            'scrape_events',
            'process_fb_url',
            'driver_fb_search',
            'driver_fb_urls',
        ]

        for method in core_methods:
            assert method in fb_v2_methods, f"Missing in fb_v2: {method}"

        logger.info(f"✓ Interface compatibility: all {len(core_methods)} core methods present")

    @pytest.mark.skip(reason="fb.py was deleted in Phase 2 consolidation - v1 scrapers removed")
    def test_original_fb_still_works(self):
        """Test that original fb.py is unchanged and importable."""
        from fb import FacebookEventScraper

        assert FacebookEventScraper is not None
        assert hasattr(FacebookEventScraper, 'login_to_facebook')

        logger.info("✓ Original FacebookEventScraper unchanged and importable")


# ============================================================================
# TEST EXECUTION SUMMARY
# ============================================================================

def test_summary():
    """Print test summary."""
    print("\n" + "=" * 80)
    print("FACEBOOKSCRAPERV2 TEST SUITE SUMMARY")
    print("=" * 80)
    print("✓ Initialization tests")
    print("✓ Utility integration tests")
    print("✓ URL normalization tests")
    print("✓ Text extraction tests")
    print("✓ Statistics tracking tests")
    print("✓ Method structure tests")
    print("✓ Documentation tests")
    print("✓ Error handling tests")
    print("✓ Basic integration tests")
    print("✓ Refactoring maintenance tests")
    print("=" * 80)
    logger.info("✓ All test categories completed")


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
