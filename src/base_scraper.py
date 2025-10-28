"""
Base scraper class for unified web scraping implementation.

This module provides an abstract base class that consolidates common patterns
and interfaces across all web scrapers (FacebookEventScraper, ImageScraper,
EventbriteScraperRaw, GeneralWebScraper).

Classes:
    BaseScraper: Abstract base class for all scrapers

Key responsibilities:
    - Unified initialization and configuration
    - Browser and page management
    - Authentication handling
    - Text and data extraction
    - Database operations
    - Error handling and retry logic
    - Statistics tracking
"""

import logging
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Set, List
import pandas as pd

from browser_utils import PlaywrightManager
from text_utils import TextExtractor
from auth_manager import AuthenticationManager
from resilience import RetryManager, CircuitBreaker
from url_nav import URLNavigator
from pdf_utils import PDFExtractor
from db_utils import DBWriter


class BaseScraper(ABC):
    """
    Abstract base class for all web scrapers.

    Consolidates common initialization, configuration, and operation patterns
    across FacebookEventScraper, ImageScraper, EventbriteScraperRaw, and GeneralWebScraper.

    Subclasses must implement:
        - scrape(): Main scraping logic
        - specific platform methods as needed
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize BaseScraper with common configuration and utilities.

        Args:
            config_path (str): Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}")

        # Initialize utility managers
        self.browser_manager = PlaywrightManager(self.config)
        self.text_extractor = TextExtractor(self.logger)
        self.auth_manager = AuthenticationManager(self.logger)
        self.retry_manager = RetryManager(logger=self.logger)
        self.url_navigator = URLNavigator(self.logger)
        self.pdf_extractor = PDFExtractor(logger=self.logger)
        self.db_writer = None  # Set by subclasses with DatabaseHandler

        # Initialize browser and page (can be set by subclasses)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # URL tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()

        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'urls_visited': 0,
            'urls_failed': 0,
            'events_extracted': 0,
            'events_written': 0,
            'errors': [],
        }

        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(logger=self.logger)

    @abstractmethod
    def scrape(self) -> pd.DataFrame:
        """
        Main scraping method - must be implemented by subclasses.

        Returns:
            pd.DataFrame: Extracted events
        """
        pass

    def set_db_writer(self, db_handler) -> None:
        """
        Set the database writer with a DatabaseHandler instance.

        Args:
            db_handler: DatabaseHandler instance
        """
        self.db_writer = DBWriter(db_handler, self.logger)
        self.logger.info("Database writer initialized")

    def get_config(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.

        Args:
            key_path (str): Dot-separated path (e.g., 'crawling.headless')
            default: Default value if not found

        Returns:
            Any: Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value.get(key, {})
            return value if value != {} else default
        except:
            return default

    def can_execute(self) -> bool:
        """
        Check if scraper can execute based on circuit breaker status.

        Returns:
            bool: True if can execute, False if circuit is broken
        """
        if not self.circuit_breaker.can_execute():
            self.logger.warning("Circuit breaker is open, cannot execute scraper")
            return False
        return True

    def record_success(self) -> None:
        """Record a successful operation for circuit breaker."""
        self.circuit_breaker.record_success()

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Record a failed operation for circuit breaker.

        Args:
            error (Exception, optional): Error that occurred
        """
        self.circuit_breaker.record_failure()
        if error:
            self.stats['errors'].append(str(error))

    def add_visited_url(self, url: str) -> None:
        """
        Track a visited URL.

        Args:
            url (str): URL that was visited
        """
        self.url_navigator.add_visited_url(url)
        self.visited_urls.add(url)
        self.stats['urls_visited'] += 1

    def add_failed_url(self, url: str) -> None:
        """
        Track a failed URL.

        Args:
            url (str): URL that failed
        """
        self.url_navigator.add_failed_url(url)
        self.failed_urls.add(url)
        self.stats['urls_failed'] += 1

    def is_visited(self, url: str) -> bool:
        """
        Check if URL has been visited.

        Args:
            url (str): URL to check

        Returns:
            bool: True if visited
        """
        return self.url_navigator.is_visited(url)

    def is_valid_url(self, url: str) -> bool:
        """
        Validate URL.

        Args:
            url (str): URL to validate

        Returns:
            bool: True if valid
        """
        return self.url_navigator.is_valid_url(url)

    def extract_text(self, html: str, min_length: int = 10) -> Optional[str]:
        """
        Extract clean text from HTML.

        Args:
            html (str): HTML content
            min_length (int): Minimum text length to consider valid

        Returns:
            Optional[str]: Extracted text or None
        """
        return self.text_extractor.extract_from_html(html, min_length)

    def extract_links(self, html: str, base_url: Optional[str] = None) -> Set[str]:
        """
        Extract all links from HTML.

        Args:
            html (str): HTML content
            base_url (str, optional): Base URL for relative links

        Returns:
            Set[str]: Set of absolute URLs
        """
        return self.text_extractor.extract_links_from_html(html, base_url)

    def extract_id_from_url(self, url: str, platform: str = "eventbrite") -> Optional[str]:
        """
        Extract unique ID from platform-specific URL.

        Supports extracting IDs from various event platforms:
        - eventbrite: /e/{event-name}-{numeric-id}?params
        - facebook: /events/{numeric-id}/
        - generic: attempts to extract numeric ID from path

        Args:
            url (str): URL to extract ID from
            platform (str): Platform identifier (eventbrite, facebook, generic)

        Returns:
            Optional[str]: Extracted ID or None if not found
        """
        import re

        try:
            if platform == "eventbrite":
                # Eventbrite URLs: /e/event-name-12345678?params
                # Match event name part (lazy), then numeric ID, then either ? or end of string
                match = re.search(r'/e/[a-z0-9\-]*?(\d+)(?:\?|$)', url, re.IGNORECASE)
                if match:
                    return match.group(1)

            elif platform == "facebook":
                # Facebook URLs: /events/123456789/ or ?event_id=123456789
                # Try both patterns
                match = re.search(r'/events/(\d+)', url)
                if match:
                    return match.group(1)
                match = re.search(r'event_id=(\d+)', url)
                if match:
                    return match.group(1)

            elif platform == "generic":
                # Generic numeric ID extraction from any URL path
                match = re.search(r'(\d+)', url.split('?')[0])
                if match:
                    return match.group(1)

            self.logger.debug(f"No ID found in URL {url} for platform {platform}")
            return None

        except Exception as e:
            self.logger.error(f"Error extracting ID from {url} (platform: {platform}): {e}")
            return None

    def write_events_to_db(self, events: List[Dict[str, Any]]) -> bool:
        """
        Write events to database.

        Args:
            events (list): List of event dictionaries

        Returns:
            bool: True if successful
        """
        if not self.db_writer:
            self.logger.error("Database writer not initialized")
            return False

        try:
            success_count, fail_count = self.db_writer.write_events_to_db(events)
            self.stats['events_written'] += success_count
            return fail_count == 0
        except Exception as e:
            self.logger.error(f"Failed to write events: {e}")
            self.record_failure(e)
            return False

    def write_events_dataframe_to_db(self, df: pd.DataFrame) -> bool:
        """
        Write DataFrame of events to database.

        Args:
            df (pd.DataFrame): Events DataFrame

        Returns:
            bool: True if successful
        """
        if not self.db_writer:
            self.logger.error("Database writer not initialized")
            return False

        try:
            success = self.db_writer.write_dataframe_to_db(df, 'events')
            if success:
                self.stats['events_written'] += len(df)
            return success
        except Exception as e:
            self.logger.error(f"Failed to write events DataFrame: {e}")
            self.record_failure(e)
            return False

    def write_url_to_db(self, url: str, parent_url: str, source: str,
                       keywords, success: bool, retry_count: int = 1) -> bool:
        """
        Write URL tracking data to database.

        Args:
            url (str): URL
            parent_url (str): Parent URL
            source (str): Source identifier
            keywords: Associated keywords
            success (bool): Whether URL was successfully processed
            retry_count (int): Number of retries

        Returns:
            bool: True if successful
        """
        if not self.db_writer:
            self.logger.error("Database writer not initialized")
            return False

        try:
            url_data = (url, parent_url, source, keywords, success, retry_count, datetime.now())
            return self.db_writer.write_url_to_db(url_data)
        except Exception as e:
            self.logger.error(f"Failed to write URL to database: {e}")
            return False

    def get_statistics(self) -> dict:
        """
        Get scraper execution statistics.

        Returns:
            dict: Statistics dictionary
        """
        elapsed_time = (datetime.now() - self.stats['start_time']).total_seconds()
        total_urls = self.stats['urls_visited'] + self.stats['urls_failed']
        success_rate = (
            (self.stats['urls_visited'] / total_urls * 100) if total_urls > 0 else 0
        )

        return {
            'scraper': self.__class__.__name__,
            'start_time': self.stats['start_time'].isoformat(),
            'elapsed_seconds': elapsed_time,
            'urls_visited': self.stats['urls_visited'],
            'urls_failed': self.stats['urls_failed'],
            'url_success_rate': f"{success_rate:.1f}%",
            'events_extracted': self.stats['events_extracted'],
            'events_written': self.stats['events_written'],
            'errors': len(self.stats['errors']),
            'circuit_breaker_state': self.circuit_breaker.state,
        }

    def log_statistics(self) -> None:
        """Log scraper statistics."""
        stats = self.get_statistics()
        self.logger.info(f"=== {self.__class__.__name__} Statistics ===")
        self.logger.info(f"Elapsed time: {stats['elapsed_seconds']:.1f}s")
        self.logger.info(f"URLs visited: {stats['urls_visited']}")
        self.logger.info(f"URLs failed: {stats['urls_failed']}")
        self.logger.info(f"Success rate: {stats['url_success_rate']}")
        self.logger.info(f"Events extracted: {stats['events_extracted']}")
        self.logger.info(f"Events written: {stats['events_written']}")
        if stats['errors'] > 0:
            self.logger.warning(f"Errors encountered: {stats['errors']}")

    def cleanup(self) -> None:
        """Clean up resources (browser, connections, etc.)."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info(f"{self.__class__.__name__} cleanup complete")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
