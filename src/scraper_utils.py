"""
Common utilities for web scrapers.

Consolidates shared patterns from ebs.py, images.py, and fb.py to reduce
code duplication and ensure consistent behavior across all scrapers.

Includes:
- Keyword filtering utilities
- Timeout/delay utilities
- URL processing utilities
- Common validation checks
"""

import logging
import random
from typing import List, Set, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# KEYWORD UTILITIES
# ============================================================================

def check_keywords(text: str, keywords_list: List[str]) -> List[str]:
    """
    Check for keywords in text (case-insensitive).

    Consolidates duplicated keyword filtering logic from ebs.py, images.py, fb.py.

    Args:
        text (str): Text to search in
        keywords_list (List[str]): Keywords to search for

    Returns:
        List[str]: Keywords found in the text (case-insensitive)

    Example:
        >>> check_keywords("I love Tango and Salsa", ["tango", "waltz", "salsa"])
        ['tango', 'salsa']
    """
    if not text or not keywords_list:
        return []

    text_lower = text.lower()
    found = [kw for kw in keywords_list if kw.lower() in text_lower]

    if found:
        logger.debug(f"check_keywords(): Found {len(found)} keywords: {found}")

    return found


def has_keywords(text: str, keywords_list: List[str]) -> bool:
    """
    Check if ANY keywords exist in text.

    Args:
        text (str): Text to search in
        keywords_list (List[str]): Keywords to search for

    Returns:
        bool: True if any keyword found, False otherwise
    """
    return len(check_keywords(text, keywords_list)) > 0


# ============================================================================
# TIMEOUT AND DELAY UTILITIES
# ============================================================================

def get_random_timeout(base_ms: int = 20000) -> int:
    """
    Generate random timeout that mimics human behavior.

    Consolidates timeout pattern appearing 20+ times across scrapers.
    Random timeout range: [base_ms / 2, base_ms * 1.5]

    Args:
        base_ms (int): Base timeout in milliseconds (default: 20000ms = 20s)

    Returns:
        int: Random timeout in milliseconds

    Example:
        >>> timeout_ms = get_random_timeout(20000)
        >>> 10000 <= timeout_ms <= 30000
        True
    """
    min_ms = base_ms // 2
    max_ms = int(base_ms * 1.5)
    timeout = random.randint(min_ms, max_ms)
    return timeout


def get_random_delay(base_seconds: float = 2.0) -> float:
    """
    Generate random delay between operations (in seconds).

    Mimics human browsing patterns with variable delays.
    Random range: [base_seconds * 0.5, base_seconds * 1.5]

    Args:
        base_seconds (float): Base delay in seconds (default: 2.0)

    Returns:
        float: Random delay in seconds

    Example:
        >>> delay = get_random_delay(2.0)
        >>> 1.0 <= delay <= 3.0
        True
    """
    min_delay = base_seconds * 0.5
    max_delay = base_seconds * 1.5
    return random.uniform(min_delay, max_delay)


# ============================================================================
# URL PROCESSING UTILITIES
# ============================================================================

class URLProcessor:
    """
    Common URL processing workflow used by all scrapers.

    Consolidates URL validation, visit tracking, and crawling limits
    from ebs.py, images.py, and fb.py.

    Attributes:
        config (dict): Configuration dictionary
        visited_urls (Set[str]): Set of visited URLs
        crawled_count (int): Count of URLs processed
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize URL processor.

        Args:
            config (dict): Configuration with keys like:
                - urls_run_limit: Maximum URLs to process
                - blacklist: Set of blacklisted domains
        """
        self.config = config or {}
        self.visited_urls: Set[str] = set()
        self.crawled_count = 0
        self.logger = logging.getLogger(__name__)

    def should_process_url(self, url: str) -> bool:
        """
        Check if URL should be processed.

        Returns False if:
        - URL already visited
        - URL is blacklisted
        - Crawling limits exceeded

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL should be processed, False otherwise
        """
        # Check if already visited
        if url in self.visited_urls:
            self.logger.debug(f"URL already visited: {url}")
            return False

        # Check blacklist
        if self._is_blacklisted(url):
            self.logger.debug(f"URL is blacklisted: {url}")
            return False

        # Check crawling limits
        urls_limit = self.config.get('urls_run_limit', 500)
        if self.crawled_count >= urls_limit:
            self.logger.info(f"Crawling limit reached: {self.crawled_count}/{urls_limit}")
            return False

        return True

    def mark_processed(self, url: str) -> None:
        """
        Mark URL as processed/visited.

        Args:
            url (str): URL to mark as visited
        """
        self.visited_urls.add(url)
        self.crawled_count += 1

    def is_visited(self, url: str) -> bool:
        """Check if URL has been visited."""
        return url in self.visited_urls

    def _is_blacklisted(self, url: str) -> bool:
        """
        Check if URL domain is blacklisted.

        Args:
            url (str): URL to check

        Returns:
            bool: True if blacklisted, False otherwise
        """
        blacklist = self.config.get('blacklist', [])
        avoid_domains = self.config.get('avoid_domains', [])

        # Check blacklist entries
        for entry in blacklist + avoid_domains:
            if entry.lower() in url.lower():
                return True

        return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            dict: Statistics including visited count and crawled count
        """
        return {
            'visited': len(self.visited_urls),
            'crawled': self.crawled_count,
            'remaining': self.config.get('urls_run_limit', 500) - self.crawled_count
        }


# ============================================================================
# TEXT VALIDATION UTILITIES
# ============================================================================

def is_valid_text(text: str, min_length: int = 10) -> bool:
    """
    Check if extracted text is valid.

    Args:
        text (str): Text to validate
        min_length (int): Minimum text length (default: 10 chars)

    Returns:
        bool: True if text is valid, False otherwise
    """
    if not text:
        return False

    text_stripped = text.strip()
    return len(text_stripped) >= min_length


def is_mostly_whitespace(text: str) -> bool:
    """
    Check if text is mostly whitespace.

    Args:
        text (str): Text to check

    Returns:
        bool: True if > 90% whitespace, False otherwise
    """
    if not text:
        return True

    whitespace_ratio = len(text) - len(text.strip())
    return whitespace_ratio / len(text) > 0.9


# ============================================================================
# COMMON VALIDATION PATTERNS
# ============================================================================

def should_skip_url_domain(url: str, skip_domains: List[str]) -> bool:
    """
    Check if URL should be skipped based on domain.

    Common skip domains: facebook.com, instagram.com, twitter.com, etc.

    Args:
        url (str): URL to check
        skip_domains (List[str]): Domains to skip

    Returns:
        bool: True if URL should be skipped, False otherwise

    Example:
        >>> should_skip_url_domain("https://facebook.com/event/123",
        ...                        ["facebook.com", "instagram.com"])
        True
    """
    url_lower = url.lower()
    return any(domain.lower() in url_lower for domain in skip_domains)


def get_domain_from_url(url: str) -> str:
    """
    Extract domain from URL.

    Args:
        url (str): URL to extract domain from

    Returns:
        str: Domain name

    Example:
        >>> get_domain_from_url("https://www.eventbrite.com/e/123")
        "eventbrite.com"
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain


# ============================================================================
# STATISTICS AND LOGGING UTILITIES
# ============================================================================

class ScraperStats:
    """
    Common statistics tracking for scrapers.

    Consolidates stats tracking from ebs.py and fb.py.
    """

    def __init__(self, scraper_name: str):
        """
        Initialize stats tracker.

        Args:
            scraper_name (str): Name of scraper (e.g., 'ebs', 'facebook', 'images')
        """
        self.scraper_name = scraper_name
        self.logger = logging.getLogger(__name__)

        # Tracking counters
        self.urls_visited = 0
        self.urls_with_extracted_text = 0
        self.urls_with_found_keywords = 0
        self.events_written_to_db = 0
        self.errors_encountered = 0

        # Timing
        self.start_time = datetime.now()
        self.end_time = None

    def record_url_visited(self) -> None:
        """Record that a URL was visited."""
        self.urls_visited += 1

    def record_text_extracted(self) -> None:
        """Record successful text extraction."""
        self.urls_with_extracted_text += 1

    def record_keywords_found(self) -> None:
        """Record that keywords were found."""
        self.urls_with_found_keywords += 1

    def record_event_written(self) -> None:
        """Record that event was written to database."""
        self.events_written_to_db += 1

    def record_error(self) -> None:
        """Record that an error occurred."""
        self.errors_encountered += 1

    def finalize(self) -> None:
        """Mark stats as finalized."""
        self.end_time = datetime.now()

    def get_duration(self) -> timedelta:
        """
        Get total duration of scraping.

        Returns:
            timedelta: Duration from start to end
        """
        end = self.end_time or datetime.now()
        return end - self.start_time

    def get_stats_dict(self) -> Dict[str, Any]:
        """
        Get all statistics as dictionary.

        Returns:
            dict: Statistics dictionary
        """
        return {
            'scraper': self.scraper_name,
            'urls_visited': self.urls_visited,
            'urls_with_text': self.urls_with_extracted_text,
            'urls_with_keywords': self.urls_with_found_keywords,
            'events_written': self.events_written_to_db,
            'errors': self.errors_encountered,
            'duration_seconds': self.get_duration().total_seconds(),
        }

    def log_summary(self) -> None:
        """Log statistics summary."""
        stats = self.get_stats_dict()
        self.logger.info(f"\n=== {self.scraper_name.upper()} Statistics ===")
        self.logger.info(f"URLs visited: {stats['urls_visited']}")
        self.logger.info(f"URLs with extracted text: {stats['urls_with_text']}")
        self.logger.info(f"URLs with keywords found: {stats['urls_with_keywords']}")
        self.logger.info(f"Events written to DB: {stats['events_written']}")
        self.logger.info(f"Errors: {stats['errors']}")
        self.logger.info(f"Duration: {stats['duration_seconds']:.2f}s")
        self.logger.info("=" * 40)
