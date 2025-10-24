"""
URL navigation and link discovery utilities for web scrapers.

This module consolidates URL tracking, link extraction, filtering, and
navigation patterns used across multiple scrapers (scraper.py, rd_ext.py, images.py).

Classes:
    URLNavigator: Unified URL discovery, filtering, and tracking

Key responsibilities:
    - URL tracking and deduplication
    - Same-domain filtering
    - Link discovery and filtering
    - URL normalization
    - Visited URL management
"""

import logging
from typing import Set, Optional, List, Tuple
from urllib.parse import urlparse, urljoin, urlunparse
import re


class URLNavigator:
    """
    Unified URL navigation and link discovery for web scrapers.

    Consolidates URL tracking, filtering, and deduplication patterns
    that are repeated across all scrapers.
    """

    # URL pattern for filtering
    EXCLUDED_EXTENSIONS = {
        '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe',
        '.css', '.js', '.woff', '.ttf', '.svg', '.ico',
    }

    EXCLUDED_DOMAINS = {
        'facebook.com', 'instagram.com', 'twitter.com', 'youtube.com',
        'google.com', 'facebook.com', 'ads.google.com',
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize URLNavigator.

        Args:
            logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()

    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs are from the same domain.

        Args:
            url1 (str): First URL
            url2 (str): Second URL

        Returns:
            bool: True if both URLs share the same domain
        """
        try:
            domain1 = urlparse(url1).netloc.lower()
            domain2 = urlparse(url2).netloc.lower()

            # Remove 'www.' prefix for comparison
            domain1 = domain1.replace('www.', '')
            domain2 = domain2.replace('www.', '')

            return domain1 == domain2

        except Exception as e:
            self.logger.error(f"Failed to compare domains: {e}")
            return False

    def is_valid_url(self, url: str) -> bool:
        """
        Validate URL format and excluded patterns.

        Args:
            url (str): URL to validate

        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            if not url or not isinstance(url, str):
                return False

            # Check excluded extensions
            if any(url.lower().endswith(ext) for ext in self.EXCLUDED_EXTENSIONS):
                return False

            # Check for excluded domains
            domain = urlparse(url).netloc.lower().replace('www.', '')
            if any(excluded in domain for excluded in self.EXCLUDED_DOMAINS):
                return False

            # Check for common exclusions in URL path
            if any(pattern in url.lower() for pattern in ['javascript:', 'mailto:', '#', '?logout']):
                return False

            # Check protocol
            if not url.startswith(('http://', 'https://')):
                return False

            return True

        except Exception as e:
            self.logger.warning(f"URL validation error for {url}: {e}")
            return False

    def normalize_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
        """
        Normalize and resolve URL.

        Handles relative URLs, fragment removal, and query parameter sorting
        for consistent URL comparison.

        Args:
            url (str): URL to normalize
            base_url (str, optional): Base URL for relative URL resolution

        Returns:
            Optional[str]: Normalized absolute URL, or None if invalid
        """
        try:
            # Handle relative URLs
            if base_url and not url.startswith(('http://', 'https://')):
                url = urljoin(base_url, url)

            # Parse URL
            parsed = urlparse(url)

            # Remove fragment
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path,
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))

            # Clean up trailing slashes
            if normalized.endswith('/') and not normalized.endswith('://'):
                normalized = normalized.rstrip('/')

            return normalized if self.is_valid_url(normalized) else None

        except Exception as e:
            self.logger.warning(f"URL normalization error for {url}: {e}")
            return None

    def add_visited_url(self, url: str) -> None:
        """
        Track a visited URL.

        Args:
            url (str): URL that was visited
        """
        normalized = self.normalize_url(url)
        if normalized:
            self.visited_urls.add(normalized)

    def add_failed_url(self, url: str) -> None:
        """
        Track a failed URL.

        Args:
            url (str): URL that failed
        """
        normalized = self.normalize_url(url)
        if normalized:
            self.failed_urls.add(normalized)

    def is_visited(self, url: str) -> bool:
        """
        Check if URL has been visited.

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL has been visited
        """
        normalized = self.normalize_url(url)
        return normalized in self.visited_urls if normalized else False

    def is_failed(self, url: str) -> bool:
        """
        Check if URL has failed.

        Args:
            url (str): URL to check

        Returns:
            bool: True if URL has failed
        """
        normalized = self.normalize_url(url)
        return normalized in self.failed_urls if normalized else False

    def should_visit_url(self, url: str, base_domain: Optional[str] = None,
                         allow_external: bool = False) -> bool:
        """
        Determine if a URL should be visited based on various criteria.

        Args:
            url (str): URL to evaluate
            base_domain (str, optional): Base domain to restrict crawling to
            allow_external (bool): Whether to allow external domain links

        Returns:
            bool: True if URL should be visited, False otherwise
        """
        # Check if already visited or failed
        if self.is_visited(url) or self.is_failed(url):
            return False

        # Validate URL
        if not self.is_valid_url(url):
            return False

        # Check domain restriction
        if base_domain and not allow_external:
            if not self.is_same_domain(url, base_domain):
                return False

        return True

    def filter_links(self, links: Set[str], base_domain: Optional[str] = None,
                     allow_external: bool = False, max_urls: Optional[int] = None) -> Set[str]:
        """
        Filter links based on validation criteria.

        Args:
            links (Set[str]): Set of links to filter
            base_domain (str, optional): Base domain to restrict crawling to
            allow_external (bool): Whether to allow external domain links
            max_urls (int, optional): Maximum number of URLs to return

        Returns:
            Set[str]: Filtered set of links
        """
        filtered = set()

        for link in links:
            if self.should_visit_url(link, base_domain, allow_external):
                filtered.add(link)

            if max_urls and len(filtered) >= max_urls:
                break

        return filtered

    def get_domain_from_url(self, url: str) -> Optional[str]:
        """
        Extract domain from URL.

        Args:
            url (str): URL to extract domain from

        Returns:
            Optional[str]: Domain, or None if invalid
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            return domain if domain else None
        except:
            return None

    def get_path_from_url(self, url: str) -> Optional[str]:
        """
        Extract path from URL.

        Args:
            url (str): URL to extract path from

        Returns:
            Optional[str]: Path component of URL
        """
        try:
            return urlparse(url).path or '/'
        except:
            return None

    def resolve_relative_url(self, relative_url: str, base_url: str) -> Optional[str]:
        """
        Resolve a relative URL against a base URL.

        Args:
            relative_url (str): Relative URL
            base_url (str): Base URL

        Returns:
            Optional[str]: Absolute URL
        """
        try:
            absolute = urljoin(base_url, relative_url)
            return self.normalize_url(absolute)
        except Exception as e:
            self.logger.warning(f"Failed to resolve relative URL: {e}")
            return None

    def get_statistics(self) -> dict:
        """
        Get URL tracking statistics.

        Returns:
            dict: Statistics about visited and failed URLs
        """
        return {
            'visited_count': len(self.visited_urls),
            'failed_count': len(self.failed_urls),
            'total_tracked': len(self.visited_urls) + len(self.failed_urls),
            'success_rate': (
                len(self.visited_urls) / (len(self.visited_urls) + len(self.failed_urls))
                if (len(self.visited_urls) + len(self.failed_urls)) > 0
                else 0
            )
        }

    def reset(self) -> None:
        """Reset all tracking."""
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.logger.info("URL navigator reset")

    def get_visited_urls(self) -> Set[str]:
        """
        Get all visited URLs.

        Returns:
            Set[str]: Set of visited URLs
        """
        return self.visited_urls.copy()

    def get_failed_urls(self) -> Set[str]:
        """
        Get all failed URLs.

        Returns:
            Set[str]: Set of failed URLs
        """
        return self.failed_urls.copy()

    def extract_domain_list(self, urls: Set[str]) -> List[str]:
        """
        Extract unique domains from a set of URLs.

        Args:
            urls (Set[str]): Set of URLs

        Returns:
            List[str]: List of unique domains
        """
        domains = set()
        for url in urls:
            domain = self.get_domain_from_url(url)
            if domain:
                domains.add(domain)
        return sorted(list(domains))

    def batch_normalize_urls(self, urls: List[str], base_url: Optional[str] = None) -> Set[str]:
        """
        Normalize a batch of URLs.

        Args:
            urls (List[str]): List of URLs to normalize
            base_url (str, optional): Base URL for relative URL resolution

        Returns:
            Set[str]: Set of normalized URLs
        """
        normalized = set()
        for url in urls:
            norm = self.normalize_url(url, base_url)
            if norm:
                normalized.add(norm)
        return normalized
