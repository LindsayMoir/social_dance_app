"""
Text extraction and processing utilities for web scraping.

This module consolidates text extraction patterns used across multiple scrapers
(scraper.py, rd_ext.py, images.py, ebs.py) including BeautifulSoup-based HTML parsing,
text cleaning, and content normalization.

Classes:
    TextExtractor: Unified text extraction from HTML and web pages

Key responsibilities:
    - HTML to text extraction using BeautifulSoup
    - Text cleaning and normalization
    - Content deduplication
    - Special content handling (scripts, styles, etc.)
"""

import logging
import re
from typing import Optional, Set
from bs4 import BeautifulSoup
from playwright.sync_api import Page as SyncPage
from playwright.async_api import Page as AsyncPage


class TextExtractor:
    """
    Unified text extraction from HTML content and Playwright pages.

    Consolidates text extraction patterns that are repeated across all scrapers,
    providing consistent text cleaning and normalization.
    """

    # Tags and content to exclude from extraction
    EXCLUDED_TAGS = {'script', 'style', 'noscript', 'meta', 'link', 'head'}

    # Common noise patterns to remove
    NOISE_PATTERNS = [
        r'\s+',  # Multiple whitespace
        r'^\s+|\s+$',  # Leading/trailing whitespace
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize TextExtractor.

        Args:
            logger (logging.Logger, optional): Logger instance for logging extraction details
        """
        self.logger = logger or logging.getLogger(__name__)

    def extract_from_html(self, html_content: str, min_length: int = 10) -> Optional[str]:
        """
        Extract clean text from raw HTML content.

        Args:
            html_content (str): Raw HTML content
            min_length (int): Minimum length of extracted text to consider valid

        Returns:
            Optional[str]: Extracted and cleaned text, or None if extraction failed
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove excluded tags
            for tag in self.EXCLUDED_TAGS:
                for element in soup.find_all(tag):
                    element.decompose()

            # Extract text
            text = ' '.join(soup.stripped_strings)

            # Clean and validate
            text = self.clean_text(text)
            if len(text) >= min_length:
                return text

            self.logger.debug(f"Extracted text too short ({len(text)} chars), returning None")
            return None

        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            return None

    def extract_from_playwright_sync(self, page: SyncPage, min_length: int = 10) -> Optional[str]:
        """
        Extract text from a Playwright page (synchronous).

        Args:
            page (Page): Playwright page instance
            min_length (int): Minimum length of extracted text to consider valid

        Returns:
            Optional[str]: Extracted and cleaned text, or None if extraction failed
        """
        try:
            html_content = page.content()
            return self.extract_from_html(html_content, min_length)
        except Exception as e:
            self.logger.error(f"Failed to extract text from Playwright page: {e}")
            return None

    async def extract_from_playwright_async(self, page: AsyncPage, min_length: int = 10) -> Optional[str]:
        """
        Extract text from a Playwright page (asynchronous).

        Args:
            page (AsyncPage): Async Playwright page instance
            min_length (int): Minimum length of extracted text to consider valid

        Returns:
            Optional[str]: Extracted and cleaned text, or None if extraction failed
        """
        try:
            html_content = await page.content()
            return self.extract_from_html(html_content, min_length)
        except Exception as e:
            self.logger.error(f"Failed to extract text from async Playwright page: {e}")
            return None

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Removes excessive whitespace, normalizes newlines, and performs
        general text cleanup.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text
        """
        # Normalize newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')

        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def extract_links_from_html(self, html_content: str, base_url: Optional[str] = None) -> Set[str]:
        """
        Extract all links from HTML content.

        Args:
            html_content (str): Raw HTML content
            base_url (str, optional): Base URL for relative link resolution

        Returns:
            Set[str]: Set of absolute URLs extracted from the page
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()

            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href'].strip()
                if href:
                    # Handle relative URLs if base_url provided
                    if base_url and not href.startswith(('http://', 'https://', 'mailto:', 'javascript:')):
                        from urllib.parse import urljoin
                        href = urljoin(base_url, href)

                    # Only include http(s) links
                    if href.startswith(('http://', 'https://')):
                        links.add(href)

            return links

        except Exception as e:
            self.logger.error(f"Failed to extract links from HTML: {e}")
            return set()

    def extract_links_from_playwright_sync(self, page: SyncPage, base_url: Optional[str] = None) -> Set[str]:
        """
        Extract all links from a Playwright page (synchronous).

        Args:
            page (Page): Playwright page instance
            base_url (str, optional): Base URL for relative link resolution

        Returns:
            Set[str]: Set of absolute URLs extracted from the page
        """
        try:
            html_content = page.content()
            return self.extract_links_from_html(html_content, base_url or page.url)
        except Exception as e:
            self.logger.error(f"Failed to extract links from Playwright page: {e}")
            return set()

    async def extract_links_from_playwright_async(self, page: AsyncPage, base_url: Optional[str] = None) -> Set[str]:
        """
        Extract all links from a Playwright page (asynchronous).

        Args:
            page (AsyncPage): Async Playwright page instance
            base_url (str, optional): Base URL for relative link resolution

        Returns:
            Set[str]: Set of absolute URLs extracted from the page
        """
        try:
            html_content = await page.content()
            return self.extract_links_from_html(html_content, base_url or page.url)
        except Exception as e:
            self.logger.error(f"Failed to extract links from async Playwright page: {e}")
            return set()

    def extract_text_from_elements(self, html_content: str, selector: str) -> Optional[str]:
        """
        Extract text from specific HTML elements using a CSS selector.

        Args:
            html_content (str): Raw HTML content
            selector (str): CSS selector for target elements

        Returns:
            Optional[str]: Combined text from matching elements, or None
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            elements = soup.select(selector)

            if not elements:
                return None

            text = ' '.join(el.get_text(strip=True) for el in elements)
            return self.clean_text(text) if text else None

        except Exception as e:
            self.logger.error(f"Failed to extract text from selector '{selector}': {e}")
            return None

    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
        """
        Split large text into manageable chunks with optional overlap.

        Useful for LLM processing where there are token limits.

        Args:
            text (str): Text to split
            chunk_size (int): Size of each chunk in characters
            overlap (int): Number of overlapping characters between chunks

        Returns:
            list: List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap

        return [chunk for chunk in chunks if chunk.strip()]
