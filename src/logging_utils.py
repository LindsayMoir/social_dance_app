"""
logging_utils.py

Utility functions for logging in the social dance app.
Provides helpers for logging extracted text and other common logging patterns.
"""

import logging


def log_extracted_text(function_name: str, url: str, extracted_text: str, logger: logging.Logger = None) -> None:
    """
    Log extracted text with first 100 and last 100 characters plus length.

    This prevents log files from being overwhelmed with thousands of characters
    of extracted text while still providing enough context to verify the extraction
    worked correctly.

    Args:
        function_name: Name of the function where extraction occurred (for debugging)
        url: The URL or identifier where the text was extracted from
        extracted_text: The full extracted text content
        logger: Logger instance to use. If None, uses root logger.

    Examples:
        >>> log_extracted_text("extract_event_text", "https://example.com", "Short text", logger)
        INFO - extract_event_text: Extracted 10 chars from https://example.com: Short text

        >>> log_extracted_text("scrape_page", "https://example.com", "A"*500, logger)
        INFO - scrape_page: Extracted 500 chars from https://example.com: AAAA...(first 100)...AAAA...(last 100)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not extracted_text:
        logger.warning(f"{function_name}: No text extracted from {url}")
        return

    text_len = len(extracted_text)

    if text_len <= 200:
        # If text is short, just show it all
        logger.info(f"{function_name}: Extracted {text_len} chars from {url}: {extracted_text}")
    else:
        # Show first 100, ellipsis, last 100, and total length
        first_100 = extracted_text[:100].replace('\n', ' ').replace('\r', ' ')
        last_100 = extracted_text[-100:].replace('\n', ' ').replace('\r', ' ')
        logger.info(f"{function_name}: Extracted {text_len:,} chars from {url}: {first_100}......{last_100}")


def log_extracted_text_summary(function_name: str, url: str, extracted_text: str, logger: logging.Logger = None) -> None:
    """
    Log a brief summary of extracted text (just the length).

    Use this when you don't need to see the content at all, just confirmation
    that text was extracted.

    Args:
        function_name: Name of the function where extraction occurred (for debugging)
        url: The URL or identifier where the text was extracted from
        extracted_text: The full extracted text content
        logger: Logger instance to use. If None, uses root logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not extracted_text:
        logger.warning(f"{function_name}: No text extracted from {url}")
        return

    text_len = len(extracted_text)
    logger.info(f"{function_name}: Extracted {text_len:,} chars from {url}")
