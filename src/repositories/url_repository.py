"""
URL repository for centralized URL management and validation.

This module consolidates all URL-related database operations previously
scattered throughout DatabaseHandler. It handles URL writing, blacklist
management, staleness checking, and normalization.
"""

from typing import Optional, Set
import logging
import pandas as pd
import re
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


class URLRepository:
    """
    Repository for managing URL operations in the database.

    Consolidates URL validation, writing, and staleness checking logic
    previously scattered across DatabaseHandler.

    Key responsibilities:
    - Blacklist loading and management
    - URL writing and normalization
    - URL staleness detection
    - URL processing decision logic
    """

    def __init__(self, db_handler):
        """
        Initialize URLRepository with database connection.

        Args:
            db_handler: DatabaseHandler instance for database operations
        """
        self.db = db_handler
        self.logger = logging.getLogger(__name__)
        self.blacklisted_domains: Set[str] = set()
        self.load_blacklist()

    def load_blacklist(self) -> None:
        """
        Loads a set of blacklisted domains from a CSV file specified in the configuration.

        The CSV file path is retrieved from config['constants']['black_list_domains'].
        The CSV is expected to have a column named 'Domain'. All domain names are converted
        to lowercase and stripped of whitespace before being added to the blacklist set.

        Side Effects:
            - Sets self.blacklisted_domains with loaded domains
            - Logs the number of loaded blacklisted domains at INFO level
        """
        try:
            csv_path = self.db.config['constants']['black_list_domains']
            df = pd.read_csv(csv_path)
            self.blacklisted_domains = set(df['Domain'].str.lower().str.strip())
            self.logger.info(f"Loaded {len(self.blacklisted_domains)} blacklisted domains.")
        except Exception as e:
            self.logger.error(f"Failed to load blacklist domains: {e}")
            self.blacklisted_domains = set()

    def is_blacklisted(self, url: str) -> bool:
        """
        Check if the given URL contains any blacklisted domain.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL contains any domain from the blacklist, False otherwise.

        Note:
            The check is case-insensitive.
        """
        if not url:
            return False

        url_lower = url.lower()
        return any(domain in url_lower for domain in self.blacklisted_domains)

    def write_url_to_db(self, url_row: tuple) -> bool:
        """
        Appends a new URL activity record to the 'urls' table in the database.

        This method processes and normalizes the provided URL data, especially the 'keywords' field,
        ensuring it is stored as a clean, comma-separated string. The data is then inserted as a new row
        into the 'urls' table using pandas' DataFrame and SQL interface.

        Args:
            url_row (tuple): A tuple containing the following fields in order:
                - link (str): The URL to be logged.
                - parent_url (str): The parent URL from which this link was found.
                - source (str): The source or context of the URL.
                - keywords (str | list | tuple | set): Associated keywords, which can be a string or an iterable.
                - relevant (bool | int): Indicator of relevance.
                - crawl_try (int): Number of crawl attempts.
                - time_stamp (str | datetime): Timestamp of the activity.

        Returns:
            bool: True if write was successful, False otherwise.

        Side Effects:
            - Appends a new row to the 'urls' table in the connected database.
            - Logs success or failure of the operation.
        """
        try:
            # 1) Unpack
            link, parent_url, source, keywords, relevant, crawl_try, time_stamp = url_row

            # 2) Normalize keywords into a simple comma-separated string
            if not isinstance(keywords, str):
                if isinstance(keywords, (list, tuple, set)):
                    keywords = ','.join(map(str, keywords))
                else:
                    keywords = str(keywords)

            # 3) Strip out braces/brackets/quotes and trim each term
            cleaned = re.sub(r'[\{\}\[\]\"]', '', keywords)
            parts = [p.strip() for p in cleaned.split(',') if p.strip()]
            keywords = ', '.join(parts)

            # 4) Build a one-row DataFrame
            df = pd.DataFrame([{
                'link': link,
                'parent_url': parent_url,
                'source': source,
                'keywords': keywords,
                'relevant': relevant,
                'crawl_try': crawl_try,
                'time_stamp': time_stamp
            }])

            # 5) Append to the table
            df.to_sql('urls', con=self.db.conn, if_exists='append', index=False)
            self.logger.info(f"write_url_to_db(): appended URL '{link}'")
            return True

        except Exception as e:
            self.logger.error(f"write_url_to_db(): failed to append URL '{url_row[0] if url_row else 'unknown'}': {e}")
            return False

    def stale_date(self, url: str) -> bool:
        """
        Determines whether the most recent event associated with the given URL is considered "stale"
        based on a configurable age threshold.

        Args:
            url (str): The URL whose most recent event's staleness is to be checked.

        Returns:
            bool:
                - True if there are no events for the URL, the most recent event's start date
                  is older than the configured threshold, or if an error occurs during the check.
                - False if the most recent event's start date is within the allowed threshold.

        Note:
            In case of any error, the method defaults to returning True (safer to re-process).
        """
        try:
            # 1. Fetch the most recent start_date for this URL
            query = """
                SELECT start_date
                FROM events_history
                WHERE url = :url
                ORDER BY start_date DESC
                LIMIT 1;
            """
            params = {'url': url}
            result = self.db.execute_query(query, params)

            # 2. If no rows returned, nothing has been recorded for this URL ⇒ treat as "stale"
            if not result:
                return True

            latest_start_date = result[0][0]

            # 3. If start_date is NULL in the DB, treat as "stale"
            if latest_start_date is None:
                return True

            # 3a. Convert whatever was returned into a Python date
            latest_date = pd.to_datetime(latest_start_date).date()

            # 4. Compute cutoff_date = today – N days
            days_threshold = int(self.db.config['clean_up']['old_events'])
            cutoff_date = datetime.now().date() - timedelta(days=days_threshold)

            # 4a. If the event's date is older than cutoff_date, it's stale → return True
            return latest_date < cutoff_date

        except Exception as e:
            self.logger.error(f"stale_date: Error checking stale date for url {url}: {e}")
            # In case of any error, default to True (safer to re-process)
            return True

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URLs by removing dynamic cache parameters that don't affect the underlying content.

        This is particularly important for Instagram and Facebook CDN URLs that include
        dynamic parameters like _nc_gid, _nc_ohc, oh, oe, etc. that change between requests
        but point to the same underlying image.

        Args:
            url (str): The original URL with potentially dynamic parameters

        Returns:
            str: Normalized URL with dynamic parameters removed
        """
        if not url:
            return url

        parsed = urlparse(url)

        # Check if this is an Instagram or Facebook CDN URL
        instagram_domains = {
            'instagram.com',
            'www.instagram.com',
            'scontent.cdninstagram.com',
            'instagram.fcxh2-1.fna.fbcdn.net',
        }

        fb_cdn_domains = {
            domain for domain in [parsed.netloc]
            if 'fbcdn.net' in domain and ('instagram' in domain or 'scontent' in domain)
        }

        is_instagram_cdn = (parsed.netloc in instagram_domains or
                           any(domain in parsed.netloc for domain in instagram_domains) or
                           bool(fb_cdn_domains))

        if not is_instagram_cdn:
            return url

        # Parse query parameters
        query_params = parse_qs(parsed.query)

        # List of dynamic parameters to remove for Instagram/FB CDN URLs
        dynamic_params = {
            '_nc_gid',     # Cache group ID - changes between sessions
            '_nc_ohc',     # Cache hash - changes between requests
            '_nc_oc',      # Cache parameter - changes between requests
            'oh',          # Hash parameter - changes between requests
            'oe',          # Expiration parameter - changes over time
            '_nc_zt',      # Zoom/time parameter
            '_nc_ad',      # Ad parameter
            '_nc_cid',     # Cache ID
            'ccb',         # Cache control parameter (sometimes)
        }

        # Remove dynamic parameters
        filtered_params = {k: v for k, v in query_params.items()
                          if k not in dynamic_params}

        # Reconstruct URL with filtered parameters
        new_query = urlencode(filtered_params, doseq=True)
        normalized_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))

        return normalized_url

    def should_process_url(self, url: str, urls_df=None, urls_gb=None) -> bool:
        """
        Determines whether a given URL should be processed based on its history in the database.

        The decision is made according to the following rules:
        0. If the URL is in the whitelist (data/urls/aaa_urls.csv), it should ALWAYS be processed.
        1. If the URL has never been seen before (i.e., no records in urls_df), it should be processed.
        2. If the most recent record for the URL has 'relevant' set to True, it should be processed.
        3. If the most recent record for the URL has 'relevant' set to False, the method checks the grouped statistics in urls_gb:
            - If the `hit_ratio` for the URL is greater than 0.1, or
            - If the number of crawl attempts (`crawl_try`) is less than or equal to 3,
            then the URL should be processed.
        4. If none of the above conditions are met, the URL should not be processed.

        Args:
            url (str): The URL to evaluate for processing.
            urls_df (DataFrame, optional): DataFrame of URL history. If None or empty, always process.
            urls_gb (DataFrame, optional): Grouped statistics for URL usefulness (hit_ratio, crawl_try).

        Returns:
            bool: True if the URL should be processed according to the criteria above, False otherwise.
        """
        import os

        if not url:
            return False

        # If urls_df is None or empty (e.g., on production), always process
        if urls_df is None or urls_df.empty:
            self.logger.info(f"should_process_url: URLs table not loaded (production mode), processing URL.")
            return True

        # Normalize URL to handle Instagram/FB CDN dynamic parameters
        normalized_url = self.normalize_url(url)

        # Log normalization if URL changed
        if normalized_url != url:
            self.logger.info(f"should_process_url: Normalized Instagram URL for comparison")

        # 0. Check if URL is in whitelist (always process) - URLs in data/urls/aaa_urls.csv should always be processed
        try:
            aaa_urls_path = os.path.join(self.db.config['input']['urls'], 'aaa_urls.csv')
            if os.path.exists(aaa_urls_path):
                aaa_urls_df = pd.read_csv(aaa_urls_path)
                if 'link' in aaa_urls_df.columns:
                    # Check if normalized_url matches any whitelist URL
                    if normalized_url in aaa_urls_df['link'].values:
                        self.logger.info(f"should_process_url: URL {normalized_url[:100]}... is in whitelist (aaa_urls.csv), processing it.")
                        return True
        except Exception as e:
            self.logger.warning(f"should_process_url: Could not check whitelist: {e}")

        # 1. Filter all rows for this normalized URL
        df_url = urls_df[urls_df['link'] == normalized_url]
        # If we've never recorded this normalized URL, process it
        if df_url.empty:
            self.logger.info(f"should_process_url: URL {normalized_url[:100]}... has never been seen before, processing it.")
            return True

        # 2. Look at the most recent "relevant" value
        last_relevant = df_url.iloc[-1]['relevant']
        if last_relevant and self.stale_date(normalized_url):
            self.logger.info(f"should_process_url: URL {normalized_url[:100]}... was last seen as relevant, processing it.")
            return True

        # 3. Last was False → check hit_ratio in urls_gb
        if urls_gb is not None and not urls_gb.empty:
            hit_row = urls_gb[urls_gb['link'] == normalized_url]

            if not hit_row.empty:
                # Extract scalars from the grouped DataFrame
                hit_ratio = hit_row.iloc[0]['hit_ratio']
                crawl_trys = hit_row.iloc[0]['crawl_try']

                if hit_ratio > 0.1 or crawl_trys <= 3:
                    self.logger.info(
                        "should_process_url: URL %s was last seen as not relevant "
                        "but hit_ratio (%.2f) > 0.1 or crawl_try (%d) ≤ 3, processing it.",
                        normalized_url[:100] + "...", hit_ratio, crawl_trys
                    )
                    return True

        # 4. Otherwise, do not process this URL
        self.logger.info(f"should_process_url: URL {normalized_url[:100]}... does not meet criteria for processing, skipping it.")
        return False
