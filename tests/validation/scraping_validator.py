"""
Scraping Validation Module

Identifies scraping failures for important URLs before production deployment.

This module checks three sources of important URLs:
1. Whitelist (aaa_urls.csv) - Always important
2. Edge cases (edge_cases.csv) - Manually curated important URLs
3. High performers - URLs with >50% historical hit ratio

Author: Claude Code
Version: 1.0.0
"""

from datetime import datetime
import json
import logging
import os
from typing import Dict, List, Optional
import pandas as pd


class ScrapingValidator:
    """
    Validates scraping success for important URLs.

    Identifies URLs from whitelist, edge cases, and high performers,
    then checks for three types of failures:
    - not_attempted: URL not scraped recently
    - marked_irrelevant: Previously successful URL now irrelevant
    - multiple_retries: Persistent scraping issues (crawl_try >= 3)

    Attributes:
        db_handler: DatabaseHandler instance for database operations
        config (dict): Configuration dictionary from config.yaml
    """

    def __init__(self, db_handler, config: dict):
        """
        Initialize the ScrapingValidator.

        Args:
            db_handler: DatabaseHandler instance
            config (dict): Configuration dictionary with validation settings
        """
        self.db_handler = db_handler
        self.config = config
        self.validation_config = config.get('testing', {}).get('validation', {})
        self.scraping_config = self.validation_config.get('scraping', {})

        # Load configuration with defaults
        self.days_back = self.scraping_config.get('days_back', 7)
        self.whitelist_file = self.scraping_config.get('whitelist_file', 'data/urls/aaa_urls.csv')
        self.edge_cases_file = self.scraping_config.get('edge_cases_file', 'data/other/edge_cases.csv')
        self.min_hit_ratio = self.scraping_config.get('min_hit_ratio', 0.50)
        self.min_attempts = self.scraping_config.get('min_attempts', 3)

    def classify_important_urls(self) -> pd.DataFrame:
        """
        Identify important URLs from three sources.

        Returns:
            pd.DataFrame: DataFrame with columns [url, source, importance_type, hit_ratio]
                         where importance_type is one of: whitelist, edge_case, high_performer
        """
        important_urls = []

        # 1. Load whitelist URLs
        if os.path.exists(self.whitelist_file):
            logging.info(f"Loading whitelist from: {self.whitelist_file}")
            whitelist_df = pd.read_csv(self.whitelist_file)

            if 'link' in whitelist_df.columns:
                for _, row in whitelist_df.iterrows():
                    important_urls.append({
                        'url': row['link'],
                        'source': row.get('source', 'Unknown'),
                        'importance_type': 'whitelist',
                        'hit_ratio': None
                    })
                logging.info(f"Loaded {len(whitelist_df)} whitelist URLs")
            else:
                logging.warning(f"Whitelist file missing 'link' column: {self.whitelist_file}")
        else:
            logging.warning(f"Whitelist file not found: {self.whitelist_file}")

        # 2. Load edge case URLs
        if os.path.exists(self.edge_cases_file):
            logging.info(f"Loading edge cases from: {self.edge_cases_file}")
            edge_cases_df = pd.read_csv(self.edge_cases_file)

            if 'link' in edge_cases_df.columns:
                for _, row in edge_cases_df.iterrows():
                    important_urls.append({
                        'url': row['link'],
                        'source': row.get('source', 'Unknown'),
                        'importance_type': 'edge_case',
                        'hit_ratio': None
                    })
                logging.info(f"Loaded {len(edge_cases_df)} edge case URLs")
            else:
                logging.warning(f"Edge cases file missing 'link' column: {self.edge_cases_file}")
        else:
            logging.warning(f"Edge cases file not found: {self.edge_cases_file}")

        # 3. Query high performers from database
        whitelist_urls = {url_dict['url'] for url_dict in important_urls}

        try:
            query = f"""
                SELECT
                    link,
                    source,
                    keywords,
                    CAST(SUM(CASE WHEN relevant = true THEN 1 ELSE 0 END) AS FLOAT) /
                        NULLIF(COUNT(*), 0) AS hit_ratio,
                    COUNT(*) as total_attempts,
                    MAX(time_stamp) as last_crawled
                FROM urls
                GROUP BY link, source, keywords
                HAVING COUNT(*) >= {self.min_attempts}
                   AND CAST(SUM(CASE WHEN relevant = true THEN 1 ELSE 0 END) AS FLOAT) /
                       NULLIF(COUNT(*), 0) > {self.min_hit_ratio}
                ORDER BY hit_ratio DESC
            """

            logging.info("Querying high-performing URLs from database...")
            high_performers = pd.read_sql_query(query, self.db_handler.conn)

            # Add high performers that aren't already in whitelist or edge cases
            high_performer_count = 0
            for _, row in high_performers.iterrows():
                if row['link'] not in whitelist_urls:
                    important_urls.append({
                        'url': row['link'],
                        'source': row['source'],
                        'importance_type': 'high_performer',
                        'hit_ratio': row['hit_ratio']
                    })
                    high_performer_count += 1

            logging.info(f"Found {high_performer_count} high-performing URLs")

        except Exception as e:
            logging.error(f"Failed to query high performers: {e}")

        # Convert to DataFrame
        if important_urls:
            result_df = pd.DataFrame(important_urls)
            logging.info(f"Total important URLs identified: {len(result_df)}")
            return result_df
        else:
            logging.warning("No important URLs found")
            return pd.DataFrame(columns=['url', 'source', 'importance_type', 'hit_ratio'])

    def check_scraping_failures(self, important_urls_df: pd.DataFrame) -> pd.DataFrame:
        """
        Check which important URLs failed in the most recent pipeline run.

        Detects three failure types:
        1. not_attempted: URL wasn't scraped at all in last N days
        2. marked_irrelevant: Previously good URL now returning no events
        3. multiple_retries: crawl_try >= 3 (indicates persistent issues)

        Args:
            important_urls_df (pd.DataFrame): DataFrame from classify_important_urls()

        Returns:
            pd.DataFrame: Failures with columns [url, source, failure_type, importance, recommendation]
        """
        if important_urls_df.empty:
            logging.warning("No important URLs to check")
            return pd.DataFrame()

        failures = []

        for idx, url_row in important_urls_df.iterrows():
            url = url_row['url']

            try:
                # Get most recent scraping attempt
                query = """
                    SELECT link, source, relevant, crawl_try, time_stamp, keywords
                    FROM urls
                    WHERE link = :url
                      AND time_stamp >= NOW() - INTERVAL :days
                    ORDER BY time_stamp DESC
                    LIMIT 1
                """

                # Use DatabaseHandler's execute_query method (handles connection properly)
                result = self.db_handler.execute_query(
                    query,
                    {"url": url, "days": f"'{self.days_back} days'"}
                )

                if not result or len(result) == 0:
                    # Failure type: URL not attempted
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'not_attempted',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': 0,
                        'keywords_found': None,
                        'recommendation': f'URL not in recent scraping run (last {self.days_back} days) - verify URL lists are loaded'
                    })

                else:
                    # Result is a list of Row objects from execute_query
                    recent = result[0]

                    # Access columns by index: [link, source, relevant, crawl_try, time_stamp, keywords]
                    link = recent[0]
                    source = recent[1]
                    relevant = recent[2]
                    crawl_try = recent[3]
                    time_stamp = recent[4]
                    keywords = recent[5]

                    # Failure type: Marked irrelevant
                    if not relevant:
                        failures.append({
                            'url': url,
                            'source': url_row['source'],
                            'failure_type': 'marked_irrelevant',
                            'importance': url_row['importance_type'],
                            'crawl_attempts': crawl_try,
                            'keywords_found': keywords,
                            'recommendation': 'Previously successful URL now irrelevant - check for page structure changes or keyword updates'
                        })

                    # Failure type: Multiple retries (persistent issues)
                    elif crawl_try >= 3:
                        failures.append({
                            'url': url,
                            'source': url_row['source'],
                            'failure_type': 'multiple_retries',
                            'importance': url_row['importance_type'],
                            'crawl_attempts': crawl_try,
                            'keywords_found': keywords,
                            'recommendation': 'High retry count indicates persistent issues - check for anti-scraping measures or timeouts'
                        })

            except Exception as e:
                logging.error(f"Error checking URL {url}: {e}")
                failures.append({
                    'url': url,
                    'source': url_row.get('source', 'Unknown'),
                    'failure_type': 'error',
                    'importance': url_row['importance_type'],
                    'crawl_attempts': None,
                    'keywords_found': None,
                    'recommendation': f'Error checking URL: {str(e)}'
                })

        if failures:
            result_df = pd.DataFrame(failures)
            logging.info(f"Found {len(result_df)} scraping failures")
            return result_df
        else:
            logging.info("No scraping failures detected")
            return pd.DataFrame()

    def generate_report(self, failures_df: pd.DataFrame) -> dict:
        """
        Generate JSON report with scraping validation results.

        Args:
            failures_df (pd.DataFrame): DataFrame from check_scraping_failures()

        Returns:
            dict: Report dictionary with summary, critical_failures, and performance_degradation
        """
        # Initialize report structure
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_failures': 0,
                'whitelist_failures': 0,
                'edge_case_failures': 0,
                'high_performer_failures': 0,
                'failure_types': {}
            },
            'critical_failures': [],
            'performance_degradation': []
        }

        if failures_df.empty:
            logging.info("No failures to report - all important URLs scraped successfully")
        else:
            # Calculate summary statistics
            report['summary']['total_failures'] = len(failures_df)
            report['summary']['whitelist_failures'] = len(failures_df[failures_df['importance'] == 'whitelist'])
            report['summary']['edge_case_failures'] = len(failures_df[failures_df['importance'] == 'edge_case'])
            report['summary']['high_performer_failures'] = len(failures_df[failures_df['importance'] == 'high_performer'])

            # Failure type counts
            if 'failure_type' in failures_df.columns:
                report['summary']['failure_types'] = failures_df['failure_type'].value_counts().to_dict()

            # Critical failures (whitelist + edge_case)
            critical_df = failures_df[failures_df['importance'].isin(['whitelist', 'edge_case'])]
            if not critical_df.empty:
                report['critical_failures'] = critical_df.to_dict('records')

            # Performance degradation (high_performer)
            degradation_df = failures_df[failures_df['importance'] == 'high_performer']
            if not degradation_df.empty:
                report['performance_degradation'] = degradation_df.to_dict('records')

        # Save report to file
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'tests/output')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'scraping_validation_report.json')

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Scraping validation report saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save scraping report: {e}")

        return report
