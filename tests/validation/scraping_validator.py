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
        # Softer thresholds for whitelisted URLs
        self.whitelist_days_back = self.scraping_config.get('whitelist_days_back', max(self.days_back, 60))
        self.consecutive_failures_threshold = self.scraping_config.get('consecutive_failures_threshold', 2)
        self.whitelist_consecutive_failures_threshold = self.scraping_config.get('whitelist_consecutive_failures_threshold', 3)
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

            logging.info("Querying high-performing URLs from database via DatabaseHandler...")
            rows = self.db_handler.execute_query(query)
            # Convert to DataFrame with explicit columns if any rows returned
            if rows:
                high_performers = pd.DataFrame(
                    rows,
                    columns=[
                        'link',
                        'source',
                        'keywords',
                        'hit_ratio',
                        'total_attempts',
                        'last_crawled'
                    ]
                )
            else:
                high_performers = pd.DataFrame(columns=['link','source','keywords','hit_ratio','total_attempts','last_crawled'])

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

        def _normalize_link_variants(u: str) -> list:
            try:
                from urllib.parse import urlsplit, urlunsplit
                parts = urlsplit(u)
                # strip query and fragment
                base = parts._replace(query='', fragment='')
                # remove trailing slash from path (except root)
                path = base.path[:-1] if base.path.endswith('/') and base.path != '/' else base.path
                base = base._replace(path=path)
                variants = set()
                # original
                variants.add(u)
                # no query/fragment
                variants.add(urlunsplit(base))
                # ensure trailing slash variant
                if not path.endswith('/') and path:
                    variants.add(urlunsplit(base._replace(path=path + '/')))
                # protocol swap http<->https
                if base.scheme == 'https':
                    variants.add(urlunsplit(base._replace(scheme='http')))
                elif base.scheme == 'http':
                    variants.add(urlunsplit(base._replace(scheme='https')))
                return list({v.rstrip('/') for v in variants})  # de-dupe minor variants
            except Exception:
                return [u]

        for idx, url_row in important_urls_df.iterrows():
            url = url_row['url']
            url_variants = _normalize_link_variants(url)

            try:
                # Choose window and thresholds based on importance
                if url_row['importance_type'] == 'whitelist':
                    window_days = int(self.whitelist_days_back)
                    fail_thresh = int(self.whitelist_consecutive_failures_threshold)
                else:
                    window_days = int(self.days_back)
                    fail_thresh = int(self.consecutive_failures_threshold)

                # Build dynamic SQL with variants
                placeholders = []
                params = {"days": window_days}
                where_or = []
                for i, v in enumerate(url_variants[:8]):
                    key = f"l{i}"
                    placeholders.append(key)
                    params[key] = v
                    where_or.append(f"link = :{key}")

                where_clause = " OR ".join(where_or) if where_or else "link = :l0"
                query = f"""
                    SELECT link, source, relevant, crawl_try, time_stamp, keywords
                    FROM urls
                    WHERE ({where_clause})
                      AND time_stamp >= NOW() - (:days * INTERVAL '1 day')
                    ORDER BY time_stamp DESC
                    LIMIT 5
                """

                result = self.db_handler.execute_query(query, params)

                if not result:
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'not_attempted',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': 0,
                        'keywords_found': None,
                        'recommendation': f'URL not in recent scraping run (last {window_days} days) - verify URL lists and normalization'
                    })
                    continue

                # Evaluate recent attempts
                relevants = [row[2] for row in result]
                crawl_try_vals = [row[3] for row in result]
                keywords_recent = result[0][5]

                any_success = any(bool(v) for v in relevants)
                # Count consecutive irrelevants from most recent backwards
                consecutive_irrelevant = 0
                for v in relevants:
                    if not v:
                        consecutive_irrelevant += 1
                    else:
                        break

                # Only flag as irrelevant if multiple consecutive recent attempts failed (threshold)
                if not any_success and consecutive_irrelevant >= fail_thresh:
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'marked_irrelevant',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': crawl_try_vals[0] if crawl_try_vals else None,
                        'keywords_found': keywords_recent,
                        'recommendation': 'Repeated recent attempts found page irrelevant - check content/keywords; whitelisted URLs use stricter criteria'
                    })
                    continue

                latest_crawl_try = crawl_try_vals[0] if crawl_try_vals else 0
                if latest_crawl_try >= 3 and not any_success:
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'multiple_retries',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': latest_crawl_try,
                        'keywords_found': keywords_recent,
                        'recommendation': 'High retry count with no recent success - check anti-scraping measures or timeouts'
                    })
                    continue

                # Single recent false but any success exists → treat as warning (log only)
                if not relevants[0] and any_success:
                    logging.warning(
                        f"Scraping warning (not failure): recent attempt irrelevant but successes exist in window for URL={url}"
                    )

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

    def check_source_distribution(self) -> dict:
        """
        Validate event source distribution - ensures major sources are present.

        Primary check: Confirms that critical event sources appear in the database.
        If any major source is missing, scraping likely failed for that source.

        Expected major sources (from successful pipeline run Jan 21, 2026):
        - Salsa Caliente
        - Victoria Summer Music
        - Victoria Latin Dance Association
        - WCS Lessons, Social Dances, and Conventions – BC Swing Dance
        - Red Hot Swing
        - Eventbrite
        - The Loft Pub Victoria

        Returns:
            dict: Distribution check results with status and warnings
        """
        logging.info("Checking event source distribution...")

        try:
            # Critical sources that MUST be present in database (presence check only)
            # These are major event providers - if missing, scraping failed
            REQUIRED_SOURCES = [
                'Salsa Caliente',
                'Victoria Summer Music',
                'Victoria Latin Dance Association',
                'WCS Lessons, Social Dances, and Conventions – BC Swing Dance',
                'Red Hot Swing',
                'Eventbrite',
                'The Loft Pub Victoria'
            ]

            EXPECTED_TOTAL_EVENTS = (1000, 2000)  # Reasonable range for total events

            # Query all sources with counts (not just top 10 - need to check for presence)
            query = """
                SELECT source, COUNT(*) AS counted
                FROM events
                GROUP BY source
                ORDER BY counted DESC
            """

            result = self.db_handler.execute_query(query)

            if not result:
                return {
                    'status': 'ERROR',
                    'message': 'Failed to query event source distribution',
                    'warnings': []
                }

            # Parse all results into dictionary for easy lookup
            all_sources = {}
            top_10_sources = []
            top_10_total = 0

            for i, row in enumerate(result):
                source = row[0]
                count = row[1]
                all_sources[source] = count

                # Keep top 10 for reporting
                if i < 10:
                    top_10_sources.append({'source': source, 'count': count})
                    top_10_total += count

            # Get total event count
            total_events = sum(all_sources.values())

            # Calculate percentage
            top_10_percentage = (top_10_total / total_events * 100) if total_events > 0 else 0

            # Validation checks
            warnings = []
            missing_sources = []
            status = 'PASS'

            # Check 1: Total event count (reasonable range check)
            if total_events < EXPECTED_TOTAL_EVENTS[0]:
                warnings.append(
                    f"Total event count ({total_events}) below expected minimum "
                    f"({EXPECTED_TOTAL_EVENTS[0]}) - possible scraping failures"
                )
                status = 'WARNING'
            elif total_events > EXPECTED_TOTAL_EVENTS[1]:
                warnings.append(
                    f"Total event count ({total_events}) above expected maximum "
                    f"({EXPECTED_TOTAL_EVENTS[1]}) - unusual activity or duplicates?"
                )
                status = 'WARNING'

            # Check 2: CRITICAL - Required sources must be present
            for required_source in REQUIRED_SOURCES:
                found = False

                # Check if any source in database contains the required source name
                for db_source in all_sources.keys():
                    if required_source in db_source or db_source in required_source:
                        found = True
                        event_count = all_sources[db_source]
                        logging.info(f"✓ Found source '{db_source}': {event_count} events")
                        break

                if not found:
                    missing_sources.append(required_source)
                    warnings.append(
                        f"CRITICAL: Required source '{required_source}' NOT FOUND in database - "
                        f"scraping likely failed for this source"
                    )
                    status = 'FAIL'  # Missing critical source = FAIL

            # Log results
            logging.info(f"Total events: {total_events}")
            logging.info(f"Total sources: {len(all_sources)}")
            logging.info(f"Top 10 sources: {top_10_total} events ({top_10_percentage:.1f}%)")

            if missing_sources:
                logging.error(f"❌ Missing {len(missing_sources)} required sources: {missing_sources}")
            elif warnings:
                for warning in warnings:
                    logging.warning(f"Source distribution check: {warning}")
            else:
                logging.info("✅ All required sources present in database")

            return {
                'status': status,
                'total_events': total_events,
                'total_sources': len(all_sources),
                'top_10_sources': top_10_sources,
                'top_10_total': top_10_total,
                'top_10_percentage': round(top_10_percentage, 1),
                'missing_sources': missing_sources,
                'warnings': warnings
            }

        except Exception as e:
            logging.error(f"Error checking source distribution: {e}")
            return {
                'status': 'ERROR',
                'message': f'Exception during source distribution check: {str(e)}',
                'warnings': []
            }

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
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'scraping_validation_report.json')

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Scraping validation report saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save scraping report: {e}")

        return report
