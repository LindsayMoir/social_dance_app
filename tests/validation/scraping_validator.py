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
from typing import Dict, List
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
        self.max_log_evidence_lines = int(self.scraping_config.get('max_log_evidence_lines', 3))
        self.log_evidence_window = int(self.scraping_config.get('log_evidence_window', 2))
        self._log_files = self._resolve_log_files()

    def _resolve_log_files(self) -> List[str]:
        """Resolve existing log files used for failure evidence."""
        configured_logs = self.config.get('logging', {})
        candidates = [
            configured_logs.get('scraper_log_file'),
            configured_logs.get('log_file'),
            'logs/fb_log.txt',
            'logs/rd_ext_log.txt',
            'logs/credential_validator_log.txt',
            'logs/ebs_log.txt',
            'logs/emails_log.txt',
        ]
        files: List[str] = []
        for path in candidates:
            if path and path not in files and os.path.exists(path):
                files.append(path)
        return files

    def _normalize_link_variants(self, u: str) -> List[str]:
        """Build URL variants for robust link matching."""
        try:
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(u)
            base = parts._replace(query='', fragment='')
            path = base.path[:-1] if base.path.endswith('/') and base.path != '/' else base.path
            base = base._replace(path=path)
            variants = set()
            variants.add(u)
            variants.add(urlunsplit(base))
            if not path.endswith('/') and path:
                variants.add(urlunsplit(base._replace(path=path + '/')))
            if base.scheme == 'https':
                variants.add(urlunsplit(base._replace(scheme='http')))
            elif base.scheme == 'http':
                variants.add(urlunsplit(base._replace(scheme='https')))

            expanded = set()
            for v in variants:
                expanded.add(v)
                p = urlsplit(v)
                if p.path and p.path != '/':
                    if p.path.endswith('/'):
                        expanded.add(urlunsplit(p._replace(path=p.path.rstrip('/'))))
                    else:
                        expanded.add(urlunsplit(p._replace(path=p.path + '/')))
            return list(expanded)
        except Exception:
            return [u]

    def _query_recent_url_rows(self, url_variants: List[str], window_days: int) -> List:
        params = {"days": int(window_days)}
        where_or = []
        for i, v in enumerate(url_variants[:8]):
            key = f"l{i}"
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
        return self.db_handler.execute_query(query, params)

    def _query_recent_child_success_count(self, url_variants: List[str], window_days: int) -> int:
        params = {"days": int(window_days)}
        where_or = []
        for i, v in enumerate(url_variants[:8]):
            key = f"p{i}"
            params[key] = v
            where_or.append(f"parent_url = :{key}")
        where_clause = " OR ".join(where_or) if where_or else "parent_url = :p0"
        query = f"""
            SELECT COUNT(*)
            FROM urls
            WHERE ({where_clause})
              AND relevant = true
              AND time_stamp >= NOW() - (:days * INTERVAL '1 day')
        """
        rows = self.db_handler.execute_query(query, params)
        return int(rows[0][0]) if rows else 0

    def _query_latest_overall_row(self, url_variants: List[str]):
        params = {}
        where_or = []
        for i, v in enumerate(url_variants[:8]):
            key = f"o{i}"
            params[key] = v
            where_or.append(f"link = :{key}")
        where_clause = " OR ".join(where_or) if where_or else "link = :o0"
        query = f"""
            SELECT link, relevant, crawl_try, time_stamp, keywords
            FROM urls
            WHERE ({where_clause})
            ORDER BY time_stamp DESC
            LIMIT 1
        """
        rows = self.db_handler.execute_query(query, params)
        return rows[0] if rows else None

    def _build_failure_details(
        self,
        failure_type: str,
        window_days: int,
        result_rows: List,
        child_success_count: int,
        latest_overall_row,
    ) -> Dict[str, object]:
        relevants = [bool(r[2]) for r in result_rows] if result_rows else []
        recent_true = int(sum(1 for v in relevants if v))
        recent_false = int(sum(1 for v in relevants if not v))
        consecutive_irrelevant = 0
        for v in relevants:
            if not v:
                consecutive_irrelevant += 1
            else:
                break

        reason_code = "unknown"
        probable_cause = "Unknown"
        if failure_type == "not_attempted":
            reason_code = "no_recent_url_rows"
            probable_cause = f"No matching urls rows in the last {window_days} days"
            if latest_overall_row is not None:
                reason_code = "not_seen_in_window_historical_exists"
                probable_cause = "Historically seen, but not recorded in current validation window"
        elif failure_type == "marked_irrelevant":
            reason_code = "recent_irrelevant_threshold_reached"
            probable_cause = "Recent attempts were irrelevant and crossed threshold"
            if child_success_count > 0:
                reason_code = "base_irrelevant_child_success_exists"
                probable_cause = "Base URL looked irrelevant but child URLs succeeded"
        elif failure_type == "multiple_retries":
            reason_code = "retry_threshold_with_no_success"
            probable_cause = "High retry count with no recent successful extraction"
        elif failure_type == "error":
            reason_code = "validator_exception"
            probable_cause = "Exception during validation"

        last_attempt_time = str(result_rows[0][4]) if result_rows and result_rows[0][4] else None
        last_relevant_time = None
        for row in result_rows:
            if bool(row[2]):
                last_relevant_time = str(row[4]) if row[4] else None
                break

        return {
            'reason_code': reason_code,
            'probable_cause': probable_cause,
            'validation_window_days': int(window_days),
            'recent_base_rows': int(len(result_rows)),
            'recent_base_relevant_true': recent_true,
            'recent_base_relevant_false': recent_false,
            'recent_consecutive_irrelevant': consecutive_irrelevant,
            'recent_relevant_child_count': int(child_success_count),
            'last_attempt_time': last_attempt_time,
            'last_relevant_time': last_relevant_time,
            'latest_seen_time_overall': str(latest_overall_row[3]) if latest_overall_row and latest_overall_row[3] else None,
            'latest_seen_relevant_overall': bool(latest_overall_row[1]) if latest_overall_row is not None else None,
            'latest_seen_crawl_try_overall': int(latest_overall_row[2]) if latest_overall_row is not None and latest_overall_row[2] is not None else None,
        }

    def _collect_log_evidence(self, url: str) -> List[str]:
        """Collect concise log evidence snippets for a URL."""
        if not url or not self._log_files:
            return []

        snippets: List[str] = []
        markers = (
            "Failed to process LLM response",
            "Both LLM providers failed",
            "Request timed out",
            "marked as irrelevant",
            "marked as relevant",
            "process_fb_url: no text",
            "No day of the week found before 'More About Discussion'",
            "extract_event_links(): Found",
            "write_url_to_db(): appended URL",
            "Skipping URL",
        )

        for log_path in self._log_files:
            if len(snippets) >= self.max_log_evidence_lines:
                break
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                for idx, line in enumerate(lines):
                    if url not in line:
                        continue
                    start = max(0, idx - self.log_evidence_window)
                    end = min(len(lines), idx + self.log_evidence_window + 1)
                    for j in range(start, end):
                        candidate = lines[j].rstrip('\n')
                        if url in candidate or any(m in candidate for m in markers):
                            snippet = f"{log_path}:{j+1}: {candidate}"
                            if snippet not in snippets:
                                snippets.append(snippet)
                                if len(snippets) >= self.max_log_evidence_lines:
                                    break
                    if len(snippets) >= self.max_log_evidence_lines:
                        break
            except Exception:
                continue
        return snippets

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

        for idx, url_row in important_urls_df.iterrows():
            url = url_row['url']
            url_variants = self._normalize_link_variants(url)

            try:
                # Choose window and thresholds based on importance
                if url_row['importance_type'] == 'whitelist':
                    window_days = int(self.whitelist_days_back)
                    fail_thresh = int(self.whitelist_consecutive_failures_threshold)
                else:
                    window_days = int(self.days_back)
                    fail_thresh = int(self.consecutive_failures_threshold)

                result = self._query_recent_url_rows(url_variants, window_days)
                child_success_count = self._query_recent_child_success_count(url_variants, window_days)
                latest_overall_row = self._query_latest_overall_row(url_variants)

                if not result:
                    if child_success_count > 0:
                        logging.info(
                            "Scraping validator: base URL %s has no direct recent rows but has %s recent relevant child URL rows.",
                            url,
                            child_success_count,
                        )
                        continue
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'not_attempted',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': 0,
                        'keywords_found': None,
                        'recommendation': f'URL not in recent scraping run (last {window_days} days) - verify URL lists and normalization',
                        **self._build_failure_details(
                            failure_type='not_attempted',
                            window_days=window_days,
                            result_rows=result,
                            child_success_count=child_success_count,
                            latest_overall_row=latest_overall_row,
                        ),
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
                    if child_success_count > 0:
                        logging.info(
                            "Scraping validator: suppressing marked_irrelevant for %s due to %s recent relevant child URL rows.",
                            url,
                            child_success_count,
                        )
                        continue
                    failures.append({
                        'url': url,
                        'source': url_row['source'],
                        'failure_type': 'marked_irrelevant',
                        'importance': url_row['importance_type'],
                        'crawl_attempts': crawl_try_vals[0] if crawl_try_vals else None,
                        'keywords_found': keywords_recent,
                        'recommendation': 'Repeated recent attempts found page irrelevant - check content/keywords; whitelisted URLs use stricter criteria',
                        **self._build_failure_details(
                            failure_type='marked_irrelevant',
                            window_days=window_days,
                            result_rows=result,
                            child_success_count=child_success_count,
                            latest_overall_row=latest_overall_row,
                        ),
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
                        'recommendation': 'High retry count with no recent success - check anti-scraping measures or timeouts',
                        **self._build_failure_details(
                            failure_type='multiple_retries',
                            window_days=window_days,
                            result_rows=result,
                            child_success_count=child_success_count,
                            latest_overall_row=latest_overall_row,
                        ),
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
                    'recommendation': f'Error checking URL: {str(e)}',
                    **self._build_failure_details(
                        failure_type='error',
                        window_days=self.whitelist_days_back if url_row['importance_type'] == 'whitelist' else self.days_back,
                        result_rows=[],
                        child_success_count=0,
                        latest_overall_row=None,
                    ),
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
                critical_records = critical_df.to_dict('records')
                for record in critical_records:
                    record['evidence'] = self._collect_log_evidence(record.get('url', ''))
                report['critical_failures'] = critical_records

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
