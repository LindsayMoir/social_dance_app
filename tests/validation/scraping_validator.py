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

from datetime import date, datetime
import json
import logging
import os
import re
import sys
from time import perf_counter
from typing import Any, Dict, List
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from output_paths import codex_review_path


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

    @staticmethod
    def _coerce_bool(value: object, default: bool = True) -> bool:
        """Interpret CSV-style truthy and falsy values safely."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if not text:
            return default
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _parse_optional_date(value: object) -> date | None:
        """Parse a CSV date value into a date, returning None when blank/invalid."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    @classmethod
    def _is_pdf_source_active(cls, row: pd.Series, today: date | None = None) -> tuple[bool, str]:
        """Determine whether a seasonal PDF source row is active for the current date."""
        today_value = today or datetime.now().date()
        if not cls._coerce_bool(row.get("enabled"), default=True):
            return False, "disabled"
        active_start = cls._parse_optional_date(row.get("active_start_date"))
        active_end = cls._parse_optional_date(row.get("active_end_date"))
        if active_start and today_value < active_start:
            return False, "before_active_start_date"
        if active_end and today_value > active_end:
            return False, "after_active_end_date"
        return True, "active"

    def _get_required_sources_for_distribution(self, today: date | None = None) -> List[str]:
        """
        Return the active required sources for source-distribution completeness checks.

        Seasonal PDF-backed sources are only required while their configured active
        window is open. This prevents known off-season sources from appearing under
        missing required resources.
        """
        required_sources = [
            'Salsa Caliente',
            'Victoria Summer Music',
            'Victoria Latin Dance Association',
            'WCS Lessons, Social Dances, and Conventions – BC Swing Dance',
            'Red Hot Swing',
            'Eventbrite',
            'The Loft Pub Victoria',
        ]

        pdf_sources_file = self.scraping_config.get('pdf_sources_file', 'data/other/pdfs.csv')
        if not pdf_sources_file or not os.path.exists(pdf_sources_file):
            return required_sources

        try:
            pdf_sources = pd.read_csv(pdf_sources_file, dtype=str).fillna("")
            seasonal_status = {}
            today_value = today or datetime.now().date()
            for _, row in pdf_sources.iterrows():
                source_name = str(row.get("source", "") or "").strip()
                if not source_name:
                    continue
                is_active, _reason = self._is_pdf_source_active(row, today=today_value)
                seasonal_status[source_name] = is_active

            filtered_sources = []
            for source_name in required_sources:
                if source_name in seasonal_status and not seasonal_status[source_name]:
                    logging.info(
                        "check_source_distribution: excluding inactive seasonal source from required set: %s",
                        source_name,
                    )
                    continue
                filtered_sources.append(source_name)
            return filtered_sources
        except Exception as exc:
            logging.warning("check_source_distribution: seasonal source filtering unavailable: %s", exc)
            return required_sources

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

    @staticmethod
    def _extract_logged_url(line: str, prefix: str, suffix: str) -> str:
        """Extract URL between prefix/suffix markers from a log line."""
        start = line.find(prefix)
        if start < 0:
            return ""
        start += len(prefix)
        end = line.find(suffix, start)
        if end < 0:
            return ""
        return line[start:end].strip()

    @staticmethod
    def _matches_logged_url(candidate_url: str, logged_url: str) -> bool:
        """Return true when a candidate URL matches a logged URL snippet."""
        candidate = (candidate_url or "").strip().lower()
        logged = (logged_url or "").strip().lower()
        if not candidate or not logged:
            return False
        if logged.endswith("..."):
            return candidate.startswith(logged[:-3])
        return candidate == logged

    def _summarize_not_attempted_reasons(self, failures_df: pd.DataFrame) -> Dict[str, object]:
        """
        Classify why not-attempted URLs were not processed in this run.

        Categories:
        - explicit_url_run_limit_skip
        - explicit_should_process_url_skip
        - unattributed_with_global_run_limit
        - other_or_unknown
        """
        empty_summary = {
            "total_not_attempted": 0,
            "global_url_run_limit_reached": False,
            "per_url_stage": {},
            "run_limit_whitelist_context": {
                "pending_scraper_owned_roots_max": 0,
                "fb_owned_roots_max": 0,
                "non_text_roots_max": 0,
            },
            "categories": {
                "explicit_url_run_limit_skip": 0,
                "explicit_should_process_url_skip": 0,
                "unattributed_with_global_run_limit": 0,
                "other_or_unknown": 0,
            },
        }
        if failures_df.empty or "failure_type" not in failures_df.columns:
            return empty_summary

        not_attempted = failures_df[failures_df["failure_type"] == "not_attempted"]
        if not_attempted.empty:
            return empty_summary

        urls = [str(url).strip() for url in not_attempted.get("url", pd.Series(dtype=str)).tolist() if str(url).strip()]
        if not urls:
            return empty_summary

        url_variants: Dict[str, List[str]] = {}
        for url in urls:
            variants = [v.lower() for v in self._normalize_link_variants(url)]
            url_variants[url] = list(dict.fromkeys(variants + [url.lower()]))

        run_limit_urls: set[str] = set()
        should_skip_urls: set[str] = set()
        global_run_limit_reached = False
        pending_scraper_owned_roots_max = 0
        fb_owned_roots_max = 0
        non_text_roots_max = 0

        for log_path in self._log_files:
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        lower_line = line.lower()
                        if "url run limit reached" in lower_line:
                            global_run_limit_reached = True
                            context_match = re.search(
                                r"url run limit reached with (\d+) scraper-owned whitelist roots still unattempted "
                                r"\(fb_owned=(\d+), non_text=(\d+)\)",
                                lower_line,
                            )
                            if context_match:
                                pending_scraper_owned_roots_max = max(
                                    pending_scraper_owned_roots_max, int(context_match.group(1))
                                )
                                fb_owned_roots_max = max(
                                    fb_owned_roots_max, int(context_match.group(2))
                                )
                                non_text_roots_max = max(
                                    non_text_roots_max, int(context_match.group(3))
                                )

                        if "skipping non-whitelist link:" in lower_line and "url run limit reached" in lower_line:
                            logged_url = self._extract_logged_url(
                                line=line,
                                prefix="skipping non-whitelist link:",
                                suffix="\n",
                            )
                            if logged_url:
                                for original_url, variants in url_variants.items():
                                    if any(self._matches_logged_url(variant, logged_url) for variant in variants):
                                        run_limit_urls.add(original_url)

                        if (
                            "should_process_url:" in lower_line
                            and "does not meet criteria for processing, skipping it." in lower_line
                        ):
                            logged_url = self._extract_logged_url(
                                line=line,
                                prefix="should_process_url: URL ",
                                suffix=" does not meet criteria for processing, skipping it.",
                            )
                            if logged_url:
                                for original_url, variants in url_variants.items():
                                    if any(self._matches_logged_url(variant, logged_url) for variant in variants):
                                        should_skip_urls.add(original_url)
            except Exception:
                continue

        categories = {
            "explicit_url_run_limit_skip": 0,
            "explicit_should_process_url_skip": 0,
            "unattributed_with_global_run_limit": 0,
            "other_or_unknown": 0,
        }
        per_url_stage: Dict[str, str] = {}
        for url in urls:
            if url in run_limit_urls:
                categories["explicit_url_run_limit_skip"] += 1
                per_url_stage[url] = "not_attempted_run_limit_skipped"
            elif url in should_skip_urls:
                categories["explicit_should_process_url_skip"] += 1
                per_url_stage[url] = "not_attempted_should_process_url_skipped"
            elif global_run_limit_reached:
                categories["unattributed_with_global_run_limit"] += 1
                per_url_stage[url] = "not_attempted_run_limit_skipped"
            else:
                categories["other_or_unknown"] += 1
                per_url_stage[url] = "not_attempted_other_or_unknown"

        return {
            "total_not_attempted": len(urls),
            "global_url_run_limit_reached": global_run_limit_reached,
            "per_url_stage": per_url_stage,
            "run_limit_whitelist_context": {
                "pending_scraper_owned_roots_max": pending_scraper_owned_roots_max,
                "fb_owned_roots_max": fb_owned_roots_max,
                "non_text_roots_max": non_text_roots_max,
            },
            "categories": categories,
        }

    def _ensure_source_distribution_history_tables(self) -> None:
        """Delegate source-distribution schema setup to DatabaseHandler."""
        ensure_method = getattr(self.db_handler, "ensure_source_distribution_history_tables", None)
        if callable(ensure_method):
            ensure_method()

    def _persist_source_counts_snapshot(
        self,
        run_id: str,
        run_ts: str,
        total_events: int,
        ordered_sources: List[Dict[str, Any]],
    ) -> None:
        """Persist this run's source counts for long-term trend monitoring."""
        insert_stmt = """
            INSERT INTO source_event_counts_history
                (run_id, run_ts, source, event_count, rank_in_run, total_events)
            VALUES
                (:run_id, :run_ts, :source, :event_count, :rank_in_run, :total_events)
            ON CONFLICT (run_id, source) DO UPDATE
            SET
                run_ts = EXCLUDED.run_ts,
                event_count = EXCLUDED.event_count,
                rank_in_run = EXCLUDED.rank_in_run,
                total_events = EXCLUDED.total_events
        """
        for rank, row in enumerate(ordered_sources, start=1):
            source = str(row.get('source', '') or '').strip()
            if not source:
                continue
            params = {
                'run_id': run_id,
                'run_ts': run_ts,
                'source': source,
                'event_count': int(row.get('count', 0) or 0),
                'rank_in_run': rank,
                'total_events': int(total_events),
            }
            self.db_handler.execute_query(insert_stmt, params)

    def _build_source_trend_summary(
        self,
        run_id: str,
        run_ts: str,
        all_sources: Dict[str, int],
        top_10_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare current run source counts against recent runs and flag anomalies.
        """
        trend_cfg = self.scraping_config.get('source_trend_monitoring', {})
        lookback_runs = int(trend_cfg.get('lookback_runs', 14) or 14)
        min_baseline_count = int(trend_cfg.get('min_baseline_count', 5) or 5)
        drop_threshold_pct = float(trend_cfg.get('drop_threshold_pct', 0.60) or 0.60)
        spike_threshold_pct = float(trend_cfg.get('spike_threshold_pct', 1.00) or 1.00)

        summary: Dict[str, Any] = {
            'enabled': True,
            'run_id': run_id,
            'run_ts': run_ts,
            'lookback_runs': lookback_runs,
            'min_baseline_count': min_baseline_count,
            'drop_threshold_pct': drop_threshold_pct,
            'spike_threshold_pct': spike_threshold_pct,
            'history_runs_used': 0,
            'baseline_top_10': [],
            'current_top_10': top_10_sources[:10],
            'alerts': [],
        }

        prev_runs_sql = """
            SELECT DISTINCT run_id, run_ts
            FROM source_event_counts_history
            WHERE run_id <> :run_id
            ORDER BY run_ts DESC
            LIMIT :limit_runs
        """
        prev_runs = self.db_handler.execute_query(
            prev_runs_sql, {'run_id': run_id, 'limit_runs': lookback_runs}
        ) or []
        if not prev_runs:
            return summary

        run_ids = [str(row[0]) for row in prev_runs if row and row[0]]
        summary['history_runs_used'] = len(run_ids)
        if not run_ids:
            return summary

        run_id_filters: List[str] = []
        params: Dict[str, Any] = {'min_baseline_count': min_baseline_count}
        for i, prev_run_id in enumerate(run_ids):
            key = f"rid_{i}"
            run_id_filters.append(f"run_id = :{key}")
            params[key] = prev_run_id
        where_clause = " OR ".join(run_id_filters) if run_id_filters else "1=0"
        baseline_sql = f"""
            SELECT source, AVG(event_count) AS avg_count
            FROM source_event_counts_history
            WHERE ({where_clause})
            GROUP BY source
            HAVING AVG(event_count) >= :min_baseline_count
            ORDER BY avg_count DESC
            LIMIT 10
        """
        baseline_rows = self.db_handler.execute_query(baseline_sql, params) or []

        baseline_top_10: List[Dict[str, Any]] = []
        baseline_map: Dict[str, float] = {}
        for row in baseline_rows:
            source = str(row[0] or '').strip()
            avg_count = float(row[1] or 0.0)
            if not source:
                continue
            baseline_top_10.append({'source': source, 'avg_count': round(avg_count, 2)})
            baseline_map[source] = avg_count
        summary['baseline_top_10'] = baseline_top_10
        if not baseline_map:
            return summary

        alert_insert_sql = """
            INSERT INTO source_distribution_alerts_history
                (run_id, run_ts, source, alert_type, severity, current_count, baseline_avg, pct_change, details_json)
            VALUES
                (:run_id, :run_ts, :source, :alert_type, :severity, :current_count, :baseline_avg, :pct_change, :details_json)
        """

        def _record_alert(
            source: str,
            alert_type: str,
            severity: str,
            current_count: int,
            baseline_avg: float,
            pct_change: float,
            details: Dict[str, Any],
        ) -> None:
            summary['alerts'].append({
                'source': source,
                'alert_type': alert_type,
                'severity': severity,
                'current_count': current_count,
                'baseline_avg': round(baseline_avg, 2),
                'pct_change': round(pct_change, 4),
                'details': details,
            })
            params = {
                'run_id': run_id,
                'run_ts': run_ts,
                'source': source,
                'alert_type': alert_type,
                'severity': severity,
                'current_count': current_count,
                'baseline_avg': baseline_avg,
                'pct_change': pct_change,
                'details_json': json.dumps(details),
            }
            self.db_handler.execute_query(alert_insert_sql, params)

        for source, baseline_avg in baseline_map.items():
            current_count = int(all_sources.get(source, 0) or 0)
            pct_change = (current_count - baseline_avg) / max(1.0, baseline_avg)
            if current_count == 0:
                _record_alert(
                    source=source,
                    alert_type='dropout',
                    severity='critical',
                    current_count=0,
                    baseline_avg=baseline_avg,
                    pct_change=-1.0,
                    details={
                        'message': 'Source was in baseline top-10 and is absent in current run',
                        'history_runs_used': summary['history_runs_used'],
                    },
                )
                continue
            if pct_change <= -abs(drop_threshold_pct):
                _record_alert(
                    source=source,
                    alert_type='drop',
                    severity='warning',
                    current_count=current_count,
                    baseline_avg=baseline_avg,
                    pct_change=pct_change,
                    details={
                        'message': 'Source count dropped significantly vs baseline',
                        'drop_threshold_pct': drop_threshold_pct,
                    },
                )
            elif pct_change >= abs(spike_threshold_pct):
                _record_alert(
                    source=source,
                    alert_type='spike',
                    severity='warning',
                    current_count=current_count,
                    baseline_avg=baseline_avg,
                    pct_change=pct_change,
                    details={
                        'message': 'Source count increased significantly vs baseline',
                        'spike_threshold_pct': spike_threshold_pct,
                    },
                )

        return summary

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
            pd.DataFrame: Failures with columns
            [url, source, failure_type, failure_stage, importance, recommendation]
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
                        'failure_stage': 'not_attempted_unattributed',
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
                        'failure_stage': 'attempted_no_keywords',
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
                        'failure_stage': 'attempted_extraction_or_llm_failure',
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
                    'failure_stage': 'attempted_extraction_or_llm_failure',
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
            overall_start = perf_counter()
            # Critical sources that MUST be present in database (presence check only)
            # These are major event providers - if missing, scraping failed
            REQUIRED_SOURCES = self._get_required_sources_for_distribution()

            EXPECTED_TOTAL_EVENTS = (1000, 2000)  # Reasonable range for total events

            # Query all sources with counts (not just top 10 - need to check for presence)
            query = """
                SELECT source, COUNT(*) AS counted
                FROM events
                GROUP BY source
                ORDER BY counted DESC
            """

            source_query_start = perf_counter()
            result = self.db_handler.execute_query(query, statement_timeout_ms=15000)
            logging.info(
                "check_source_distribution: source count query completed in %.2fs",
                perf_counter() - source_query_start,
            )

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
            ordered_sources = []

            for i, row in enumerate(result):
                source = row[0]
                count = row[1]
                all_sources[source] = count
                ordered_sources.append({'source': source, 'count': count})

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

            run_id = str(os.getenv("DS_RUN_ID", "") or datetime.now().strftime("%Y%m%d-%H%M%S"))
            run_ts = datetime.now().isoformat()
            trend_summary: Dict[str, Any] = {
                'enabled': False,
                'run_id': run_id,
                'run_ts': run_ts,
                'alerts': [],
            }

            # Persist and analyze source trends (non-fatal)
            try:
                ensure_start = perf_counter()
                self._ensure_source_distribution_history_tables()
                logging.info(
                    "check_source_distribution: ensure_source_distribution_history_tables completed in %.2fs",
                    perf_counter() - ensure_start,
                )

                snapshot_start = perf_counter()
                self._persist_source_counts_snapshot(run_id, run_ts, int(total_events), ordered_sources)
                logging.info(
                    "check_source_distribution: persist_source_counts_snapshot completed in %.2fs",
                    perf_counter() - snapshot_start,
                )

                trend_start = perf_counter()
                trend_summary = self._build_source_trend_summary(
                    run_id=run_id,
                    run_ts=run_ts,
                    all_sources=all_sources,
                    top_10_sources=top_10_sources,
                )
                logging.info(
                    "check_source_distribution: build_source_trend_summary completed in %.2fs",
                    perf_counter() - trend_start,
                )
                if trend_summary.get('alerts'):
                    status = 'WARNING' if status == 'PASS' else status
                    warnings.append(
                        f"Source trend alerts: {len(trend_summary.get('alerts', []))} anomaly/anomalies flagged."
                    )
            except Exception as trend_err:
                logging.warning(f"Source trend monitoring unavailable: {trend_err}")
                trend_summary = {
                    'enabled': False,
                    'run_id': run_id,
                    'run_ts': run_ts,
                    'alerts': [],
                    'error': str(trend_err),
                }

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

            logging.info(
                "check_source_distribution: completed in %.2fs",
                perf_counter() - overall_start,
            )

            return {
                'status': status,
                'total_events': total_events,
                'total_sources': len(all_sources),
                'top_10_sources': top_10_sources,
                'top_10_total': top_10_total,
                'top_10_percentage': round(top_10_percentage, 1),
                'missing_sources': missing_sources,
                'warnings': warnings,
                'trend_monitoring': trend_summary,
            }

        except Exception as e:
            logging.error(f"Error checking source distribution: {e}")
            return {
                'status': 'ERROR',
                'message': f'Exception during source distribution check: {str(e)}',
                'warnings': []
            }

    def generate_report(
        self,
        failures_df: pd.DataFrame,
        important_urls_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Generate JSON report with scraping validation results.

        Args:
            failures_df (pd.DataFrame): DataFrame from check_scraping_failures()
            important_urls_df (pd.DataFrame | None): DataFrame from classify_important_urls()

        Returns:
            dict: Report dictionary with summary, critical_failures, and performance_degradation
        """
        # Initialize report structure
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_failures': 0,
                'total_failures_raw': 0,
                'total_important_urls': 0,
                'attempted_url_denominator': 0,
                'attempted_failure_rate': None,
                'post_scrape_failures': 0,
                'keyword_failures_after_scrape': 0,
                'pre_scrape_skipped_failures_excluded': 0,
                'whitelist_failures': 0,
                'edge_case_failures': 0,
                'high_performer_failures': 0,
                'failure_types': {},
                'failure_stage_counts': {
                    'run_limit_skipped': 0,
                    'should_process_url_skipped': 0,
                    'attempted_no_keywords': 0,
                    'attempted_extraction_or_llm_failure': 0,
                    'other_or_unknown': 0,
                },
                'not_attempted_reason_breakdown': {
                    'total_not_attempted': 0,
                    'global_url_run_limit_reached': False,
                    'categories': {}
                }
            },
            'critical_failures': [],
            'performance_degradation': []
        }

        total_important_urls = int(len(important_urls_df)) if important_urls_df is not None else 0
        report['summary']['total_important_urls'] = total_important_urls

        if failures_df.empty:
            logging.info("No failures to report - all important URLs scraped successfully")
            if total_important_urls > 0:
                report['summary']['attempted_url_denominator'] = total_important_urls
                report['summary']['attempted_failure_rate'] = 0.0
        else:
            failures_df = failures_df.copy()
            # Calculate summary statistics
            report['summary']['total_failures_raw'] = len(failures_df)
            report['summary']['whitelist_failures'] = len(failures_df[failures_df['importance'] == 'whitelist'])
            report['summary']['edge_case_failures'] = len(failures_df[failures_df['importance'] == 'edge_case'])
            report['summary']['high_performer_failures'] = len(failures_df[failures_df['importance'] == 'high_performer'])

            # Failure type counts
            if 'failure_type' in failures_df.columns:
                report['summary']['failure_types'] = failures_df['failure_type'].value_counts().to_dict()
                report['summary']['not_attempted_reason_breakdown'] = self._summarize_not_attempted_reasons(
                    failures_df
                )
                categories = (
                    report['summary']
                    .get('not_attempted_reason_breakdown', {})
                    .get('categories', {})
                )
                per_url_stage = (
                    report['summary']
                    .get('not_attempted_reason_breakdown', {})
                    .get('per_url_stage', {})
                )
                if per_url_stage and 'url' in failures_df.columns and 'failure_type' in failures_df.columns:
                    def _normalize_stage_for_row(row):
                        if str(row.get('failure_type', '')) != 'not_attempted':
                            return row.get('failure_stage')
                        return per_url_stage.get(
                            str(row.get('url', '') or ''),
                            row.get('failure_stage', 'not_attempted_unattributed')
                        )
                    failures_df['failure_stage'] = failures_df.apply(_normalize_stage_for_row, axis=1)

                pre_scrape_skips = int(categories.get('explicit_should_process_url_skip', 0) or 0)
                report['summary']['pre_scrape_skipped_failures_excluded'] = pre_scrape_skips
                not_attempted_count = int(report['summary']['failure_types'].get('not_attempted', 0) or 0)
                keyword_failures = int(report['summary']['failure_types'].get('marked_irrelevant', 0) or 0)
                post_scrape_failures = max(
                    0,
                    int(report['summary']['total_failures_raw']) - not_attempted_count,
                )
                report['summary']['post_scrape_failures'] = post_scrape_failures
                report['summary']['keyword_failures_after_scrape'] = keyword_failures
                report['summary']['total_failures'] = post_scrape_failures

                attempted_denominator = max(0, total_important_urls - not_attempted_count)
                report['summary']['attempted_url_denominator'] = attempted_denominator
                if attempted_denominator > 0:
                    report['summary']['attempted_failure_rate'] = round(
                        post_scrape_failures / attempted_denominator, 4
                    )
                else:
                    report['summary']['attempted_failure_rate'] = None

                stage_counts = failures_df['failure_stage'].value_counts().to_dict() if 'failure_stage' in failures_df.columns else {}
                run_limit_skipped = int(categories.get('explicit_url_run_limit_skip', 0) or 0) + int(
                    categories.get('unattributed_with_global_run_limit', 0) or 0
                )
                report['summary']['failure_stage_counts'] = {
                    'run_limit_skipped': run_limit_skipped,
                    'should_process_url_skipped': int(categories.get('explicit_should_process_url_skip', 0) or 0),
                    'attempted_no_keywords': int(stage_counts.get('attempted_no_keywords', 0) or 0),
                    'attempted_extraction_or_llm_failure': int(stage_counts.get('attempted_extraction_or_llm_failure', 0) or 0),
                    'other_or_unknown': int(stage_counts.get('not_attempted_other_or_unknown', 0) or 0),
                }
            else:
                report['summary']['total_failures'] = int(report['summary']['total_failures_raw'])

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

        output_path = codex_review_path('scraping_validation_report.json')

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logging.info(f"Scraping validation report saved: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save scraping report: {e}")

        return report
