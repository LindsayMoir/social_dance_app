#!/usr/bin/env python3
"""
Pre-commit Validation Test Runner

Main orchestrator for validation testing framework.
Runs both scraping validation and chatbot batch testing.

This script can be run standalone or called from pipeline.py.
Exit codes:
- 0: Tests passed or completed with warnings
- 1: Tests failed (critical issues found)

Author: Claude Code
Version: 1.0.0
"""

import sys
import os
import csv
import json
from collections import Counter

# Add src to path for imports (calculate path relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, script_dir)  # Also add tests/validation for local imports

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('src/.env')  # Load from src/.env since that's where credentials are stored

from datetime import datetime, timedelta
import logging
import yaml
import random

# Import validation modules
from scraping_validator import ScrapingValidator
from chatbot_evaluator import (
    TestQuestionGenerator,
    ChatbotTestExecutor,
    ChatbotScorer,
    generate_chatbot_report
)

# Import existing utilities
from db import DatabaseHandler
from llm import LLMHandler
from logging_config import setup_logging
from email_notifier import send_report_email


class ValidationTestRunner:
    """
    Main orchestrator for validation testing.

    Runs both scraping validation and chatbot testing,
    generates reports, and returns overall status.

    Attributes:
        config (dict): Configuration from config.yaml
        db_handler: DatabaseHandler instance
        llm_handler: LLMHandler instance
    """

    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the ValidationTestRunner.

        Args:
            config_path (str): Path to config.yaml file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize database and LLM handlers
        self.db_handler = DatabaseHandler(self.config)
        self.llm_handler = LLMHandler(config_path=config_path)

        # Get validation configuration
        self.validation_config = self.config.get('testing', {}).get('validation', {})

        logging.info("ValidationTestRunner initialized")

    def run_all_validations(self) -> dict:
        """
        Run both scraping validation and chatbot testing.

        Returns:
            dict: Combined results with overall status
        """
        logging.info("=" * 80)
        logging.info("PRE-COMMIT VALIDATION TESTS")
        logging.info("=" * 80)

        results = {
            'timestamp': datetime.now().isoformat(),
            'scraping_validation': None,
            'chatbot_testing': None,
            'overall_status': 'PASS',
            'reliability_gates': None,
        }

        # 1. SCRAPING VALIDATION
        try:
            logging.info("\n" + "=" * 70)
            logging.info("SCRAPING VALIDATION")
            logging.info("=" * 70)

            validator = ScrapingValidator(self.db_handler, self.config)

            # Classify important URLs
            important_urls = validator.classify_important_urls()
            logging.info(f"Identified {len(important_urls)} important URLs")

            # Check for failures
            failures = validator.check_scraping_failures(important_urls)
            logging.info(f"Found {len(failures)} scraping failures")

            # Check source distribution
            distribution_check = validator.check_source_distribution()

            # Generate report
            scraping_report = validator.generate_report(failures)
            scraping_report['source_distribution'] = distribution_check  # Add to report
            results['scraping_validation'] = scraping_report

            # Check for critical failures
            whitelist_failures = scraping_report['summary'].get('whitelist_failures', 0)
            edge_case_failures = scraping_report['summary'].get('edge_case_failures', 0)

            if whitelist_failures > 0 or edge_case_failures > 0:
                results['overall_status'] = 'WARNING'
                logging.warning(
                    f"⚠️  Critical failures detected: {whitelist_failures} whitelist, "
                    f"{edge_case_failures} edge case URLs failed"
                )

            # Check source distribution status
            if distribution_check['status'] == 'FAIL':
                results['overall_status'] = 'FAIL'
                logging.error("❌ Source distribution check FAILED")
            elif distribution_check['status'] == 'WARNING':
                if results['overall_status'] == 'PASS':
                    results['overall_status'] = 'WARNING'
                logging.warning("⚠️  Source distribution check has warnings")

        except Exception as e:
            logging.error(f"Scraping validation failed: {e}", exc_info=True)
            results['scraping_validation'] = {'error': str(e)}
            results['overall_status'] = 'WARNING'  # Don't fail entire validation

        # 2. CHATBOT TESTING
        try:
            logging.info("\n" + "=" * 70)
            logging.info("CHATBOT BATCH TESTING")
            logging.info("=" * 70)

            # Generate questions
            chatbot_config = self.validation_config.get('chatbot', {})
            template_file = chatbot_config.get(
                'question_templates',
                'tests/data/chatbot_test_questions.yaml'
            )

            question_gen = TestQuestionGenerator(template_file)
            questions = question_gen.generate_all_questions()
            logging.info(f"Generated {len(questions)} test questions")

            # Optionally sample a random subset of questions based on config.crawling.random_test_limit
            try:
                sample_n = int(self.config.get('crawling', {}).get('random_test_limit', 0) or 0)
            except Exception:
                sample_n = 0

            if sample_n > 0 and len(questions) > sample_n:
                random.shuffle(questions)
                questions = questions[:sample_n]
                logging.info(f"Randomly selected {len(questions)} questions (random_test_limit={sample_n})")

            # Execute tests
            executor = ChatbotTestExecutor(self.config, self.db_handler)
            test_results = executor.execute_all_tests(questions)
            logging.info(f"Executed {len(test_results)} tests")

            # Score results
            scorer = ChatbotScorer(self.llm_handler)
            scored_results = scorer.score_all_results(test_results)
            logging.info("Scoring complete")

            # Generate report
            output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')
            chatbot_report = generate_chatbot_report(scored_results, output_dir)
            results['chatbot_testing'] = chatbot_report

            # Check thresholds
            avg_score = chatbot_report['summary']['average_score']
            exec_rate = chatbot_report['summary']['execution_success_rate']

            score_threshold = chatbot_config.get('score_threshold', 70)
            exec_threshold = chatbot_config.get('execution_threshold', 0.90)

            if avg_score < score_threshold:
                results['overall_status'] = 'FAIL'
                logging.error(
                    f"❌ Chatbot average score ({avg_score:.1f}) "
                    f"below threshold ({score_threshold})"
                )
            elif exec_rate < exec_threshold:
                if results['overall_status'] == 'PASS':
                    results['overall_status'] = 'WARNING'
                logging.warning(
                    f"⚠️  SQL execution rate ({exec_rate:.2%}) "
                    f"below threshold ({exec_threshold:.0%})"
                )
            else:
                logging.info(
                    f"✅ Chatbot tests passed: avg_score={avg_score:.1f}, "
                    f"exec_rate={exec_rate:.2%}"
                )

        except Exception as e:
            logging.error(f"Chatbot testing failed: {e}", exc_info=True)
            results['chatbot_testing'] = {'error': str(e)}
            results['overall_status'] = 'FAIL'

        # 3. RELIABILITY GATE EVALUATION (PHASE 2)
        try:
            alias_audit_summary = self._summarize_address_alias_audit()
            llm_activity_summary = self._summarize_llm_provider_activity(results.get('timestamp'))
            reliability_scorecard = self._summarize_reliability_scorecard(
                results, llm_activity_summary, alias_audit_summary
            )
            gate_eval = self._evaluate_reliability_gates(reliability_scorecard)
            results['reliability_gates'] = gate_eval

            if gate_eval.get('status') == 'FAIL':
                results['overall_status'] = 'FAIL'
                failed = gate_eval.get('failed_gates', [])
                logging.error(
                    "❌ Reliability gates failed (%d): %s",
                    len(failed),
                    ", ".join(g.get('name', 'unknown') for g in failed),
                )
            elif gate_eval.get('status') == 'WARNING' and results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'
        except Exception as e:
            logging.error(f"Reliability gate evaluation failed: {e}", exc_info=True)
            if results['overall_status'] == 'PASS':
                results['overall_status'] = 'WARNING'
            results['reliability_gates'] = {
                'enabled': True,
                'status': 'ERROR',
                'error': str(e),
                'failed_gates': [],
                'gate_results': [],
                'thresholds': {},
            }

        # 4. GENERATE COMPREHENSIVE HTML REPORT
        try:
            if self.validation_config.get('reporting', {}).get('generate_html', True):
                self._generate_html_report(results)
        except Exception as e:
            logging.error(f"HTML report generation failed: {e}", exc_info=True)

        # 5. SEND EMAIL NOTIFICATION (if configured)
        try:
            self._send_email_notification(results)
        except Exception as e:
            logging.error(f"Email notification failed: {e}", exc_info=True)
            # Don't fail the entire validation if email fails

        # Final status
        logging.info("\n" + "=" * 80)
        logging.info(f"VALIDATION COMPLETE - STATUS: {results['overall_status']}")
        logging.info("=" * 80)

        return results

    def _generate_html_report(self, results: dict) -> None:
        """
        Generate HTML report combining both validations.

        Args:
            results (dict): Combined validation results
        """
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')
        output_path = os.path.join(output_dir, 'comprehensive_test_report.html')
        alias_audit_summary = self._summarize_address_alias_audit()
        llm_activity_summary = self._summarize_llm_provider_activity(results.get('timestamp'))
        reliability_scorecard = self._summarize_reliability_scorecard(results, llm_activity_summary, alias_audit_summary)
        reliability_issues = self._extract_reliability_issues(results, llm_activity_summary, alias_audit_summary)
        reliability_gates = results.get('reliability_gates') or self._evaluate_reliability_gates(reliability_scorecard)
        optimization_plan = self._build_optimization_plan(results, reliability_scorecard, llm_activity_summary)

        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pre-Commit Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 30px; }}
        h3 {{ color: #777; margin-top: 20px; }}
        .status-pass {{ color: #28a745; font-weight: bold; font-size: 1.3em; }}
        .status-warning {{ color: #ffc107; font-weight: bold; font-size: 1.3em; }}
        .status-fail {{ color: #dc3545; font-weight: bold; font-size: 1.3em; }}
        .metric-container {{ display: flex; flex-wrap: wrap; margin: 20px 0; }}
        .metric {{ flex: 0 0 200px; margin: 10px 20px 10px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #007bff; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .problematic {{ background-color: #ffe6e6; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
        .error-box {{ background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pre-Commit Validation Report</h1>
        <p class="timestamp"><strong>Timestamp:</strong> {results['timestamp']}</p>
        <p><strong>Overall Status:</strong>
            <span class="status-{results['overall_status'].lower()}">{results['overall_status']}</span>
        </p>

        <h2>1. Reliability Scorecard</h2>
        {self._build_reliability_scorecard_html(reliability_scorecard, reliability_issues, reliability_gates)}

        <h2>2. Scraping Validation</h2>
        {self._build_scraping_html(results.get('scraping_validation'))}

        <h2>3. Chatbot Testing</h2>
        {self._build_chatbot_html(results.get('chatbot_testing'))}

        <h2>4. Address Alias Audit</h2>
        {self._build_address_alias_audit_html(alias_audit_summary)}

        <h2>5. LLM Provider Activity</h2>
        {self._build_llm_provider_activity_html(llm_activity_summary)}

        <h2>6. Optimization Recommendations</h2>
        {self._build_optimization_html(optimization_plan)}

        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #999; font-size: 0.9em;">
            Generated by Claude Code Validation Framework v1.0.0
        </p>
    </div>
</body>
</html>
"""

        # Write HTML file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        scorecard_path = os.path.join(output_dir, 'reliability_scorecard.json')
        issues_path = os.path.join(output_dir, 'reliability_issues.json')
        with open(scorecard_path, 'w', encoding='utf-8') as f:
            json.dump(reliability_scorecard, f, indent=2)
        with open(issues_path, 'w', encoding='utf-8') as f:
            json.dump({"issues": reliability_issues}, f, indent=2)
        gates_path = os.path.join(output_dir, 'reliability_gates.json')
        with open(gates_path, 'w', encoding='utf-8') as f:
            json.dump(reliability_gates, f, indent=2)
        optimization_path = os.path.join(output_dir, 'reliability_optimization.json')
        with open(optimization_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_plan, f, indent=2)

        logging.info(f"HTML report saved: {output_path}")
        logging.info(f"Reliability scorecard saved: {scorecard_path}")
        logging.info(f"Reliability issues saved: {issues_path}")
        logging.info(f"Reliability gates saved: {gates_path}")
        logging.info(f"Reliability optimization saved: {optimization_path}")

    @staticmethod
    def _escape_html(value: str) -> str:
        """Escape untrusted text for safe HTML rendering."""
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _summarize_address_alias_audit(self) -> dict:
        """
        Summarize address alias audit CSV for report display.
        Missing/invalid files are reported as non-fatal metadata.
        """
        output_cfg = self.config.get('output', {})
        csv_path = output_cfg.get('address_alias_audit', 'output/address_alias_hits.csv')
        days_window = int(
            self.validation_config.get('reporting', {}).get('address_alias_audit_days', 14) or 14
        )

        summary = {
            'available': False,
            'path': csv_path,
            'days_window': days_window,
            'rows_analyzed': 0,
            'decision_counts': {},
            'rule_counts': [],
            'rule_decision_counts': [],
            'daily_counts': [],
            'top_candidates': [],
            'error': '',
        }

        if not os.path.exists(csv_path):
            summary['error'] = f"Audit file not found: {csv_path}"
            return summary

        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as file:
                rows = [{k: (v or '').strip() for k, v in row.items()} for row in csv.DictReader(file)]
        except Exception as e:
            summary['error'] = f"Failed reading audit file: {e}"
            return summary

        if not rows:
            summary['error'] = "Audit file is empty."
            return summary

        def extract_day(ts: str) -> str:
            ts = (ts or '').strip()
            if not ts:
                return 'unknown'
            try:
                return datetime.fromisoformat(ts).date().isoformat()
            except ValueError:
                return ts[:10] if len(ts) >= 10 else 'unknown'

        if days_window > 0:
            valid_days = sorted({extract_day(r.get('timestamp', '')) for r in rows if extract_day(r.get('timestamp', '')) != 'unknown'})
            keep_days = set(valid_days[-days_window:]) if valid_days else set()
            rows = [r for r in rows if not keep_days or extract_day(r.get('timestamp', '')) in keep_days]

        decision_counts: Counter = Counter()
        rule_counts: Counter = Counter()
        rule_decision_counts: Counter = Counter()
        daily_counts: Counter = Counter()
        candidate_counts: Counter = Counter()

        for row in rows:
            day = extract_day(row.get('timestamp', ''))
            decision = row.get('decision', '') or '(blank)'
            rule_name = row.get('rule_name', '') or '(blank)'
            candidate = row.get('candidate', '') or '(blank)'

            decision_counts.update([decision])
            rule_counts.update([rule_name])
            rule_decision_counts.update([f"{rule_name} | {decision}"])
            daily_counts.update([day])
            candidate_counts.update([candidate])

        summary['available'] = True
        summary['rows_analyzed'] = len(rows)
        summary['decision_counts'] = dict(decision_counts)
        summary['rule_counts'] = rule_counts.most_common(8)
        summary['rule_decision_counts'] = rule_decision_counts.most_common(10)
        summary['daily_counts'] = daily_counts.most_common(14)
        summary['top_candidates'] = candidate_counts.most_common(8)
        return summary

    def _build_address_alias_audit_html(self, alias_data: dict) -> str:
        """Build HTML for address alias audit section."""
        if not alias_data:
            return "<p class='error-box'>❌ Address alias audit summary unavailable</p>"

        if not alias_data.get('available'):
            err = self._escape_html(alias_data.get('error', 'Address alias audit unavailable'))
            path = self._escape_html(alias_data.get('path', ''))
            return (
                "<p class='error-box'>"
                f"⚠ Address alias audit not available.<br>{err}<br><strong>Path:</strong> {path}"
                "</p>"
            )

        decision_counts = alias_data.get('decision_counts', {})
        applied_count = int(decision_counts.get('applied', 0))
        conflict_skips = int(decision_counts.get('skipped_conflict', 0))
        rows_analyzed = int(alias_data.get('rows_analyzed', 0))
        unique_rules = len(alias_data.get('rule_counts', []))

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{rows_analyzed}</div>
                <div class="metric-label">Rows Analyzed (Recent Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{applied_count}</div>
                <div class="metric-label">Alias Matches Applied</div>
            </div>
            <div class="metric">
                <div class="metric-value">{conflict_skips}</div>
                <div class="metric-label">Conflict Skips</div>
            </div>
            <div class="metric">
                <div class="metric-value">{unique_rules}</div>
                <div class="metric-label">Active Rules (Top List)</div>
            </div>
        </div>
        """

        if alias_data.get('daily_counts'):
            html += "<h3>Daily Volume</h3><table><tr><th>Day</th><th>Count</th></tr>"
            for day, count in alias_data['daily_counts']:
                html += f"<tr><td>{self._escape_html(day)}</td><td>{count}</td></tr>"
            html += "</table>"

        if alias_data.get('rule_decision_counts'):
            html += "<h3>Top Rule + Decision Pairs</h3><table><tr><th>Rule | Decision</th><th>Count</th></tr>"
            for key, count in alias_data['rule_decision_counts']:
                html += f"<tr><td>{self._escape_html(key)}</td><td>{count}</td></tr>"
            html += "</table>"

        if alias_data.get('top_candidates'):
            html += "<h3>Top Candidate Text</h3><table><tr><th>Candidate</th><th>Count</th></tr>"
            for candidate, count in alias_data['top_candidates']:
                html += f"<tr><td>{self._escape_html(candidate)}</td><td>{count}</td></tr>"
            html += "</table>"

        html += (
            f"<p><strong>Audit Source:</strong> {self._escape_html(alias_data.get('path', ''))}"
            f" (window: last {alias_data.get('days_window', 14)} day(s) present)</p>"
        )
        return html

    @staticmethod
    def _parse_log_timestamp(line: str) -> datetime | None:
        """Parse timestamp prefix in log lines (YYYY-MM-DD HH:MM:SS)."""
        try:
            return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _summarize_llm_provider_activity(self, report_timestamp: str | None) -> dict:
        """
        Summarize provider-level LLM activity from recent logs.
        """
        hours_window = int(
            self.validation_config.get('reporting', {}).get('llm_activity_hours', 24) or 24
        )
        end_ts = datetime.now()
        if report_timestamp:
            try:
                end_ts = datetime.fromisoformat(report_timestamp)
            except ValueError:
                pass
        start_ts = end_ts - timedelta(hours=hours_window)

        log_files = [
            "logs/emails_log.txt",
            "logs/gs_log.txt",
            "logs/rd_ext_log.txt",
            "logs/ebs_log.txt",
            "logs/scraper_log.txt",
            "logs/fb_log.txt",
            "logs/images_log.txt",
            "logs/read_pdfs_log.txt",
            "logs/db_log.txt",
            "logs/clean_up_log.txt",
            "logs/dedup_llm_log.txt",
            "logs/irrelevant_rows_log.txt",
            "logs/validation_tests_log.txt",
        ]

        providers = ("openai", "openrouter", "mistral", "gemini")
        stats = {
            p: {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
                "rate_limits": 0,
            }
            for p in providers
        }
        file_attempts: Counter = Counter()
        model_attempts: Counter = Counter()
        total_provider_exhausted = 0
        lines_scanned = 0

        def _inc_if(line: str, needle: str, provider: str, metric: str) -> None:
            if needle in line:
                stats[provider][metric] += 1

        for path in log_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        ts = self._parse_log_timestamp(line)
                        if ts is None or ts < start_ts or ts > end_ts:
                            continue

                        lines_scanned += 1
                        low = line.lower()

                        if "query_llm(): querying openai" in low:
                            stats["openai"]["attempts"] += 1
                            file_attempts[path] += 1
                        if "query_llm(): querying openrouter" in low:
                            stats["openrouter"]["attempts"] += 1
                            file_attempts[path] += 1
                        if "query_llm(): querying mistral" in low:
                            stats["mistral"]["attempts"] += 1
                            file_attempts[path] += 1
                        if "query_llm(): querying gemini" in low:
                            stats["gemini"]["attempts"] += 1
                            file_attempts[path] += 1
                        for provider in providers:
                            marker = f"query_llm(): querying {provider} model "
                            idx = low.find(marker)
                            if idx >= 0:
                                model = line[idx + len(marker):].strip().split()[0]
                                if model:
                                    model_attempts[f"{provider}:{model}"] += 1

                        _inc_if(low, "query_llm(): openai response received", "openai", "successes")
                        _inc_if(low, "query_llm(): openrouter response received", "openrouter", "successes")
                        _inc_if(low, "query_llm(): mistral response received", "mistral", "successes")
                        _inc_if(low, "query_llm(): gemini response received", "gemini", "successes")

                        _inc_if(low, "query_llm(): openai query failed", "openai", "failures")
                        _inc_if(low, "query_llm(): openrouter query failed", "openrouter", "failures")
                        _inc_if(low, "query_llm(): mistral query failed", "mistral", "failures")
                        _inc_if(low, "query_llm(): gemini query failed", "gemini", "failures")

                        if "request timed out" in low or "read timed out" in low:
                            if "openai query failed" in low:
                                stats["openai"]["timeouts"] += 1
                            if "gemini query failed" in low:
                                stats["gemini"]["timeouts"] += 1

                        if "status 429" in low or "too many requests" in low or "rate limit exceeded" in low:
                            if "openrouter query failed" in low:
                                stats["openrouter"]["rate_limits"] += 1
                            if "mistral query failed" in low:
                                stats["mistral"]["rate_limits"] += 1
                            if "openai query failed" in low:
                                stats["openai"]["rate_limits"] += 1
                            if "gemini query failed" in low:
                                stats["gemini"]["rate_limits"] += 1

                        if "all configured llm providers failed to provide a response" in low:
                            total_provider_exhausted += 1
            except Exception as e:
                logging.warning("_summarize_llm_provider_activity: Failed to parse %s: %s", path, e)

        total_attempts = sum(v["attempts"] for v in stats.values())
        total_rate_limits = sum(v["rate_limits"] for v in stats.values())
        total_timeouts = sum(v["timeouts"] for v in stats.values())
        gemini_attempts = int(stats.get("gemini", {}).get("attempts", 0))

        thresholds_cfg = (
            self.validation_config.get("reporting", {}).get("llm_activity_thresholds", {})
        )
        high_cfg = thresholds_cfg.get("high", {}) if isinstance(thresholds_cfg, dict) else {}
        medium_cfg = thresholds_cfg.get("medium", {}) if isinstance(thresholds_cfg, dict) else {}

        high_attempts = int(high_cfg.get("attempts", 150) or 150)
        high_exhausted = int(high_cfg.get("provider_exhausted", 10) or 10)
        high_rate_limits = int(high_cfg.get("rate_limits", 25) or 25)
        high_timeouts = int(high_cfg.get("timeouts", 15) or 15)
        high_gemini = int(high_cfg.get("gemini_attempts", 20) or 20)

        medium_attempts = int(medium_cfg.get("attempts", 60) or 60)
        medium_exhausted = int(medium_cfg.get("provider_exhausted", 3) or 3)
        medium_rate_limits = int(medium_cfg.get("rate_limits", 10) or 10)
        medium_timeouts = int(medium_cfg.get("timeouts", 5) or 5)
        medium_gemini = int(medium_cfg.get("gemini_attempts", 5) or 5)

        pressure_reasons: list[str] = []
        if total_attempts >= high_attempts:
            pressure_reasons.append(f"high attempt volume ({total_attempts})")
        if total_provider_exhausted >= high_exhausted:
            pressure_reasons.append(f"many full fallback failures ({total_provider_exhausted})")
        if total_rate_limits >= high_rate_limits:
            pressure_reasons.append(f"heavy rate limiting ({total_rate_limits})")
        if total_timeouts >= high_timeouts:
            pressure_reasons.append(f"many timeouts ({total_timeouts})")
        if gemini_attempts >= high_gemini:
            pressure_reasons.append(f"frequent Gemini fallback ({gemini_attempts})")

        pressure_level = "LOW"
        if pressure_reasons:
            pressure_level = "HIGH"
        else:
            medium_signals = [
                total_attempts >= medium_attempts,
                total_provider_exhausted >= medium_exhausted,
                total_rate_limits >= medium_rate_limits,
                total_timeouts >= medium_timeouts,
                gemini_attempts >= medium_gemini,
            ]
            if any(medium_signals):
                pressure_level = "MEDIUM"

        return {
            "window_hours": hours_window,
            "start_ts": start_ts.isoformat(sep=" "),
            "end_ts": end_ts.isoformat(sep=" "),
            "providers": stats,
            "total_attempts": total_attempts,
            "provider_exhausted_count": total_provider_exhausted,
            "lines_scanned": lines_scanned,
            "top_files": file_attempts.most_common(8),
            "top_models": model_attempts.most_common(12),
            "total_rate_limits": total_rate_limits,
            "total_timeouts": total_timeouts,
            "pressure_level": pressure_level,
            "pressure_reasons": pressure_reasons,
        }

    def _build_llm_provider_activity_html(self, llm_data: dict) -> str:
        """Build HTML for LLM provider usage/cost pressure section."""
        if not llm_data:
            return "<p class='error-box'>❌ LLM activity summary unavailable</p>"

        total_attempts = int(llm_data.get("total_attempts", 0))
        exhausted = int(llm_data.get("provider_exhausted_count", 0))
        lines_scanned = int(llm_data.get("lines_scanned", 0))
        total_rate_limits = int(llm_data.get("total_rate_limits", 0))
        total_timeouts = int(llm_data.get("total_timeouts", 0))
        pressure_level = str(llm_data.get("pressure_level", "LOW")).upper()
        pressure_class = {
            "LOW": "status-pass",
            "MEDIUM": "status-warning",
            "HIGH": "status-fail",
        }.get(pressure_level, "status-warning")

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{total_attempts}</div>
                <div class="metric-label">LLM Attempts (Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{exhausted}</div>
                <div class="metric-label">All-Provider Exhausted</div>
            </div>
            <div class="metric">
                <div class="metric-value">{lines_scanned}</div>
                <div class="metric-label">Log Lines Scanned</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_rate_limits}</div>
                <div class="metric-label">Rate Limits (All Providers)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_timeouts}</div>
                <div class="metric-label">Timeouts (All Providers)</div>
            </div>
        </div>
        """

        html += (
            "<p><strong>Cost Pressure:</strong> "
            f"<span class='{pressure_class}'>{self._escape_html(pressure_level)}</span></p>"
        )
        reasons = llm_data.get("pressure_reasons", [])
        if reasons:
            html += "<ul>"
            for reason in reasons:
                html += f"<li>{self._escape_html(reason)}</li>"
            html += "</ul>"

        html += (
            "<p><strong>Window:</strong> "
            f"{self._escape_html(llm_data.get('start_ts', ''))} to {self._escape_html(llm_data.get('end_ts', ''))} "
            f"({int(llm_data.get('window_hours', 24))}h)</p>"
        )

        html += "<h3>Provider Breakdown</h3>"
        html += "<table><tr><th>Provider</th><th>Attempts</th><th>Successes</th><th>Failures</th><th>Timeouts</th><th>Rate Limits</th></tr>"
        providers = llm_data.get("providers", {})
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            p = providers.get(provider, {})
            html += (
                "<tr>"
                f"<td>{provider}</td>"
                f"<td>{int(p.get('attempts', 0))}</td>"
                f"<td>{int(p.get('successes', 0))}</td>"
                f"<td>{int(p.get('failures', 0))}</td>"
                f"<td>{int(p.get('timeouts', 0))}</td>"
                f"<td>{int(p.get('rate_limits', 0))}</td>"
                "</tr>"
            )
        html += "</table>"

        top_files = llm_data.get("top_files", [])
        if top_files:
            html += "<h3>Top Log Files by LLM Attempts</h3><table><tr><th>Log File</th><th>Attempts</th></tr>"
            for path, attempts in top_files:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(os.path.basename(path))}</td>"
                    f"<td>{int(attempts)}</td>"
                    "</tr>"
                )
            html += "</table>"

        top_models = llm_data.get("top_models", [])
        if top_models:
            html += "<h3>Top Provider/Model Attempts</h3><table><tr><th>Provider:Model</th><th>Attempts</th></tr>"
            for model_key, attempts in top_models:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(model_key)}</td>"
                    f"<td>{int(attempts)}</td>"
                    "</tr>"
                )
            html += "</table>"

        return html

    def _summarize_reliability_scorecard(self, results: dict, llm_data: dict, alias_data: dict) -> dict:
        """Build baseline reliability metrics from report-driving data."""
        scraping = results.get('scraping_validation') or {}
        scraping_summary = scraping.get('summary', {}) if isinstance(scraping, dict) else {}
        critical_failures = len(scraping.get('critical_failures', []) or []) if isinstance(scraping, dict) else 0

        chatbot = results.get('chatbot_testing') or {}
        chatbot_summary = chatbot.get('summary', {}) if isinstance(chatbot, dict) else {}
        avg_score = float(chatbot_summary.get('average_score', 0) or 0)
        exec_rate = float(chatbot_summary.get('execution_success_rate', 0) or 0)

        total_attempts = int((llm_data or {}).get('total_attempts', 0) or 0)
        providers = (llm_data or {}).get('providers', {}) or {}
        total_successes = sum(int((providers.get(p, {}) or {}).get('successes', 0) or 0) for p in ("openai", "openrouter", "mistral", "gemini"))
        total_rate_limits = int((llm_data or {}).get('total_rate_limits', 0) or 0)
        total_timeouts = int((llm_data or {}).get('total_timeouts', 0) or 0)
        exhausted = int((llm_data or {}).get('provider_exhausted_count', 0) or 0)
        llm_success_rate = (total_successes / total_attempts) if total_attempts else 0.0
        rate_limit_rate = (total_rate_limits / total_attempts) if total_attempts else 0.0

        decision_counts = (alias_data or {}).get('decision_counts', {}) or {}
        alias_conflicts = int(decision_counts.get('skipped_conflict', 0) or 0)

        score = 100.0
        score -= min(30.0, critical_failures * 3.0)
        score -= max(0.0, min(20.0, 90.0 - avg_score))
        score -= max(0.0, min(20.0, (1.0 - exec_rate) * 20.0))
        score -= min(15.0, rate_limit_rate * 100.0)
        score -= min(10.0, exhausted * 2.0)
        score = max(0.0, min(100.0, score))

        if score >= 85:
            status = "HEALTHY"
        elif score >= 70:
            status = "WATCH"
        else:
            status = "AT_RISK"

        return {
            "timestamp": results.get("timestamp", datetime.now().isoformat()),
            "status": status,
            "score": round(score, 1),
            "metrics": {
                "scrape_critical_failures": critical_failures,
                "scrape_total_failures": int(scraping_summary.get('total_failures', 0) or 0),
                "chatbot_average_score": round(avg_score, 2),
                "chatbot_execution_success_rate": round(exec_rate, 4),
                "llm_attempts": total_attempts,
                "llm_success_rate": round(llm_success_rate, 4),
                "llm_rate_limits": total_rate_limits,
                "llm_timeouts": total_timeouts,
                "llm_provider_exhausted": exhausted,
                "llm_cost_pressure": (llm_data or {}).get("pressure_level", "LOW"),
                "address_alias_conflict_skips": alias_conflicts,
            },
        }

    def _extract_reliability_issues(self, results: dict, llm_data: dict, alias_data: dict) -> list[dict]:
        """Normalize major reliability signals into a JSON-friendly issue list."""
        now_ts = str(results.get("timestamp") or datetime.now().isoformat())
        issues: list[dict] = []

        scraping = results.get('scraping_validation') or {}
        for idx, failure in enumerate((scraping.get('critical_failures') or [])[:30], start=1):
            url = str(failure.get('url', '') or '')
            reason = str(failure.get('reason_code', failure.get('failure_type', 'unknown')) or 'unknown')
            issues.append({
                "issue_id": f"SCRAPE-{idx:03d}",
                "timestamp": now_ts,
                "category": "Scrape Coverage Miss",
                "severity": "high",
                "step": "scraping_validation",
                "provider": "",
                "url": url,
                "input_signature": reason,
                "expected": "Event source should produce expected events.",
                "actual": f"Critical scrape failure detected ({reason}).",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        chatbot = results.get('chatbot_testing') or {}
        for idx, cat in enumerate((chatbot.get('problem_categories') or [])[:10], start=1):
            ex = cat.get('example', {}) or {}
            issues.append({
                "issue_id": f"CHATBOT-{idx:03d}",
                "timestamp": now_ts,
                "category": "LLM Semantic Failure",
                "severity": "medium",
                "step": "chatbot_testing",
                "provider": "",
                "url": "",
                "input_signature": str(cat.get('name', 'unknown')),
                "expected": "Chatbot intent + SQL output should meet quality threshold.",
                "actual": str(ex.get('reason', 'Category below target threshold.')),
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        providers = (llm_data or {}).get("providers", {}) or {}
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            pdata = providers.get(provider, {}) or {}
            failures = int(pdata.get("failures", 0) or 0)
            timeouts = int(pdata.get("timeouts", 0) or 0)
            rate_limits = int(pdata.get("rate_limits", 0) or 0)
            if failures <= 0 and timeouts <= 0 and rate_limits <= 0:
                continue
            issues.append({
                "issue_id": f"LLM-{provider.upper()}-001",
                "timestamp": now_ts,
                "category": "Provider Reliability Failure",
                "severity": "high" if (timeouts > 0 or rate_limits > 0) else "medium",
                "step": "llm_activity",
                "provider": provider,
                "url": "",
                "input_signature": "provider_failure_signal",
                "expected": "Provider should respond within timeout and quota limits.",
                "actual": f"failures={failures}, timeouts={timeouts}, rate_limits={rate_limits}",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        decision_counts = (alias_data or {}).get('decision_counts', {}) or {}
        alias_conflicts = int(decision_counts.get('skipped_conflict', 0) or 0)
        if alias_conflicts > 0:
            issues.append({
                "issue_id": "ADDRESS-ALIAS-001",
                "timestamp": now_ts,
                "category": "Data Integrity Failure",
                "severity": "medium",
                "step": "address_alias_audit",
                "provider": "",
                "url": "",
                "input_signature": "skipped_conflict",
                "expected": "Alias normalization should avoid unresolved conflicts.",
                "actual": f"Conflict skips observed: {alias_conflicts}",
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        gates = results.get('reliability_gates') or {}
        for gate in (gates.get('failed_gates') or []):
            gate_name = str(gate.get('name', 'unknown_gate'))
            issues.append({
                "issue_id": f"GATE-{gate_name.upper().replace('-', '_')}",
                "timestamp": now_ts,
                "category": "Reliability Gate Failure",
                "severity": "high",
                "step": "reliability_gates",
                "provider": "",
                "url": "",
                "input_signature": gate_name,
                "expected": f"Gate '{gate_name}' threshold should pass.",
                "actual": str(gate.get('detail', 'Gate threshold breached.')),
                "status": "open",
                "owner": "unassigned",
                "first_seen": now_ts,
                "last_seen": now_ts,
            })

        return issues

    def _build_reliability_scorecard_html(self, scorecard: dict, issues: list[dict], gates: dict | None = None) -> str:
        """Render baseline reliability scorecard + normalized issue summary."""
        if not scorecard:
            return "<p class='error-box'>❌ Reliability scorecard unavailable</p>"

        status = str(scorecard.get("status", "WATCH")).upper()
        status_class = {
            "HEALTHY": "status-pass",
            "WATCH": "status-warning",
            "AT_RISK": "status-fail",
        }.get(status, "status-warning")
        metrics = scorecard.get("metrics", {}) or {}
        open_issues = len(issues or [])

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{float(scorecard.get('score', 0)):.1f}</div>
                <div class="metric-label">Reliability Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self._escape_html(status)}</div>
                <div class="metric-label">Reliability Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{open_issues}</div>
                <div class="metric-label">Normalized Open Issues</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(metrics.get('llm_attempts', 0))}</div>
                <div class="metric-label">LLM Attempts (Window)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{int(metrics.get('llm_rate_limits', 0))}</div>
                <div class="metric-label">LLM Rate Limits</div>
            </div>
        </div>
        <p><strong>Status:</strong> <span class="{status_class}">{self._escape_html(status)}</span></p>
        """

        html += "<h3>Core Reliability Metrics</h3>"
        html += "<table><tr><th>Metric</th><th>Value</th></tr>"
        for key in (
            "scrape_critical_failures",
            "scrape_total_failures",
            "chatbot_average_score",
            "chatbot_execution_success_rate",
            "llm_success_rate",
            "llm_provider_exhausted",
            "llm_cost_pressure",
            "address_alias_conflict_skips",
        ):
            html += (
                "<tr>"
                f"<td>{self._escape_html(key)}</td>"
                f"<td>{self._escape_html(metrics.get(key, ''))}</td>"
                "</tr>"
            )
        html += "</table>"

        gate_payload = gates or {}
        gate_status = str(gate_payload.get("status", "PASS")).upper()
        gate_status_class = {
            "PASS": "status-pass",
            "WARNING": "status-warning",
            "FAIL": "status-fail",
            "ERROR": "status-fail",
        }.get(gate_status, "status-warning")
        html += "<h3>Reliability Gates</h3>"
        html += (
            "<p><strong>Gate Status:</strong> "
            f"<span class=\"{gate_status_class}\">{self._escape_html(gate_status)}</span></p>"
        )
        gate_results = gate_payload.get("gate_results", []) or []
        if gate_results:
            html += "<table><tr><th>Gate</th><th>Status</th><th>Detail</th></tr>"
            for gate in gate_results:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(gate.get('name', ''))}</td>"
                    f"<td>{self._escape_html(gate.get('status', ''))}</td>"
                    f"<td>{self._escape_html(gate.get('detail', ''))}</td>"
                    "</tr>"
                )
            html += "</table>"

        if issues:
            html += "<h3>Top Reliability Issues</h3>"
            html += "<table><tr><th>ID</th><th>Category</th><th>Severity</th><th>Step</th><th>Provider</th><th>Actual</th></tr>"
            for issue in issues[:20]:
                html += (
                    "<tr>"
                    f"<td>{self._escape_html(issue.get('issue_id', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('category', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('severity', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('step', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('provider', ''))}</td>"
                    f"<td>{self._escape_html(issue.get('actual', ''))}</td>"
                    "</tr>"
                )
            html += "</table>"
        else:
            html += "<p>✅ No normalized reliability issues in current window.</p>"

        return html

    def _evaluate_reliability_gates(self, scorecard: dict) -> dict:
        """Evaluate configurable reliability gates and return pass/fail details."""
        reporting_cfg = self.validation_config.get('reporting', {})
        gates_cfg = reporting_cfg.get('reliability_gates', {}) if isinstance(reporting_cfg, dict) else {}
        enabled = bool(gates_cfg.get('enabled', True))
        thresholds = gates_cfg.get('thresholds', {}) if isinstance(gates_cfg, dict) else {}
        metrics = (scorecard or {}).get('metrics', {}) or {}

        defaults = {
            "min_reliability_score": 75,
            "min_chatbot_execution_success_rate": 0.90,
            "min_chatbot_average_score": 70,
            "max_llm_rate_limits": 25,
            "max_llm_timeouts": 15,
            "max_llm_provider_exhausted": 10,
            "max_scrape_critical_failures": 0,
        }
        merged_thresholds = {**defaults, **(thresholds or {})}

        if not enabled:
            return {
                "enabled": False,
                "status": "PASS",
                "thresholds": merged_thresholds,
                "gate_results": [],
                "failed_gates": [],
                "evaluated_at": datetime.now().isoformat(),
            }

        gate_results: list[dict] = []

        def check_min(name: str, actual: float, minimum: float) -> None:
            ok = actual >= minimum
            gate_results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": f"actual={actual} min_required={minimum}",
            })

        def check_max(name: str, actual: float, maximum: float) -> None:
            ok = actual <= maximum
            gate_results.append({
                "name": name,
                "status": "PASS" if ok else "FAIL",
                "detail": f"actual={actual} max_allowed={maximum}",
            })

        check_min(
            "min_reliability_score",
            float((scorecard or {}).get("score", 0) or 0),
            float(merged_thresholds["min_reliability_score"]),
        )
        check_min(
            "min_chatbot_execution_success_rate",
            float(metrics.get("chatbot_execution_success_rate", 0) or 0),
            float(merged_thresholds["min_chatbot_execution_success_rate"]),
        )
        check_min(
            "min_chatbot_average_score",
            float(metrics.get("chatbot_average_score", 0) or 0),
            float(merged_thresholds["min_chatbot_average_score"]),
        )
        check_max(
            "max_llm_rate_limits",
            float(metrics.get("llm_rate_limits", 0) or 0),
            float(merged_thresholds["max_llm_rate_limits"]),
        )
        check_max(
            "max_llm_timeouts",
            float(metrics.get("llm_timeouts", 0) or 0),
            float(merged_thresholds["max_llm_timeouts"]),
        )
        check_max(
            "max_llm_provider_exhausted",
            float(metrics.get("llm_provider_exhausted", 0) or 0),
            float(merged_thresholds["max_llm_provider_exhausted"]),
        )
        check_max(
            "max_scrape_critical_failures",
            float(metrics.get("scrape_critical_failures", 0) or 0),
            float(merged_thresholds["max_scrape_critical_failures"]),
        )

        failed_gates = [g for g in gate_results if g.get("status") == "FAIL"]
        status = "FAIL" if failed_gates else "PASS"
        return {
            "enabled": True,
            "status": status,
            "thresholds": merged_thresholds,
            "gate_results": gate_results,
            "failed_gates": failed_gates,
            "evaluated_at": datetime.now().isoformat(),
        }

    def _build_optimization_plan(self, results: dict, scorecard: dict, llm_data: dict) -> dict:
        """
        Phase 4: Build continuous optimization recommendations from observed outcomes.
        """
        providers = (llm_data or {}).get("providers", {}) or {}
        default_order = list(self.config.get("llm", {}).get("chatbot_provider_order", ["openai", "openrouter", "gemini"]))
        provider_scores: dict[str, dict] = {}
        ranked: list[tuple[str, float]] = []
        recommendations: list[str] = []

        for provider in ("openai", "openrouter", "mistral", "gemini"):
            pdata = providers.get(provider, {}) or {}
            attempts = int(pdata.get("attempts", 0) or 0)
            successes = int(pdata.get("successes", 0) or 0)
            failures = int(pdata.get("failures", 0) or 0)
            rate_limits = int(pdata.get("rate_limits", 0) or 0)
            timeouts = int(pdata.get("timeouts", 0) or 0)
            success_rate = (successes / attempts) if attempts else 0.0
            failure_rate = (failures / attempts) if attempts else 0.0
            rate_limit_rate = (rate_limits / attempts) if attempts else 0.0
            timeout_rate = (timeouts / attempts) if attempts else 0.0
            # Reliability-weighted score: emphasize successful completion, penalize unstable providers.
            score = (success_rate * 100.0) - (rate_limit_rate * 40.0) - (timeout_rate * 30.0) - (failure_rate * 20.0)
            provider_scores[provider] = {
                "attempts": attempts,
                "success_rate": round(success_rate, 4),
                "failure_rate": round(failure_rate, 4),
                "rate_limit_rate": round(rate_limit_rate, 4),
                "timeout_rate": round(timeout_rate, 4),
                "health_score": round(score, 2),
            }
            ranked.append((provider, score))
            if attempts > 0 and rate_limit_rate >= 0.20:
                recommendations.append(
                    f"{provider}: high rate-limit ratio ({rate_limits}/{attempts}); reduce priority or increase cooldown."
                )
            if attempts > 0 and timeout_rate >= 0.15:
                recommendations.append(
                    f"{provider}: elevated timeout ratio ({timeouts}/{attempts}); keep as fallback until stable."
                )

        ranked.sort(key=lambda x: x[1], reverse=True)
        optimized_order = [p for p, _ in ranked]

        current_metrics = (scorecard or {}).get("metrics", {}) or {}
        if float(current_metrics.get("llm_success_rate", 0) or 0) < 0.8:
            recommendations.append("Overall LLM success rate is below 80%; tighten routing and fallback depth for chatbot requests.")
        if str(current_metrics.get("llm_cost_pressure", "LOW")).upper() in {"MEDIUM", "HIGH"}:
            recommendations.append("Cost pressure is elevated; prioritize stable low-cost providers and avoid deep fallback chains.")
        if not recommendations:
            recommendations.append("Routing health is stable. Keep current order and continue monitoring 7-day trend deltas.")

        return {
            "generated_at": datetime.now().isoformat(),
            "current_chatbot_provider_order": default_order,
            "recommended_chatbot_provider_order": optimized_order,
            "provider_scores": provider_scores,
            "quality_snapshot": {
                "chatbot_average_score": current_metrics.get("chatbot_average_score", 0),
                "chatbot_execution_success_rate": current_metrics.get("chatbot_execution_success_rate", 0),
                "reliability_score": (scorecard or {}).get("score", 0),
            },
            "recommendations": recommendations,
            "config_patch_preview": {
                "llm": {
                    "chatbot_provider_order": optimized_order
                }
            },
        }

    def _build_optimization_html(self, optimization_plan: dict) -> str:
        """Render optimization recommendations section."""
        if not optimization_plan:
            return "<p class='error-box'>❌ Optimization recommendations unavailable</p>"

        current_order = optimization_plan.get("current_chatbot_provider_order", [])
        recommended_order = optimization_plan.get("recommended_chatbot_provider_order", [])
        html = (
            f"<p><strong>Current chatbot provider order:</strong> {self._escape_html(' -> '.join(current_order))}</p>"
            f"<p><strong>Recommended order:</strong> {self._escape_html(' -> '.join(recommended_order))}</p>"
        )
        html += "<h3>Provider Health Scores</h3>"
        html += "<table><tr><th>Provider</th><th>Attempts</th><th>Success Rate</th><th>Rate Limit Rate</th><th>Timeout Rate</th><th>Health Score</th></tr>"
        provider_scores = optimization_plan.get("provider_scores", {}) or {}
        for provider in ("openai", "openrouter", "mistral", "gemini"):
            stats = provider_scores.get(provider, {}) or {}
            html += (
                "<tr>"
                f"<td>{self._escape_html(provider)}</td>"
                f"<td>{int(stats.get('attempts', 0) or 0)}</td>"
                f"<td>{float(stats.get('success_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('rate_limit_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('timeout_rate', 0) or 0):.2%}</td>"
                f"<td>{float(stats.get('health_score', 0) or 0):.2f}</td>"
                "</tr>"
            )
        html += "</table>"

        recs = optimization_plan.get("recommendations", []) or []
        if recs:
            html += "<h3>Recommended Actions</h3><ul>"
            for rec in recs:
                html += f"<li>{self._escape_html(rec)}</li>"
            html += "</ul>"

        html += (
            "<p><strong>Patch Preview:</strong> "
            "<code>llm.chatbot_provider_order</code> "
            f"→ {self._escape_html(str(recommended_order))}</p>"
        )
        return html

    def _build_scraping_html(self, scraping_data: dict) -> str:
        """Build HTML for scraping validation section."""
        if not scraping_data:
            return "<p class='error-box'>❌ Scraping validation did not run</p>"

        if 'error' in scraping_data:
            return f"<p class='error-box'>❌ Scraping validation failed: {scraping_data['error']}</p>"

        summary = scraping_data['summary']

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{summary['total_failures']}</div>
                <div class="metric-label">Total Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['whitelist_failures']}</div>
                <div class="metric-label">Whitelist Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['edge_case_failures']}</div>
                <div class="metric-label">Edge Case Failures</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['high_performer_failures']}</div>
                <div class="metric-label">High-Performer Failures</div>
            </div>
        </div>

        <h3>Critical Failures</h3>
        """

        if scraping_data.get('critical_failures'):
            html += (
                "<table><tr>"
                "<th>URL</th><th>Source</th><th>Type</th><th>Failure Type</th>"
                "<th>Reason Code</th><th>Probable Cause</th>"
                "<th>Last Attempt</th><th>Child Successes</th><th>Evidence</th><th>Recommendation</th>"
                "</tr>"
            )
            for failure in scraping_data['critical_failures'][:20]:  # Limit to 20
                evidence_lines = failure.get('evidence') or []
                if evidence_lines:
                    evidence_html = "<br>".join(
                        line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        for line in evidence_lines
                    )
                else:
                    evidence_html = ""
                html += f"""
                <tr class="problematic">
                    <td>{failure['url'][:60]}...</td>
                    <td>{failure['source']}</td>
                    <td>{failure['importance']}</td>
                    <td>{failure['failure_type']}</td>
                    <td>{failure.get('reason_code', '')}</td>
                    <td>{failure.get('probable_cause', '')}</td>
                    <td>{failure.get('last_attempt_time', '')}</td>
                    <td>{failure.get('recent_relevant_child_count', '')}</td>
                    <td style="font-size:0.85em;">{evidence_html}</td>
                    <td>{failure['recommendation']}</td>
                </tr>
                """
            html += "</table>"
        else:
            html += "<p>✅ No critical failures</p>"

        # Add source distribution check
        if 'source_distribution' in scraping_data:
            dist = scraping_data['source_distribution']
            html += "<h3>Source Distribution Check</h3>"

            # Status and metrics
            status_class = dist['status'].lower()
            html += f"<p><strong>Status:</strong> <span class='status-{status_class}'>{dist['status']}</span></p>"

            if 'total_events' in dist:
                html += f"""
                <div class="metric-container">
                    <div class="metric">
                        <div class="metric-value">{dist['total_events']}</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('top_10_total', 0)}</div>
                        <div class="metric-label">Top 10 Sources</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('top_10_percentage', 0)}%</div>
                        <div class="metric-label">Top 10 Percentage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{dist.get('total_sources', 0)}</div>
                        <div class="metric-label">Total Sources</div>
                    </div>
                </div>
                """

                # Missing sources (CRITICAL)
                if dist.get('missing_sources'):
                    html += "<h4>❌ Missing Required Sources</h4>"
                    html += "<p class='error-box'>The following required sources are NOT present in the database. Scraping likely failed for these sources:</p>"
                    html += "<ul>"
                    for missing_source in dist['missing_sources']:
                        html += f"<li class='error-box'><strong>{missing_source}</strong></li>"
                    html += "</ul>"

                # Top 10 sources table
                if 'top_10_sources' in dist:
                    html += "<h4>Top 10 Event Sources</h4>"
                    html += "<table><tr><th>Rank</th><th>Source</th><th>Event Count</th></tr>"
                    for i, source_info in enumerate(dist['top_10_sources'], 1):
                        html += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{source_info['source']}</td>
                            <td>{source_info['count']}</td>
                        </tr>
                        """
                    html += "</table>"

            # Warnings
            if dist.get('warnings'):
                html += "<h4>Warnings</h4><ul>"
                for warning in dist['warnings']:
                    html += f"<li class='error-box'>{warning}</li>"
                html += "</ul>"
            elif not dist.get('missing_sources'):
                html += "<p>✅ All required sources present in database</p>"

        return html

    def _build_chatbot_html(self, chatbot_data: dict) -> str:
        """Build HTML for chatbot testing section."""
        if not chatbot_data:
            return "<p class='error-box'>❌ Chatbot testing did not run</p>"

        if 'error' in chatbot_data:
            return f"<p class='error-box'>❌ Chatbot testing failed: {chatbot_data['error']}</p>"

        summary = chatbot_data['summary']

        html = f"""
        <div class="metric-container">
            <div class="metric">
                <div class="metric-value">{summary['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['average_score']:.1f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['execution_success_rate']:.1%}</div>
                <div class="metric-label">Execution Success</div>
            </div>
        </div>

        <h3>Score Distribution</h3>
        <table>
            <tr><th>Score Range</th><th>Count</th></tr>
        """

        for range_name, count in summary['score_distribution'].items():
            html += f"<tr><td>{range_name}</td><td>{count}</td></tr>"

        html += "</table>"

        if chatbot_data.get('problematic_questions'):
            html += "<h3>Problematic Questions (Score < 50)</h3>"
            html += "<table><tr><th>Question</th><th>Category</th><th>Score</th><th>Reasoning</th></tr>"
            for q in chatbot_data['problematic_questions'][:10]:  # Show top 10
                html += f"""
                <tr class="problematic">
                    <td>{q['question']}</td>
                    <td>{q['category']}</td>
                    <td>{q['score']}</td>
                    <td>{q['reasoning'][:100]}...</td>
                </tr>
                """
            html += "</table>"

        # New: Problem Categories (score < 90)
        if chatbot_data.get('problem_categories'):
            html += "<h3>Problem Categories (Score < 90)</h3>"
            for cat in chatbot_data['problem_categories']:
                html += f"<h4>{cat['name']} ({cat['count']} issues)</h4>"
                example_q = cat.get('example', {}).get('question', '')
                example_reason = cat.get('example', {}).get('reason', '')
                html += f"<p><strong>Example:</strong> {example_q}<br><em>{example_reason}</em></p>"
                example_sql = cat.get('example', {}).get('sql', '')
                if example_sql:
                    html += f"<pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:10px; border-radius:4px; border:1px solid #eee;\"><code>{example_sql}</code></pre>"
                if cat.get('questions'):
                    html += "<ul>"
                    for qtext in cat['questions']:
                        html += f"<li>{qtext}</li>"
                    html += "</ul>"
                if cat.get('recommendation'):
                    html += f"<p><strong>Recommendation:</strong> {cat['recommendation']}</p>"

        return html

    def _send_email_notification(self, results: dict) -> None:
        """
        Send email notification with validation results.

        Args:
            results (dict): Combined validation results
        """
        output_dir = self.validation_config.get('reporting', {}).get('output_dir', 'output')

        # Build summary for email body
        summary = {}

        # Add chatbot testing summary
        if results.get('chatbot_testing') and 'summary' in results['chatbot_testing']:
            chatbot_summary = results['chatbot_testing']['summary']
            summary['total_tests'] = chatbot_summary.get('total_tests', 0)
            summary['execution_success_rate'] = chatbot_summary.get('execution_success_rate', 0)
            summary['average_score'] = chatbot_summary.get('average_score', 0)
            summary['interpretation_pass_rate'] = chatbot_summary.get('interpretation_evaluation', {}).get('interpretation_pass_rate', 0)

        # Add scraping validation summary
        if results.get('scraping_validation') and 'summary' in results['scraping_validation']:
            scraping_summary = results['scraping_validation']['summary']
            summary['total_failures'] = scraping_summary.get('total_failures', 0)
            summary['whitelist_failures'] = scraping_summary.get('whitelist_failures', 0)

        # Add overall status
        summary['overall_status'] = results['overall_status']
        summary['timestamp'] = results['timestamp']

        # Collect attachment paths — only include the comprehensive HTML report
        html_report = os.path.join(output_dir, 'comprehensive_test_report.html')
        attachments = [html_report] if os.path.exists(html_report) else []

        # Send email
        logging.info("Attempting to send email notification...")
        success = send_report_email(
            report_summary=summary,
            attachment_paths=attachments,
            test_type="Pre-Commit Validation"
        )

        if success:
            logging.info("✓ Email notification sent successfully")
        else:
            logging.info("Email notification skipped (not configured or failed)")


def main():
    """Main entry point for standalone execution."""
    # Setup logging
    setup_logging('validation_tests')

    try:
        # Run validations
        runner = ValidationTestRunner()
        results = runner.run_all_validations()

        # Exit with appropriate code
        if results['overall_status'] == 'FAIL':
            logging.error("Validation tests FAILED")
            sys.exit(1)
        elif results['overall_status'] == 'WARNING':
            logging.warning("Validation tests completed with WARNINGS")
            sys.exit(0)  # Don't block pipeline
        else:
            logging.info("Validation tests PASSED")
            sys.exit(0)

    except Exception as e:
        logging.error(f"Fatal error in validation tests: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
