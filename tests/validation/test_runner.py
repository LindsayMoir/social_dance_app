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

# Add src to path for imports (calculate path relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(repo_root, 'src'))
sys.path.insert(0, script_dir)  # Also add tests/validation for local imports

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv('src/.env')  # Load from src/.env since that's where credentials are stored

from datetime import datetime
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
            'overall_status': 'PASS'
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

        # 3. GENERATE COMPREHENSIVE HTML REPORT
        try:
            if self.validation_config.get('reporting', {}).get('generate_html', True):
                self._generate_html_report(results)
        except Exception as e:
            logging.error(f"HTML report generation failed: {e}", exc_info=True)

        # 4. SEND EMAIL NOTIFICATION (if configured)
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

        <h2>1. Scraping Validation</h2>
        {self._build_scraping_html(results.get('scraping_validation'))}

        <h2>2. Chatbot Testing</h2>
        {self._build_chatbot_html(results.get('chatbot_testing'))}

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

        logging.info(f"HTML report saved: {output_path}")

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
            html += "<table><tr><th>URL</th><th>Source</th><th>Type</th><th>Failure Type</th><th>Recommendation</th></tr>"
            for failure in scraping_data['critical_failures'][:20]:  # Limit to 20
                html += f"""
                <tr class="problematic">
                    <td>{failure['url'][:60]}...</td>
                    <td>{failure['source']}</td>
                    <td>{failure['importance']}</td>
                    <td>{failure['failure_type']}</td>
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
