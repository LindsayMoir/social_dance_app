#!/usr/bin/env python3
"""
Run the comprehensive chatbot report using the CSV input and log the run.

This script intentionally forces rebuilding from `output/chatbot_test_results.csv`
to validate the updated HTML report template, then emails the result using the
existing notification system.
"""

import os
import logging
import sys
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tests', 'validation'))

from logging_config import setup_logging

# Reuse utilities from the existing report tool
from tests.validation.report_from_csv import (
    load_config,
    rehydrate_scored_results_from_csv,
    build_html,
)
from tests.validation.chatbot_evaluator import generate_chatbot_report
from src.email_notifier import send_report_email


def main():
    setup_logging('run_report_from_csv')

    config = load_config('config/config.yaml')
    output_dir = (
        config.get('testing', {})
        .get('validation', {})
        .get('reporting', {})
        .get('output_dir', 'output')
    )

    csv_path = os.path.join(output_dir, 'chatbot_test_results.csv')
    if not os.path.exists(csv_path):
        logging.error("CSV not found at %s. Aborting.", csv_path)
        return 1

    logging.info("Starting report run from CSV: %s", csv_path)

    # Rehydrate and rebuild JSON + HTML strictly from CSV
    scored_results = rehydrate_scored_results_from_csv(csv_path)
    chatbot_report = generate_chatbot_report(scored_results, output_dir)

    results = {
        'timestamp': datetime.now().isoformat(),
        'scraping_validation': None,
        'chatbot_testing': chatbot_report,
        'overall_status': 'PASS',
    }

    html_path = os.path.join(output_dir, 'comprehensive_test_report.html')
    build_html(results, html_path)

    # Attempt to email the report
    summary = {
        'total_tests': chatbot_report.get('summary', {}).get('total_tests', 0),
        'execution_success_rate': chatbot_report.get('summary', {}).get('execution_success_rate', 0),
        'average_score': chatbot_report.get('summary', {}).get('average_score', 0),
        'overall_status': results['overall_status'],
        'timestamp': results['timestamp'],
    }

    attachments = [html_path] if os.path.exists(html_path) else []
    remediation_md = os.path.join(output_dir, 'remediation_plan.md')
    remediation_json = os.path.join(output_dir, 'remediation_plan.json')
    if not (os.path.exists(remediation_md) and os.path.exists(remediation_json)):
        try:
            subprocess.run(
                [sys.executable, "tests/validation/remediation_planner.py"],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except Exception as e:
            logging.warning("Could not generate remediation plan before email: %s", e)
    if os.path.exists(remediation_md):
        attachments.append(remediation_md)
    if os.path.exists(remediation_json):
        attachments.append(remediation_json)
    logging.info("Attempting to send email with %d attachment(s)", len(attachments))
    ok = send_report_email(report_summary=summary, attachment_paths=attachments, test_type='Chatbot Report')
    if ok:
        logging.info("Email sent successfully with attachment: %s", html_path)
    else:
        logging.warning("Email send skipped or failed. Verify .env SMTP config.")

    logging.info("Report run complete. HTML: %s", html_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
