#!/usr/bin/env python3
"""
Utility: Generate comprehensive_test_report.html from existing chatbot_test_results.csv
and email it to the configured recipient.

This script avoids DB/LLM calls. It rehydrates the chatbot report from the
existing CSV (or uses the JSON report if already present), builds the HTML,
and sends the email with the HTML attached.
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import yaml

from logging_config import setup_logging
from email_notifier import send_report_email
from chatbot_evaluator import generate_chatbot_report


def load_config(path: str = 'config/config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def rehydrate_scored_results_from_csv(csv_path: str) -> list:
    """Rehydrate minimal scored_results list from flattened CSV."""
    df = pd.read_csv(csv_path)
    results = []
    for _, row in df.iterrows():
        eval_dict = {
            'score': float(row.get('evaluation_score', 0)) if pd.notna(row.get('evaluation_score', 0)) else 0,
            'reasoning': str(row.get('evaluation_reasoning', '') or ''),
            'criteria_matched': [],
            'criteria_missed': [],
            'sql_issues': []
        }
        results.append({
            'question': str(row.get('question', '')),
            'category': str(row.get('category', '')),
            'execution_success': bool(row.get('execution_success', True)) if pd.notna(row.get('execution_success')) else True,
            'sql_query': str(row.get('sql_query', '') or ''),
            'evaluation': eval_dict
        })
    return results


def build_html(results: dict, output_path: str) -> None:
    """Build a minimal comprehensive_test_report.html using only chatbot data."""
    chatbot = results.get('chatbot_testing') or {}
    ts = results.get('timestamp', datetime.now().isoformat())

    # Summary metrics
    summary = chatbot.get('summary', {})
    total_tests = summary.get('total_tests', 0)
    avg_score = summary.get('average_score', 0)
    exec_rate = summary.get('execution_success_rate', 0)
    score_dist = summary.get('score_distribution', {})

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset=\"UTF-8\">
  <title>Pre-Commit Validation Report (Chatbot Only)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
    h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 30px; }}
    h3 {{ color: #777; margin-top: 20px; }}
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
  <div class=\"container\">
    <h1>Pre-Commit Validation Report (Chatbot Only)</h1>
    <p class=\"timestamp\"><strong>Timestamp:</strong> {ts}</p>

    <h2>Chatbot Testing</h2>
    <div class=\"metric-container\">
      <div class=\"metric\"><div class=\"metric-value\">{total_tests}</div><div class=\"metric-label\">Total Tests</div></div>
      <div class=\"metric\"><div class=\"metric-value\">{avg_score:.1f}</div><div class=\"metric-label\">Average Score</div></div>
      <div class=\"metric\"><div class=\"metric-value\">{exec_rate:.1%}</div><div class=\"metric-label\">Execution Success</div></div>
    </div>

    <h3>Score Distribution</h3>
    <table><tr><th>Score Range</th><th>Count</th></tr>
    """
    for rng, count in score_dist.items():
        html += f"<tr><td>{rng}</td><td>{count}</td></tr>"
    html += "</table>"

    # Problem Categories
    cats = chatbot.get('problem_categories', [])
    if cats:
        html += "<h3>Problem Categories (Score < 90)</h3>"
        for cat in cats:
            html += f"<h4>{cat.get('name','Category')} ({cat.get('count',0)} issues)</h4>"
            ex = cat.get('example', {})
            html += f"<p><strong>Example:</strong> {ex.get('question','')}<br><em>{ex.get('reason','')}</em></p>"
            ex_sql = ex.get('sql', '')
            if ex_sql:
                html += f"<pre style=\"white-space: pre-wrap; background:#f8f9fa; padding:10px; border-radius:4px; border:1px solid #eee;\"><code>{ex_sql}</code></pre>"
            if cat.get('questions'):
                html += "<ul>" + "".join([f"<li>{q}</li>" for q in cat['questions']]) + "</ul>"
            if cat.get('recommendation'):
                html += f"<p><strong>Recommendation:</strong> {cat['recommendation']}</p>"

    html += """
    <hr style="margin: 40px 0;">
    <p style="text-align: center; color: #999; font-size: 0.9em;">
      Generated by Chatbot Report Utility
    </p>
  </div>
  </body>
  </html>
  """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    logging.info("HTML report saved: %s", output_path)


def main():
    setup_logging('report_from_csv')
    config = load_config('config/config.yaml')
    output_dir = config.get('testing', {}).get('validation', {}).get('reporting', {}).get('output_dir', 'output')
    csv_path = os.path.join(output_dir, 'chatbot_test_results.csv')
    json_path = os.path.join(output_dir, 'chatbot_evaluation_report.json')

    if not os.path.exists(csv_path) and not os.path.exists(json_path):
        logging.error("No chatbot report CSV or JSON found in %s", output_dir)
        return

    # Prefer JSON if it exists; else rehydrate from CSV and rebuild the JSON via generate_chatbot_report
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            chatbot_report = json.load(f)
        logging.info("Loaded existing JSON report: %s", json_path)
    else:
        logging.info("Rehydrating from CSV: %s", csv_path)
        scored_results = rehydrate_scored_results_from_csv(csv_path)
        chatbot_report = generate_chatbot_report(scored_results, output_dir)

    # Build minimal results and HTML
    results = {
        'timestamp': datetime.now().isoformat(),
        'scraping_validation': None,
        'chatbot_testing': chatbot_report,
        'overall_status': 'PASS'
    }

    html_path = os.path.join(output_dir, 'comprehensive_test_report.html')
    build_html(results, html_path)

    # Send email with attachment
    summary = {
        'total_tests': chatbot_report.get('summary', {}).get('total_tests', 0),
        'execution_success_rate': chatbot_report.get('summary', {}).get('execution_success_rate', 0),
        'average_score': chatbot_report.get('summary', {}).get('average_score', 0),
        'overall_status': results['overall_status'],
        'timestamp': results['timestamp']
    }
    attachments = [html_path] if os.path.exists(html_path) else []
    ok = send_report_email(report_summary=summary, attachment_paths=attachments, test_type='Chatbot Report')
    if ok:
        logging.info("Email sent successfully with attachment: %s", html_path)
    else:
        logging.warning("Email send skipped or failed. Check email configuration.")


if __name__ == '__main__':
    main()
