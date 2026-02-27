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
import csv
from collections import Counter
from datetime import datetime, timedelta

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

    alias_data = summarize_address_alias_audit(results.get('config', {}))
    llm_data = summarize_llm_provider_activity(results.get('timestamp'), results.get('config', {}))
    reliability_scorecard = summarize_reliability_scorecard(results, llm_data, alias_data)
    reliability_gates = evaluate_reliability_gates(reliability_scorecard, results.get('config', {}))
    reliability_issues = extract_reliability_issues(results, llm_data, alias_data, reliability_gates)
    optimization_plan = build_optimization_plan(results, reliability_scorecard, llm_data)
    output_dir = os.path.dirname(output_path)
    trend_summary = update_and_summarize_reliability_history(output_dir, reliability_scorecard)
    reliability_issues, registry_summary = update_reliability_issue_registry(output_dir, reliability_issues)
    action_queue = build_action_queue(reliability_gates, optimization_plan, reliability_issues, registry_summary)

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

    <h2>Reliability Scorecard</h2>
    {build_reliability_scorecard_html(reliability_scorecard, reliability_issues, reliability_gates, trend_summary, registry_summary)}

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

    # Address Alias Audit
    html += "<h2>Address Alias Audit</h2>"
    html += build_address_alias_audit_html(alias_data)

    # LLM Provider Activity
    html += "<h2>LLM Provider Activity</h2>"
    html += build_llm_provider_activity_html(llm_data)

    # Optimization Recommendations
    html += "<h2>Optimization Recommendations</h2>"
    html += build_optimization_html(optimization_plan)
    html += "<h2>Reliability Action Queue</h2>"
    html += build_action_queue_html(action_queue)

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
    with open(os.path.join(output_dir, 'reliability_scorecard.json'), 'w', encoding='utf-8') as f:
        json.dump(reliability_scorecard, f, indent=2)
    with open(os.path.join(output_dir, 'reliability_issues.json'), 'w', encoding='utf-8') as f:
        json.dump({"issues": reliability_issues}, f, indent=2)
    with open(os.path.join(output_dir, 'reliability_gates.json'), 'w', encoding='utf-8') as f:
        json.dump(reliability_gates, f, indent=2)
    with open(os.path.join(output_dir, 'reliability_optimization.json'), 'w', encoding='utf-8') as f:
        json.dump(optimization_plan, f, indent=2)
    with open(os.path.join(output_dir, 'reliability_action_queue.json'), 'w', encoding='utf-8') as f:
        json.dump(action_queue, f, indent=2)
    logging.info("HTML report saved: %s", output_path)


def escape_html(value: str) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def summarize_address_alias_audit(config: dict) -> dict:
    output_cfg = config.get('output', {}) if isinstance(config, dict) else {}
    validation_cfg = config.get('testing', {}).get('validation', {}) if isinstance(config, dict) else {}
    csv_path = output_cfg.get('address_alias_audit', 'output/address_alias_hits.csv')
    days_window = int(validation_cfg.get('reporting', {}).get('address_alias_audit_days', 14) or 14)

    summary = {
        'available': False,
        'path': csv_path,
        'days_window': days_window,
        'rows_analyzed': 0,
        'decision_counts': {},
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
    rule_decision_counts: Counter = Counter()
    daily_counts: Counter = Counter()
    candidate_counts: Counter = Counter()

    for row in rows:
        day = extract_day(row.get('timestamp', ''))
        decision = row.get('decision', '') or '(blank)'
        rule_name = row.get('rule_name', '') or '(blank)'
        candidate = row.get('candidate', '') or '(blank)'

        decision_counts.update([decision])
        rule_decision_counts.update([f"{rule_name} | {decision}"])
        daily_counts.update([day])
        candidate_counts.update([candidate])

    summary['available'] = True
    summary['rows_analyzed'] = len(rows)
    summary['decision_counts'] = dict(decision_counts)
    summary['rule_decision_counts'] = rule_decision_counts.most_common(10)
    summary['daily_counts'] = daily_counts.most_common(14)
    summary['top_candidates'] = candidate_counts.most_common(8)
    return summary


def build_address_alias_audit_html(alias_data: dict) -> str:
    if not alias_data:
        return "<p class='error-box'>❌ Address alias audit summary unavailable</p>"

    if not alias_data.get('available'):
        return (
            "<p class='error-box'>"
            f"⚠ Address alias audit not available.<br>{escape_html(alias_data.get('error', 'unavailable'))}<br>"
            f"<strong>Path:</strong> {escape_html(alias_data.get('path', ''))}</p>"
        )

    decision_counts = alias_data.get('decision_counts', {})
    html = (
        "<div class=\"metric-container\">"
        f"<div class=\"metric\"><div class=\"metric-value\">{int(alias_data.get('rows_analyzed', 0))}</div>"
        "<div class=\"metric-label\">Rows Analyzed (Recent Window)</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{int(decision_counts.get('applied', 0))}</div>"
        "<div class=\"metric-label\">Alias Matches Applied</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{int(decision_counts.get('skipped_conflict', 0))}</div>"
        "<div class=\"metric-label\">Conflict Skips</div></div>"
        "</div>"
    )

    if alias_data.get('daily_counts'):
        html += "<h3>Daily Volume</h3><table><tr><th>Day</th><th>Count</th></tr>"
        for day, count in alias_data['daily_counts']:
            html += f"<tr><td>{escape_html(day)}</td><td>{count}</td></tr>"
        html += "</table>"

    if alias_data.get('rule_decision_counts'):
        html += "<h3>Top Rule + Decision Pairs</h3><table><tr><th>Rule | Decision</th><th>Count</th></tr>"
        for key, count in alias_data['rule_decision_counts']:
            html += f"<tr><td>{escape_html(key)}</td><td>{count}</td></tr>"
        html += "</table>"

    if alias_data.get('top_candidates'):
        html += "<h3>Top Candidate Text</h3><table><tr><th>Candidate</th><th>Count</th></tr>"
        for candidate, count in alias_data['top_candidates']:
            html += f"<tr><td>{escape_html(candidate)}</td><td>{count}</td></tr>"
        html += "</table>"

    html += (
        f"<p><strong>Audit Source:</strong> {escape_html(alias_data.get('path', ''))}"
        f" (window: last {alias_data.get('days_window', 14)} day(s) present)</p>"
    )
    return html


def parse_log_timestamp(line: str) -> datetime | None:
    try:
        return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def summarize_llm_provider_activity(report_timestamp: str | None, config: dict) -> dict:
    validation_cfg = config.get('testing', {}).get('validation', {}) if isinstance(config, dict) else {}
    hours_window = int(validation_cfg.get('reporting', {}).get('llm_activity_hours', 24) or 24)

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
        p: {"attempts": 0, "successes": 0, "failures": 0, "timeouts": 0, "rate_limits": 0}
        for p in providers
    }
    file_attempts: Counter = Counter()
    model_attempts: Counter = Counter()
    total_provider_exhausted = 0
    lines_scanned = 0

    for path in log_files:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    ts = parse_log_timestamp(line)
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

                    if "query_llm(): openai response received" in low:
                        stats["openai"]["successes"] += 1
                    if "query_llm(): openrouter response received" in low:
                        stats["openrouter"]["successes"] += 1
                    if "query_llm(): mistral response received" in low:
                        stats["mistral"]["successes"] += 1
                    if "query_llm(): gemini response received" in low:
                        stats["gemini"]["successes"] += 1

                    if "query_llm(): openai query failed" in low:
                        stats["openai"]["failures"] += 1
                    if "query_llm(): openrouter query failed" in low:
                        stats["openrouter"]["failures"] += 1
                    if "query_llm(): mistral query failed" in low:
                        stats["mistral"]["failures"] += 1
                    if "query_llm(): gemini query failed" in low:
                        stats["gemini"]["failures"] += 1

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
            logging.warning("summarize_llm_provider_activity: Failed to parse %s: %s", path, e)

    total_attempts = sum(v["attempts"] for v in stats.values())
    total_rate_limits = sum(v["rate_limits"] for v in stats.values())
    total_timeouts = sum(v["timeouts"] for v in stats.values())
    gemini_attempts = int(stats.get("gemini", {}).get("attempts", 0))

    thresholds_cfg = validation_cfg.get('reporting', {}).get('llm_activity_thresholds', {})
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


def build_llm_provider_activity_html(llm_data: dict) -> str:
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

    html = (
        "<div class=\"metric-container\">"
        f"<div class=\"metric\"><div class=\"metric-value\">{total_attempts}</div><div class=\"metric-label\">LLM Attempts (Window)</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{exhausted}</div><div class=\"metric-label\">All-Provider Exhausted</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{lines_scanned}</div><div class=\"metric-label\">Log Lines Scanned</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{total_rate_limits}</div><div class=\"metric-label\">Rate Limits (All Providers)</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{total_timeouts}</div><div class=\"metric-label\">Timeouts (All Providers)</div></div>"
        "</div>"
    )
    html += (
        "<p><strong>Cost Pressure:</strong> "
        f"<span class=\"{pressure_class}\">{escape_html(pressure_level)}</span></p>"
    )
    reasons = llm_data.get("pressure_reasons", [])
    if reasons:
        html += "<ul>"
        for reason in reasons:
            html += f"<li>{escape_html(reason)}</li>"
        html += "</ul>"
    html += (
        f"<p><strong>Window:</strong> {escape_html(llm_data.get('start_ts', ''))} to {escape_html(llm_data.get('end_ts', ''))} "
        f"({int(llm_data.get('window_hours', 24))}h)</p>"
    )

    html += "<h3>Provider Breakdown</h3><table><tr><th>Provider</th><th>Attempts</th><th>Successes</th><th>Failures</th><th>Timeouts</th><th>Rate Limits</th></tr>"
    providers = llm_data.get("providers", {})
    for provider in ("openai", "openrouter", "mistral", "gemini"):
        p = providers.get(provider, {})
        html += (
            "<tr>"
            f"<td>{provider}</td><td>{int(p.get('attempts', 0))}</td><td>{int(p.get('successes', 0))}</td>"
            f"<td>{int(p.get('failures', 0))}</td><td>{int(p.get('timeouts', 0))}</td><td>{int(p.get('rate_limits', 0))}</td>"
            "</tr>"
        )
    html += "</table>"

    top_files = llm_data.get("top_files", [])
    if top_files:
        html += "<h3>Top Log Files by LLM Attempts</h3><table><tr><th>Log File</th><th>Attempts</th></tr>"
        for path, attempts in top_files:
            html += f"<tr><td>{escape_html(os.path.basename(path))}</td><td>{int(attempts)}</td></tr>"
        html += "</table>"

    top_models = llm_data.get("top_models", [])
    if top_models:
        html += "<h3>Top Provider/Model Attempts</h3><table><tr><th>Provider:Model</th><th>Attempts</th></tr>"
        for model_key, attempts in top_models:
            html += f"<tr><td>{escape_html(model_key)}</td><td>{int(attempts)}</td></tr>"
        html += "</table>"

    return html


def summarize_reliability_scorecard(results: dict, llm_data: dict, alias_data: dict) -> dict:
    """Build baseline reliability metrics from report-driving data."""
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


def extract_reliability_issues(results: dict, llm_data: dict, alias_data: dict, gates: dict | None = None) -> list[dict]:
    """Normalize reliability signals into machine-readable issue records."""
    now_ts = str(results.get("timestamp") or datetime.now().isoformat())
    issues: list[dict] = []

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

    gate_payload = gates or {}
    for gate in (gate_payload.get('failed_gates') or []):
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


def build_reliability_scorecard_html(
    scorecard: dict,
    issues: list[dict],
    gates: dict | None = None,
    trends: dict | None = None,
    registry_summary: dict | None = None,
) -> str:
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

    html = (
        "<div class=\"metric-container\">"
        f"<div class=\"metric\"><div class=\"metric-value\">{float(scorecard.get('score', 0)):.1f}</div><div class=\"metric-label\">Reliability Score</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{escape_html(status)}</div><div class=\"metric-label\">Reliability Status</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{open_issues}</div><div class=\"metric-label\">Normalized Open Issues</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{int(metrics.get('llm_attempts', 0))}</div><div class=\"metric-label\">LLM Attempts (Window)</div></div>"
        f"<div class=\"metric\"><div class=\"metric-value\">{int(metrics.get('llm_rate_limits', 0))}</div><div class=\"metric-label\">LLM Rate Limits</div></div>"
        "</div>"
    )
    html += (
        "<p><strong>Status:</strong> "
        f"<span class=\"{status_class}\">{escape_html(status)}</span></p>"
    )
    html += "<h3>Core Reliability Metrics</h3>"
    html += "<table><tr><th>Metric</th><th>Value</th></tr>"
    for key in (
        "chatbot_average_score",
        "chatbot_execution_success_rate",
        "llm_success_rate",
        "llm_provider_exhausted",
        "llm_cost_pressure",
        "address_alias_conflict_skips",
    ):
        html += f"<tr><td>{escape_html(key)}</td><td>{escape_html(metrics.get(key, ''))}</td></tr>"
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
        f"<span class=\"{gate_status_class}\">{escape_html(gate_status)}</span></p>"
    )
    gate_results = gate_payload.get("gate_results", []) or []
    if gate_results:
        html += "<table><tr><th>Gate</th><th>Status</th><th>Detail</th></tr>"
        for gate in gate_results:
            html += (
                "<tr>"
                f"<td>{escape_html(gate.get('name', ''))}</td>"
                f"<td>{escape_html(gate.get('status', ''))}</td>"
                f"<td>{escape_html(gate.get('detail', ''))}</td>"
                "</tr>"
            )
        html += "</table>"

    if issues:
        html += "<h3>Top Reliability Issues</h3>"
        html += "<table><tr><th>ID</th><th>Category</th><th>Severity</th><th>Step</th><th>Provider</th><th>Occurrences</th><th>Actual</th></tr>"
        for issue in issues[:20]:
            html += (
                "<tr>"
                f"<td>{escape_html(issue.get('issue_id', ''))}</td>"
                f"<td>{escape_html(issue.get('category', ''))}</td>"
                f"<td>{escape_html(issue.get('severity', ''))}</td>"
                f"<td>{escape_html(issue.get('step', ''))}</td>"
                f"<td>{escape_html(issue.get('provider', ''))}</td>"
                f"<td>{int(issue.get('occurrence_count', 1) or 1)}</td>"
                f"<td>{escape_html(issue.get('actual', ''))}</td>"
                "</tr>"
            )
        html += "</table>"
    else:
        html += "<p>✅ No normalized reliability issues in current window.</p>"

    trends_payload = trends or {}
    if trends_payload:
        html += "<h3>Trend Snapshot</h3>"
        html += "<table><tr><th>Window</th><th>Runs</th><th>Average Score</th><th>Latest Score</th><th>Delta</th></tr>"
        for window in ("7d", "30d"):
            t = trends_payload.get(window, {}) or {}
            html += (
                "<tr>"
                f"<td>{window}</td>"
                f"<td>{int(t.get('runs', 0) or 0)}</td>"
                f"<td>{float(t.get('average_score', 0) or 0):.1f}</td>"
                f"<td>{float(t.get('latest_score', 0) or 0):.1f}</td>"
                f"<td>{float(t.get('delta_latest_vs_avg', 0) or 0):+.1f}</td>"
                "</tr>"
            )
        html += "</table>"

    reg = registry_summary or {}
    if reg:
        html += (
            f"<p><strong>Issue Registry:</strong> total_tracked={int(reg.get('total_tracked', 0) or 0)}, "
            f"recurring_in_current_run={int(reg.get('recurring_in_current_run', 0) or 0)}</p>"
        )
    return html


def evaluate_reliability_gates(scorecard: dict, config: dict) -> dict:
    reporting_cfg = (
        (config or {}).get('testing', {}).get('validation', {}).get('reporting', {})
        if isinstance(config, dict) else {}
    )
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
        gate_results.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": f"actual={actual} min_required={minimum}"})

    def check_max(name: str, actual: float, maximum: float) -> None:
        ok = actual <= maximum
        gate_results.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": f"actual={actual} max_allowed={maximum}"})

    check_min("min_reliability_score", float((scorecard or {}).get("score", 0) or 0), float(merged_thresholds["min_reliability_score"]))
    check_min("min_chatbot_execution_success_rate", float(metrics.get("chatbot_execution_success_rate", 0) or 0), float(merged_thresholds["min_chatbot_execution_success_rate"]))
    check_min("min_chatbot_average_score", float(metrics.get("chatbot_average_score", 0) or 0), float(merged_thresholds["min_chatbot_average_score"]))
    check_max("max_llm_rate_limits", float(metrics.get("llm_rate_limits", 0) or 0), float(merged_thresholds["max_llm_rate_limits"]))
    check_max("max_llm_timeouts", float(metrics.get("llm_timeouts", 0) or 0), float(merged_thresholds["max_llm_timeouts"]))
    check_max("max_llm_provider_exhausted", float(metrics.get("llm_provider_exhausted", 0) or 0), float(merged_thresholds["max_llm_provider_exhausted"]))

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


def build_optimization_plan(results: dict, scorecard: dict, llm_data: dict) -> dict:
    config = results.get("config", {}) if isinstance(results, dict) else {}
    providers = (llm_data or {}).get("providers", {}) or {}
    default_order = list((config.get("llm", {}) if isinstance(config, dict) else {}).get("chatbot_provider_order", ["openai", "openrouter", "gemini"]))
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
            recommendations.append(f"{provider}: high rate-limit ratio ({rate_limits}/{attempts}); reduce priority or increase cooldown.")
        if attempts > 0 and timeout_rate >= 0.15:
            recommendations.append(f"{provider}: elevated timeout ratio ({timeouts}/{attempts}); keep as fallback until stable.")

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
        "config_patch_preview": {"llm": {"chatbot_provider_order": optimized_order}},
    }


def build_optimization_html(optimization_plan: dict) -> str:
    if not optimization_plan:
        return "<p class='error-box'>❌ Optimization recommendations unavailable</p>"
    current_order = optimization_plan.get("current_chatbot_provider_order", [])
    recommended_order = optimization_plan.get("recommended_chatbot_provider_order", [])
    html = (
        f"<p><strong>Current chatbot provider order:</strong> {escape_html(' -> '.join(current_order))}</p>"
        f"<p><strong>Recommended order:</strong> {escape_html(' -> '.join(recommended_order))}</p>"
    )
    html += "<h3>Provider Health Scores</h3>"
    html += "<table><tr><th>Provider</th><th>Attempts</th><th>Success Rate</th><th>Rate Limit Rate</th><th>Timeout Rate</th><th>Health Score</th></tr>"
    provider_scores = optimization_plan.get("provider_scores", {}) or {}
    for provider in ("openai", "openrouter", "mistral", "gemini"):
        stats = provider_scores.get(provider, {}) or {}
        html += (
            "<tr>"
            f"<td>{escape_html(provider)}</td>"
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
            html += f"<li>{escape_html(rec)}</li>"
        html += "</ul>"
    html += (
        "<p><strong>Patch Preview:</strong> "
        "<code>llm.chatbot_provider_order</code> "
        f"→ {escape_html(str(recommended_order))}</p>"
    )
    return html


def build_action_queue_html(action_queue: dict) -> str:
    if not action_queue:
        return "<p class='error-box'>❌ Action queue unavailable</p>"
    items = action_queue.get("items", []) or []
    if not items:
        return "<p>✅ No queued reliability actions.</p>"
    html = "<table><tr><th>Priority</th><th>Action</th><th>Reason</th><th>Suggested Change</th></tr>"
    for item in items:
        html += (
            "<tr>"
            f"<td>{escape_html(item.get('priority', ''))}</td>"
            f"<td>{escape_html(item.get('title', ''))}</td>"
            f"<td>{escape_html(item.get('reason', ''))}</td>"
            f"<td>{escape_html(item.get('suggested_change', ''))}</td>"
            "</tr>"
        )
    html += "</table>"
    return html


def update_and_summarize_reliability_history(output_dir: str, scorecard: dict) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "reliability_history.jsonl")
    record = {
        "timestamp": scorecard.get("timestamp", datetime.now().isoformat()),
        "score": float(scorecard.get("score", 0) or 0),
        "status": scorecard.get("status", "WATCH"),
    }
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    rows: list[dict] = []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        rows = [record]

    now = datetime.now()
    summary: dict[str, dict] = {}
    for days, label in ((7, "7d"), (30, "30d")):
        cutoff = now - timedelta(days=days)
        scoped = []
        for row in rows:
            ts = str(row.get("timestamp", "") or "")
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            if dt >= cutoff:
                scoped.append(float(row.get("score", 0) or 0))
        avg_score = (sum(scoped) / len(scoped)) if scoped else 0.0
        latest_score = float(record.get("score", 0) or 0)
        summary[label] = {
            "runs": len(scoped),
            "average_score": round(avg_score, 2),
            "latest_score": round(latest_score, 2),
            "delta_latest_vs_avg": round(latest_score - avg_score, 2) if scoped else 0.0,
        }
    summary["path"] = history_path
    return summary


def update_reliability_issue_registry(output_dir: str, issues: list[dict]) -> tuple[list[dict], dict]:
    os.makedirs(output_dir, exist_ok=True)
    registry_path = os.path.join(output_dir, "reliability_issue_registry.json")
    payload = {"issues": {}, "updated_at": datetime.now().isoformat()}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("issues"), dict):
                    payload = loaded
        except Exception:
            pass

    registry = payload.get("issues", {})
    now_ts = datetime.now().isoformat()
    for issue in issues:
        key = f"{issue.get('issue_id', '')}|{issue.get('input_signature', '')}"
        entry = registry.get(key, {})
        first_seen = entry.get("first_seen", issue.get("first_seen", now_ts))
        occurrence = int(entry.get("occurrence_count", 0) or 0) + 1
        registry[key] = {
            "issue_id": issue.get("issue_id", ""),
            "category": issue.get("category", ""),
            "severity": issue.get("severity", ""),
            "step": issue.get("step", ""),
            "provider": issue.get("provider", ""),
            "input_signature": issue.get("input_signature", ""),
            "first_seen": first_seen,
            "last_seen": now_ts,
            "occurrence_count": occurrence,
            "latest_actual": issue.get("actual", ""),
        }
        issue["occurrence_count"] = occurrence

    payload["issues"] = registry
    payload["updated_at"] = now_ts
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    recurring_current = sum(1 for i in issues if int(i.get("occurrence_count", 1) or 1) > 1)
    top_recurring = sorted(
        registry.values(),
        key=lambda x: int(x.get("occurrence_count", 0) or 0),
        reverse=True
    )[:10]
    return issues, {
        "path": registry_path,
        "total_tracked": len(registry),
        "recurring_in_current_run": recurring_current,
        "top_recurring": top_recurring,
    }


def build_action_queue(gates: dict, optimization_plan: dict, issues: list[dict], registry_summary: dict) -> dict:
    items: list[dict] = []
    for gate in (gates or {}).get("failed_gates", []) or []:
        items.append({
            "priority": "P0",
            "title": f"Fix gate failure: {gate.get('name', '')}",
            "reason": gate.get("detail", "Gate threshold breached."),
            "suggested_change": "Adjust provider routing/threshold regressions and rerun validation.",
        })
    for recurring in (registry_summary or {}).get("top_recurring", [])[:3]:
        occ = int(recurring.get("occurrence_count", 0) or 0)
        if occ < 2:
            continue
        items.append({
            "priority": "P1",
            "title": f"Resolve recurring issue: {recurring.get('issue_id', '')}",
            "reason": f"Recurring {occ} times since {recurring.get('first_seen', '')}.",
            "suggested_change": "Add deterministic regression test and tighten parser/fallback for this signature.",
        })
    for rec in (optimization_plan or {}).get("recommendations", [])[:3]:
        items.append({
            "priority": "P2",
            "title": "Apply optimization recommendation",
            "reason": rec,
            "suggested_change": "Review config_patch_preview and apply routing change if aligned with quality goals.",
        })
    dedup: dict[str, dict] = {}
    for item in items:
        title = str(item.get("title", "") or "")
        if title and title not in dedup:
            dedup[title] = item
    return {
        "generated_at": datetime.now().isoformat(),
        "issue_count": len(issues or []),
        "items": list(dedup.values())[:12],
    }


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
        'overall_status': 'PASS',
        'config': config
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
