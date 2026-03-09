#!/usr/bin/env python3
"""
Build a section-aligned remediation plan from comprehensive report artifacts.

This is a read-only planning step. It does not modify source code.
Outputs:
- output/remediation_plan.json
- output/remediation_plan.md
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_report_sections(html_path: str) -> List[str]:
    """Extract section headings from comprehensive_test_report.html."""
    if not os.path.exists(html_path):
        return []
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
    except Exception:
        return []

    sections: List[str] = []
    for raw in re.findall(r"<h2>\s*(.*?)\s*</h2>", html, flags=re.IGNORECASE | re.DOTALL):
        clean = re.sub(r"<[^>]+>", "", raw).strip()
        clean = re.sub(r"^\d+\.\s*", "", clean)
        if clean:
            sections.append(clean)
    return sections


def first_matching_issue(issues_by_step: Dict[str, List[dict]], step: str) -> dict:
    rows = issues_by_step.get(step, [])
    return rows[0] if rows else {}


def make_section_assessment(
    name: str,
    needs_remediation: bool,
    priority: str,
    issue_understanding: str,
    fix_plan: List[str],
    validation: str,
    evidence: List[str],
) -> dict:
    return {
        "section": name,
        "needs_remediation": needs_remediation,
        "priority": priority if needs_remediation else "NONE",
        "issue_understanding": issue_understanding,
        "fix_plan": fix_plan if needs_remediation else [],
        "validation": validation if needs_remediation else "No remediation required for this section.",
        "evidence": evidence,
    }


def _step_to_section(step: str) -> str:
    mapping = {
        "scraping_validation": "Scraping Validation",
        "chatbot_testing": "Chatbot Testing",
        "chatbot_performance": "Chatbot Performance",
        "llm_activity": "LLM Provider Activity",
        "llm_extraction_quality": "LLM Extraction Quality Scorecard",
        "scraper_network_health": "Scraper Network Reliability",
        "fb_block_health": "Facebook Block Health",
        "fb.py": "Facebook Block Health",
        "address_alias_audit": "Address Alias Audit",
        "reliability_gates": "Reliability Scorecard",
        "fb_ig_funnel": "FB/IG URL Funnel",
    }
    return mapping.get(step, "Reliability Scorecard")


def _parse_iso_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _compute_section_issue_trends(issue_rows: List[dict]) -> Dict[str, dict]:
    now = datetime.now()
    trends: Dict[str, dict] = {}
    for row in issue_rows:
        section = _step_to_section(str(row.get("step", "") or ""))
        occ = int(row.get("occurrence_count", 1) or 1)
        first_seen_raw = str(row.get("first_seen", "") or "")
        first_seen_dt = _parse_iso_ts(first_seen_raw)
        unresolved_age_days = 0.0
        if first_seen_dt:
            unresolved_age_days = max(0.0, (now - first_seen_dt).total_seconds() / 86400.0)
        # Trend score boosts persistent, frequently recurring issues.
        trend_score = float(occ) + min(unresolved_age_days, 30.0) * 0.25
        existing = trends.get(section, {
            "issue_count": 0,
            "max_occurrence_count": 0,
            "max_unresolved_age_days": 0.0,
            "trend_score": 0.0,
        })
        existing["issue_count"] = int(existing["issue_count"]) + 1
        existing["max_occurrence_count"] = max(int(existing["max_occurrence_count"]), occ)
        existing["max_unresolved_age_days"] = max(float(existing["max_unresolved_age_days"]), unresolved_age_days)
        existing["trend_score"] = max(float(existing["trend_score"]), trend_score)
        trends[section] = existing
    return trends


def assess_sections(output_dir: str, report_sections: List[str]) -> List[dict]:
    scorecard = load_json(os.path.join(output_dir, "reliability_scorecard.json"))
    gates = load_json(os.path.join(output_dir, "reliability_gates.json"))
    issues = load_json(os.path.join(output_dir, "reliability_issues.json"))
    optimization = load_json(os.path.join(output_dir, "reliability_optimization.json"))
    action_queue = load_json(os.path.join(output_dir, "reliability_action_queue.json"))
    scraping = load_json(os.path.join(output_dir, "scraping_validation_report.json"))
    chatbot = load_json(os.path.join(output_dir, "chatbot_evaluation_report.json"))
    llm_quality = load_json(os.path.join(output_dir, "llm_extraction_quality.json"))
    chatbot_performance = load_json(os.path.join(output_dir, "chatbot_performance.json"))
    suspicious_deletes = load_json(os.path.join(output_dir, "suspicious_deletes.json"))

    issue_rows = issues.get("issues", []) if isinstance(issues.get("issues"), list) else []
    section_issue_trends = _compute_section_issue_trends(issue_rows)
    issues_by_step: Dict[str, List[dict]] = {}
    for row in issue_rows:
        step = str(row.get("step", "")).strip()
        issues_by_step.setdefault(step, []).append(row)

    scraping_summary = scraping.get("summary", {}) if isinstance(scraping.get("summary"), dict) else {}
    chatbot_summary = chatbot.get("summary", {}) if isinstance(chatbot.get("summary"), dict) else {}
    failed_gates = gates.get("failed_gates", []) if isinstance(gates.get("failed_gates"), list) else []
    action_items = action_queue.get("items", []) if isinstance(action_queue.get("items"), list) else []
    recommendations = optimization.get("recommendations", []) if isinstance(optimization.get("recommendations"), list) else []

    assessments: List[dict] = []
    for section_name in report_sections:
        key = section_name.lower().strip()

        if key == "reliability scorecard":
            needs = str(scorecard.get("status", "UNKNOWN")).upper() in {"AT_RISK", "FAIL"}
            failed = [g.get("name", "unknown") for g in failed_gates]
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P0",
                    issue_understanding=(
                        f"Reliability score is {scorecard.get('score', 0)} with status "
                        f"{scorecard.get('status', 'UNKNOWN')}. "
                        f"Failed gates: {', '.join(failed) if failed else 'none'}."
                    ),
                    fix_plan=[
                        "Resolve each failed gate by addressing the underlying metric driver (coverage, exception rate, timeout count).",
                        "Keep request-level telemetry visible but grade pass/fail only on agreed URL-level denominators.",
                        "Re-run validation and confirm both score and gate status improve.",
                    ],
                    validation="`reliability_scorecard.status` is HEALTHY and `reliability_gates.status` is PASS.",
                    evidence=[
                        f"score={scorecard.get('score', 0)}",
                        f"status={scorecard.get('status', 'UNKNOWN')}",
                        f"failed_gates={len(failed_gates)}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "scraping validation":
            critical = int(len(scraping.get("critical_failures", []) or []))
            total_failures = int(scraping_summary.get("total_failures", 0) or 0)
            needs = critical > 0
            top_issue = first_matching_issue(issues_by_step, "scraping_validation")
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P0",
                    issue_understanding=(
                        f"Scraping validation reports {critical} critical failure(s) and "
                        f"{total_failures} total failure(s) in-window. "
                        f"Primary signature: {top_issue.get('input_signature', 'n/a')}."
                    ),
                    fix_plan=[
                        "Separate and report skip reasons explicitly: run-limit skip, should_process_url skip, and attempted-but-failed.",
                        "For critical URLs marked irrelevant repeatedly, audit keyword rules and extractor output before final relevance decision.",
                        "Re-test critical URL set and ensure critical failure count returns to threshold.",
                    ],
                    validation="`scraping_validation_report.summary.whitelist_failures` and edge-case critical failures are at/under configured limits.",
                    evidence=[
                        f"critical_failures={critical}",
                        f"total_failures={total_failures}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "chatbot testing":
            avg_score = float(chatbot_summary.get("average_score", 0) or 0)
            exec_rate = float(chatbot_summary.get("execution_success_rate", 0) or 0)
            problem_categories = chatbot.get("problem_categories", []) if isinstance(chatbot.get("problem_categories"), list) else []
            category_gate = chatbot.get("category_gate", {}) if isinstance(chatbot.get("category_gate"), dict) else {}
            category_gate_status = str(category_gate.get("status", "UNKNOWN") or "UNKNOWN").upper()
            needs = len(problem_categories) > 0 or category_gate_status in {"FAIL", "WARNING"}
            top_issue = first_matching_issue(issues_by_step, "chatbot_testing")
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        f"Chatbot quality has avg_score={avg_score:.1f}, exec_rate={exec_rate:.2%}. "
                        f"There are {len(problem_categories)} problem category bucket(s), "
                        f"category_gate={category_gate_status}, "
                        f"top issue signature: {top_issue.get('input_signature', 'n/a')}."
                    ),
                    fix_plan=[
                        "Align prompt policy and SQL generator behavior for failing intent classes (date arithmetic/weekend logic, style filters).",
                        "Add deterministic regression tests for each named problem category example.",
                        "Require category-level pass checks before releasing prompt/routing changes.",
                    ],
                    validation="Problem categories are cleared or reduced to acceptable level and average score remains above threshold.",
                    evidence=[
                        f"average_score={avg_score:.1f}",
                        f"execution_success_rate={exec_rate:.2%}",
                        f"problem_categories={len(problem_categories)}",
                        f"category_gate={category_gate_status}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "chatbot performance":
            status = str(chatbot_performance.get("status", "UNKNOWN") or "UNKNOWN").upper()
            query_latency = chatbot_performance.get("query_latency_ms", {}) if isinstance(chatbot_performance, dict) else {}
            p95_ms = float(query_latency.get("p95", 0.0) or 0.0)
            unfinished = int(chatbot_performance.get("unfinished_request_count", 0) or 0)
            slow_count = len(chatbot_performance.get("slow_requests", []) or []) if isinstance(chatbot_performance, dict) else 0
            needs = status in {"WATCH", "DEGRADED"} or unfinished > 0
            top_issue = first_matching_issue(issues_by_step, "chatbot_performance")
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P0" if status == "DEGRADED" else "P1",
                    issue_understanding=(
                        f"Chatbot performance status={status}, query_p95_ms={p95_ms:.1f}, "
                        f"unfinished_requests={unfinished}, slow_requests={slow_count}. "
                        f"Issue signature: {top_issue.get('input_signature', 'n/a')}."
                    ),
                    fix_plan=[
                        "Tune the slowest stages first (query SQL generation retries, interpretation LLM calls, confirm SQL execution) based on stage p95.",
                        "Review slow-request samples with request_id-linked question and SQL snippets to identify problematic intents/prompts.",
                        "Set and enforce performance budgets (p95 + unfinished-request count) and alert when breached.",
                    ],
                    validation="`chatbot_performance.status` is HEALTHY with query p95 and unfinished-request counts below thresholds.",
                    evidence=[
                        f"status={status}",
                        f"query_p95_ms={p95_ms:.1f}",
                        f"unfinished_request_count={unfinished}",
                        f"slow_requests={slow_count}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "scraper network reliability":
            metric_map = scorecard.get("metrics", {}) if isinstance(scorecard.get("metrics"), dict) else {}
            exception_rate = float(metric_map.get("scraper_exception_rate", 0) or 0)
            timeout_count = int(metric_map.get("scraper_timeout_count", 0) or 0)
            degraded = bool(metric_map.get("scraper_network_degraded", False))
            needs = degraded or exception_rate > 0.20 or timeout_count > 50
            top_issue = first_matching_issue(issues_by_step, "scraper_network_health")
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P0",
                    issue_understanding=(
                        f"Network reliability shows exception_rate={exception_rate:.4f}, "
                        f"timeout_count={timeout_count}, degraded={degraded}. "
                        f"Issue signature: {top_issue.get('input_signature', 'n/a')}."
                    ),
                    fix_plan=[
                        "Keep request-level reliability metrics for observability, but avoid mixing with URL-level grading denominators.",
                        "Tune retry/backoff and per-domain concurrency for high-timeout/high-exception domains.",
                        "Add explicit reporting for attempted URL count versus skipped URL count in this window.",
                    ],
                    validation="Exception and timeout metrics fall below configured thresholds for two consecutive runs.",
                    evidence=[
                        f"scraper_exception_rate={exception_rate:.4f}",
                        f"scraper_timeout_count={timeout_count}",
                        f"scraper_network_degraded={degraded}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "facebook block health":
            fb_issue = first_matching_issue(issues_by_step, "fb_block_health")
            needs = bool(fb_issue)
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        "No direct fb_block_health issue appears in registry."
                        if not needs
                        else f"Facebook block health issue detected: {fb_issue.get('actual', '')}"
                    ),
                    fix_plan=[
                        "Review fb_log run-scoped completion versus block markers and ensure progress ratio messaging is explicit.",
                        "Report attempted/successful FB/IG URL counts and throttle state separately from hard blocks.",
                    ],
                    validation="FB block section reflects run-scoped attempted/success counts and no false block labeling.",
                    evidence=[
                        f"issue_present={needs}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "address alias audit":
            alias_issue = first_matching_issue(issues_by_step, "address_alias_audit")
            needs = bool(alias_issue)
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        "No explicit alias-audit reliability issue detected in this run."
                        if not needs
                        else f"Alias audit issue detected: {alias_issue.get('actual', '')}"
                    ),
                    fix_plan=[
                        "Add stricter safeguards to prevent over-broad alias mapping and surface top candidate collisions.",
                        "Require manual review threshold for low-confidence alias merges before applying to address_id.",
                    ],
                    validation="No high-risk alias collisions and no unexpected venue/address remaps in audit output.",
                    evidence=[
                        f"issue_present={needs}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "llm provider activity":
            metric_map = scorecard.get("metrics", {}) if isinstance(scorecard.get("metrics"), dict) else {}
            cost_pressure = str(metric_map.get("llm_cost_pressure", "UNKNOWN"))
            provider_issue = first_matching_issue(issues_by_step, "llm_activity")
            needs = bool(provider_issue)
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        f"LLM activity shows cost_pressure={cost_pressure}. "
                        f"Provider issue: {provider_issue.get('actual', 'none')}."
                    ),
                    fix_plan=[
                        "Use total LLM calls across all logs as denominator for cost-pressure and provider reliability reporting.",
                        "Tune provider routing/cooldowns using current thresholds and observed rate-limit behavior.",
                        "Keep thresholds in config aligned to current run volume expectations.",
                    ],
                    validation="Provider failure/rate-limit signals trend down and cost-pressure label aligns with configured thresholds.",
                    evidence=[
                        f"llm_cost_pressure={cost_pressure}",
                        f"provider_issue_present={needs}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "llm extraction quality scorecard":
            total_urls = int(llm_quality.get("total_urls", 0) or 0)
            hard_failure_rate = float(llm_quality.get("hard_failure_rate", 0) or 0)
            needs = total_urls > 0 and hard_failure_rate > 0.10
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        f"Extraction quality shows total_urls={total_urls}, "
                        f"hard_failure_rate={hard_failure_rate:.2%}."
                    ),
                    fix_plan=[
                        "Prioritize parser/schema hard-failure signatures by top failing URLs and model breakdown.",
                        "Strengthen fallback path for malformed or truncated model outputs.",
                        "Track parse quality per provider and route away from unstable paths.",
                    ],
                    validation="Hard failure rate remains below 10% and schema error count decreases in next run.",
                    evidence=[
                        f"total_urls={total_urls}",
                        f"hard_failure_rate={hard_failure_rate:.2%}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "optimization recommendations":
            needs = len(recommendations) > 0
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P2",
                    issue_understanding=(
                        f"There are {len(recommendations)} optimization recommendation(s) generated."
                    ),
                    fix_plan=[
                        "Apply only recommendations that directly reduce open high-severity issues.",
                        "Convert accepted recommendations into tracked tasks with owner and acceptance criteria.",
                    ],
                    validation="Accepted recommendations are reflected in config/code and validated in next report.",
                    evidence=[
                        f"recommendations={len(recommendations)}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "reliability action queue":
            needs = len(action_items) > 0
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P0",
                    issue_understanding=(
                        f"Action queue contains {len(action_items)} item(s), "
                        "indicating unresolved reliability work."
                    ),
                    fix_plan=[
                        "Execute P0 items first and tie each item to a measurable metric/gate in the next run.",
                        "Close or downgrade items only after validation evidence confirms improvement.",
                    ],
                    validation="Queue P0 items are resolved and queue size trends downward across runs.",
                    evidence=[
                        f"action_items={len(action_items)}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "fb/ig url funnel":
            fbig_issue = first_matching_issue(issues_by_step, "fb_ig_funnel")
            needs = bool(fbig_issue)
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        "No direct fb_ig_funnel issue in registry."
                        if not needs
                        else f"Funnel issue: {fbig_issue.get('actual', '')}"
                    ),
                    fix_plan=[
                        "Report base denominator explicitly: in-db FB/IG URLs, attempted URLs, successful URLs, URLs with events.",
                        "Use success percentage as `urls_with_events / urls_passed_for_scraping` and keep jail progress ratio separate.",
                    ],
                    validation="Funnel metrics are internally consistent and match user-defined denominator rules.",
                    evidence=[
                        f"issue_present={needs}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        if key == "likely incorrect deletes":
            flagged_count = int(suspicious_deletes.get("flagged_count", 0) or 0)
            needs = flagged_count > 0
            assessments.append(
                make_section_assessment(
                    name=section_name,
                    needs_remediation=needs,
                    priority="P1",
                    issue_understanding=(
                        f"Likely incorrect delete detector flagged {flagged_count} row(s) for review."
                    ),
                    fix_plan=[
                        "Prioritize model-driven/fuzzy/clustered deletes for review; keep exact dedup deletes as volume metric.",
                        "For flagged deletes, compare deleted row vs canonical/kept row before restoring.",
                        "Tune suspicious-delete rules based on false positives and preserve audit context on each delete.",
                    ],
                    validation="Flagged incorrect deletes trend down and spot checks show no critical false deletions.",
                    evidence=[
                        f"flagged_count={flagged_count}",
                        f"exact_duplicate_deletes={int(suspicious_deletes.get('exact_duplicate_deletes', 0) or 0)}",
                        f"fuzzy_model_candidate_deletes={int(suspicious_deletes.get('fuzzy_model_candidate_deletes', 0) or 0)}",
                    ],
                )
            )
            assessments[-1]["trend"] = section_issue_trends.get(section_name, {})
            continue

        # Unknown / newly introduced report section: preserve explicit traceability.
        assessments.append(
            make_section_assessment(
                name=section_name,
                needs_remediation=False,
                priority="NONE",
                issue_understanding="No planner rule is defined yet for this report section.",
                fix_plan=[],
                validation="No remediation generated for unknown section type.",
                evidence=[],
            )
        )
        assessments[-1]["trend"] = section_issue_trends.get(section_name, {})

    return assessments


def build_plan(output_dir: str) -> dict:
    html_path = os.path.join(output_dir, "comprehensive_test_report.html")
    report_sections = parse_report_sections(html_path)

    if not report_sections:
        report_sections = [
            "Reliability Scorecard",
            "Scraping Validation",
            "Chatbot Testing",
            "Chatbot Performance",
            "Scraper Network Reliability",
            "Facebook Block Health",
            "Address Alias Audit",
            "LLM Provider Activity",
            "LLM Extraction Quality Scorecard",
            "Optimization Recommendations",
            "Reliability Action Queue",
        ]

    section_assessments = assess_sections(output_dir, report_sections)
    remediation_sections = [s for s in section_assessments if s.get("needs_remediation")]
    action_queue = load_json(os.path.join(output_dir, "reliability_action_queue.json"))
    action_items = action_queue.get("items", []) if isinstance(action_queue.get("items"), list) else []

    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "NONE": 9}
    remediation_sections = sorted(
        remediation_sections,
        key=lambda row: (
            priority_rank.get(str(row.get("priority", "P2")), 9),
            -float((row.get("trend", {}) or {}).get("trend_score", 0.0) or 0.0),
            -int((row.get("trend", {}) or {}).get("max_occurrence_count", 0) or 0),
            row.get("section", ""),
        ),
    )

    execution_order: List[dict] = []
    for idx, section in enumerate(remediation_sections, start=1):
        section_name = str(section.get("section", "") or "")
        section_actions = [
            item for item in action_items
            if str(item.get("source_section", "") or "").strip().lower() == section_name.strip().lower()
        ]
        execution_order.append({
            "order": idx,
            "section": section_name,
            "priority": section.get("priority", "P2"),
            "trend_score": round(float((section.get("trend", {}) or {}).get("trend_score", 0.0) or 0.0), 2),
            "max_occurrence_count": int((section.get("trend", {}) or {}).get("max_occurrence_count", 0) or 0),
            "max_unresolved_age_days": round(float((section.get("trend", {}) or {}).get("max_unresolved_age_days", 0.0) or 0.0), 2),
            "acceptance_tests": [str(item.get("acceptance_test", "") or "") for item in section_actions if item.get("acceptance_test")],
            "metric_keys": [str(item.get("metric_key", "") or "") for item in section_actions if item.get("metric_key")],
            "action_titles": [str(item.get("title", "") or "") for item in section_actions if item.get("title")],
            "owners": [str(item.get("owner", "") or "") for item in section_actions if item.get("owner")],
            "statuses": [str(item.get("status", "") or "") for item in section_actions if item.get("status")],
        })

    return {
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "report": "comprehensive_test_report.html",
            "scorecard": "reliability_scorecard.json",
            "gates": "reliability_gates.json",
            "issues": "reliability_issues.json",
            "optimization": "reliability_optimization.json",
            "action_queue": "reliability_action_queue.json",
            "scraping_validation": "scraping_validation_report.json",
            "chatbot": "chatbot_evaluation_report.json",
            "chatbot_performance": "chatbot_performance.json",
            "llm_extraction_quality": "llm_extraction_quality.json",
            "suspicious_deletes": "suspicious_deletes.json",
        },
        "summary": {
            "report_sections": len(section_assessments),
            "sections_requiring_remediation": len(remediation_sections),
            "sections_without_remediation": len(section_assessments) - len(remediation_sections),
        },
        "section_review_matrix": section_assessments,
        "remediation_sections": remediation_sections,
        "execution_order": execution_order,
        "next_step": "Implement P0/P1 remediations first, then re-run validation and update this plan.",
    }


def write_markdown(path: str, plan: dict) -> None:
    lines: List[str] = [
        "# Remediation Plan",
        "",
        f"- Generated: {plan.get('generated_at', '')}",
        f"- Report Sections Reviewed: {plan.get('summary', {}).get('report_sections', 0)}",
        f"- Sections Requiring Remediation: {plan.get('summary', {}).get('sections_requiring_remediation', 0)}",
        f"- Sections Without Remediation: {plan.get('summary', {}).get('sections_without_remediation', 0)}",
        "",
        "## Section Review Matrix",
    ]

    for idx, row in enumerate(plan.get("section_review_matrix", []), start=1):
        status = "REMEDIATE" if row.get("needs_remediation") else "NO_ACTION"
        lines.extend(
            [
                f"{idx}. {row.get('section', 'Unknown Section')} [{status}]",
                f"   - Understanding: {row.get('issue_understanding', '')}",
                f"   - Evidence: {', '.join(row.get('evidence', []) or ['n/a'])}",
            ]
        )

    remediation_sections = plan.get("remediation_sections", [])
    lines.extend(["", "## Remediation Details"])
    if not remediation_sections:
        lines.append("No remediation is required based on current report artifacts.")
    else:
        for row in remediation_sections:
            lines.extend(
                [
                    "",
                    f"### {row.get('section', 'Unknown Section')} [{row.get('priority', 'P2')}]",
                    f"Understanding: {row.get('issue_understanding', '')}",
                    "Fix Plan:",
                ]
            )
            for i, step in enumerate(row.get("fix_plan", []), start=1):
                lines.append(f"{i}. {step}")
            lines.append(f"Validation: {row.get('validation', '')}")

    execution_order = plan.get("execution_order", []) if isinstance(plan.get("execution_order"), list) else []
    lines.extend(["", "## Section-Linked Execution Order"])
    if not execution_order:
        lines.append("No section-linked execution order available.")
    else:
        for row in execution_order:
            lines.append(
                f"{int(row.get('order', 0) or 0)}. {row.get('section', 'Unknown Section')} "
                f"[{row.get('priority', 'P2')}] "
                f"(trend_score={row.get('trend_score', 0)}, "
                f"max_occurrence={row.get('max_occurrence_count', 0)}, "
                f"max_age_days={row.get('max_unresolved_age_days', 0)})"
            )
            metric_keys = row.get("metric_keys", []) or []
            acceptance_tests = row.get("acceptance_tests", []) or []
            action_titles = row.get("action_titles", []) or []
            owners = row.get("owners", []) or []
            statuses = row.get("statuses", []) or []
            lines.append(f"   - Metric Keys: {', '.join(metric_keys) if metric_keys else 'n/a'}")
            lines.append(f"   - Actions: {', '.join(action_titles) if action_titles else 'n/a'}")
            lines.append(f"   - Owners: {', '.join(owners) if owners else 'n/a'}")
            lines.append(f"   - Statuses: {', '.join(statuses) if statuses else 'n/a'}")
            lines.append(f"   - Acceptance: {', '.join(acceptance_tests) if acceptance_tests else 'n/a'}")

    lines.extend(["", f"Next Step: {plan.get('next_step', '')}", ""])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    cfg = load_config("config/config.yaml")
    output_dir = (
        cfg.get("testing", {})
        .get("validation", {})
        .get("reporting", {})
        .get("output_dir", "output")
    )
    os.makedirs(output_dir, exist_ok=True)

    plan = build_plan(output_dir)
    json_path = os.path.join(output_dir, "remediation_plan.json")
    md_path = os.path.join(output_dir, "remediation_plan.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    write_markdown(md_path, plan)

    print(f"Remediation plan written: {json_path}")
    print(f"Remediation plan written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
