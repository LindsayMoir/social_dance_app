#!/usr/bin/env python3
"""
Build a remediation plan from reliability artifacts.

This is a read-only planning step. It does not modify source code.
Outputs:
- output/remediation_plan.json
- output/remediation_plan.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime

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


def build_plan(output_dir: str) -> dict:
    scorecard = load_json(os.path.join(output_dir, "reliability_scorecard.json"))
    gates = load_json(os.path.join(output_dir, "reliability_gates.json"))
    issues = load_json(os.path.join(output_dir, "reliability_issues.json"))
    optimization = load_json(os.path.join(output_dir, "reliability_optimization.json"))
    action_queue = load_json(os.path.join(output_dir, "reliability_action_queue.json"))

    issue_rows = issues.get("issues", []) if isinstance(issues.get("issues"), list) else []
    failed_gates = gates.get("failed_gates", []) if isinstance(gates.get("failed_gates"), list) else []
    recs = optimization.get("recommendations", []) if isinstance(optimization.get("recommendations"), list) else []
    action_items = action_queue.get("items", []) if isinstance(action_queue.get("items"), list) else []

    priorities: list[dict] = []

    for gate in failed_gates:
        priorities.append({
            "priority": "P0",
            "title": f"Resolve failed gate: {gate.get('name', 'unknown')}",
            "why": gate.get("detail", "Reliability threshold breached."),
            "owner": "engineering",
            "validation": "Re-run validation step; gate must pass.",
        })

    high_provider_issues = [
        i for i in issue_rows
        if str(i.get("category", "")) == "Provider Reliability Failure"
        and str(i.get("severity", "")).lower() == "high"
    ]
    for issue in high_provider_issues[:3]:
        priorities.append({
            "priority": "P1",
            "title": f"Stabilize provider: {issue.get('provider', 'unknown')}",
            "why": issue.get("actual", "Provider instability detected."),
            "owner": "engineering",
            "validation": "Verify rate limits/timeouts drop in next run.",
        })

    for rec in recs[:3]:
        priorities.append({
            "priority": "P2",
            "title": "Apply optimization recommendation",
            "why": rec,
            "owner": "engineering",
            "validation": "Observe trend improvement over 7 days.",
        })

    # Fall back to action queue items if no explicit priorities generated.
    if not priorities:
        for item in action_items[:5]:
            priorities.append({
                "priority": item.get("priority", "P2"),
                "title": item.get("title", "Action item"),
                "why": item.get("reason", ""),
                "owner": "engineering",
                "validation": item.get("suggested_change", ""),
            })

    return {
        "generated_at": datetime.now().isoformat(),
        "inputs": {
            "scorecard": "reliability_scorecard.json",
            "gates": "reliability_gates.json",
            "issues": "reliability_issues.json",
            "optimization": "reliability_optimization.json",
            "action_queue": "reliability_action_queue.json",
        },
        "summary": {
            "reliability_score": scorecard.get("score", 0),
            "reliability_status": scorecard.get("status", "UNKNOWN"),
            "gate_status": gates.get("status", "UNKNOWN"),
            "issue_count": len(issue_rows),
            "failed_gates": len(failed_gates),
        },
        "priorities": priorities[:10],
        "next_step": "Implement top priorities with regression tests and commit each change set.",
    }


def write_markdown(path: str, plan: dict) -> None:
    lines = [
        "# Remediation Plan",
        "",
        f"- Generated: {plan.get('generated_at', '')}",
        f"- Reliability Score: {plan.get('summary', {}).get('reliability_score', 0)}",
        f"- Reliability Status: {plan.get('summary', {}).get('reliability_status', 'UNKNOWN')}",
        f"- Gate Status: {plan.get('summary', {}).get('gate_status', 'UNKNOWN')}",
        f"- Issue Count: {plan.get('summary', {}).get('issue_count', 0)}",
        "",
        "## Priorities",
    ]
    for idx, item in enumerate(plan.get("priorities", []), start=1):
        lines.extend([
            f"{idx}. [{item.get('priority', 'P2')}] {item.get('title', '')}",
            f"   - Why: {item.get('why', '')}",
            f"   - Owner: {item.get('owner', 'engineering')}",
            f"   - Validation: {item.get('validation', '')}",
        ])
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
