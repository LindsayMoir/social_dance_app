#!/usr/bin/env python3
"""Helpers for parser-first remediation workflow rendering."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ParserWorkflowRenderer:
    """Render parser remediation guidance from replay validation artifacts."""

    escape_html: Callable[[str], str]

    _PARSER_CATEGORIES = frozenset(
        {
            "wrong_date",
            "wrong_time",
            "wrong_location_or_address",
            "wrong_source",
            "no_event_extracted_replay",
            "wrong_replay_event_selection",
            "wrong_replay_source_url",
            "duplicate_event_identity",
            "missing_url_baseline",
            "url_unreachable_replay",
        }
    )

    def guidance_for_category(self, category: str) -> dict[str, str]:
        """Map replay mismatch categories to concrete remediation guidance."""
        normalized = str(category or "").strip().lower()
        guidance_map = {
            "wrong_date": {
                "what_to_do": "Fix date extraction before changing runtime or model routing.",
                "how_to_do": "Open the failed source URL, compare the source text with the Original/Re-scraped rows, then tighten the scraper/prompt or recurring-date guard in the source write path if the weekday/date pair is inconsistent.",
                "where_to_change": "Original scraper/prompt or shared DB write guard",
                "acceptance_test": "The targeted replay row matches on start_date in the next validation report.",
            },
            "wrong_time": {
                "what_to_do": "Fix time parsing and listing-page event selection.",
                "how_to_do": "Inspect the source page for multiple events on the same date. If the wrong row was selected, tighten replay matching in tests/validation/test_runner.py. If the source was parsed incorrectly, tighten the scraper or prompt time extraction path.",
                "where_to_change": "Replay matcher first for listing pages; otherwise original scraper/prompt",
                "acceptance_test": "The targeted replay row matches on start_time in the next validation report.",
            },
            "wrong_location_or_address": {
                "what_to_do": "Fix venue/address resolution using explicit venue tokens before fuzzy matching.",
                "how_to_do": "Compare the venue text on the page with the stored location. Tighten the scraper or DB normalization rule so explicit venue names win over weak address inference.",
                "where_to_change": "Original scraper normalization or DB venue/address resolver",
                "acceptance_test": "The targeted replay row matches on location in the next validation report.",
            },
            "wrong_source": {
                "what_to_do": "Fix source attribution before adding new extraction logic.",
                "how_to_do": "Review how the source field was assigned for the failed URL and tighten source ownership logic so the domain or named source is preserved through write.",
                "where_to_change": "Original scraper/source attribution path",
                "acceptance_test": "The targeted replay row keeps the expected source value in the next validation report.",
            },
            "no_event_extracted_replay": {
                "what_to_do": "Add or tighten a deterministic parser fallback for this page shape.",
                "how_to_do": "Open the failed URL and classify the page type. If the page is low-text, add a site-specific parser, link-follow fallback, or OCR/vision fallback before expanding model usage.",
                "where_to_change": "Original scraper for that page type or fallback extraction path",
                "acceptance_test": "The failed URL produces at least one replay candidate event on the next validation run.",
            },
            "wrong_replay_event_selection": {
                "what_to_do": "Tighten multi-event listing-page replay matching before touching source scrapers.",
                "how_to_do": "Adjust tests/validation/test_runner.py so same-page replay candidates must match on event_name or start_time after date, instead of allowing a same-date wrong event to win.",
                "where_to_change": "Replay matcher in tests/validation/test_runner.py",
                "acceptance_test": "The replay matcher picks the correct event from the same page URL in the next validation report.",
            },
            "wrong_replay_source_url": {
                "what_to_do": "Preserve the replay source URL and stop cross-domain drift during comparison.",
                "how_to_do": "Keep the original replay page URL as the comparison key and store any mentioned external URL separately, so social posts cannot be compared against a different website URL.",
                "where_to_change": "Replay fetch/matcher in tests/validation/test_runner.py",
                "acceptance_test": "The replay row keeps the baseline source URL and no longer fails due to cross-domain drift.",
            },
            "duplicate_event_identity": {
                "what_to_do": "Strengthen duplicate identity keys for this source pattern.",
                "how_to_do": "Review how repeated events from the same source are keyed and tighten identity on URL plus normalized event_name, start_date, and start_time before insert.",
                "where_to_change": "Dedup identity/write path",
                "acceptance_test": "The duplicated replay failure no longer appears and duplicate rate does not increase in the next run.",
            },
            "missing_url_baseline": {
                "what_to_do": "Fix empty source URL persistence before doing parser work.",
                "how_to_do": "Trace the write path for the failed row and enforce a non-empty normalized URL before the event is inserted.",
                "where_to_change": "Original write path before DB insert",
                "acceptance_test": "The baseline row has a non-empty URL and replay can fetch it on the next validation run.",
            },
            "url_unreachable_replay": {
                "what_to_do": "Treat this as a source availability issue, not a parser issue.",
                "how_to_do": "Retry the URL manually, then decide whether the source is dead, blocked, or needs retry/backoff logic. Do not tune parsers until the source is reachable.",
                "where_to_change": "Source availability/retry handling, not parser logic",
                "acceptance_test": "The replay URL becomes reachable or is explicitly classified as a bad source.",
            },
        }
        return guidance_map.get(
            normalized,
            {
                "what_to_do": "Review this replay mismatch and apply the narrowest deterministic fix first.",
                "how_to_do": "Open the failed URL, compare the source content with the Original/Re-scraped rows, then decide whether the bug is in the source scraper, the DB write guard, or the replay matcher.",
                "where_to_change": "Determine whether the source scraper, replay matcher, or DB write guard is responsible",
                "acceptance_test": "The targeted mismatch category count decreases in the next validation report.",
            },
        )

    def build_parser_improvement_workflow_html(
        self,
        accuracy_replay_summary: dict[str, Any] | None,
        recommendation_plan: dict[str, Any] | None,
        action_queue: dict[str, Any] | None,
        domain_evaluation_summary: dict[str, Any] | None,
    ) -> str:
        """Render an ordered parser-first remediation workflow for the HTML report."""
        if not isinstance(accuracy_replay_summary, dict) or accuracy_replay_summary.get("error"):
            return "<p class='error-box'>❌ Parser remediation workflow unavailable because replay accuracy data is unavailable.</p>"

        replay_rows = accuracy_replay_summary.get("rows", [])
        if not isinstance(replay_rows, list):
            replay_rows = []

        failed_rows: list[dict[str, Any]] = []
        for row in replay_rows:
            if not isinstance(row, dict) or bool(row.get("is_match")):
                continue
            category = str(row.get("mismatch_category") or "").strip()
            if category not in self._PARSER_CATEGORIES:
                continue
            failed_rows.append(row)

        if not failed_rows:
            return (
                "<p>✅ No parser-first replay mismatches are currently queued. "
                "Work the broader recommendation plan only after the next run creates fresh parser evidence.</p>"
            )

        category_counts = Counter(str(row.get("mismatch_category") or "").strip() for row in failed_rows)
        dominant_category = next(iter(category_counts.most_common(1)), ("", 0))[0]
        dominant_guidance = self.guidance_for_category(dominant_category)

        category_rows = ""
        for category, count in category_counts.most_common(5):
            guidance = self.guidance_for_category(category)
            category_rows += (
                "<tr>"
                f"<td>{self.escape_html(category)}</td>"
                f"<td>{count}</td>"
                f"<td>{self.escape_html(guidance['what_to_do'])}</td>"
                f"<td>{self.escape_html(guidance['where_to_change'])}</td>"
                f"<td>{self.escape_html(guidance['how_to_do'])}</td>"
                f"<td>{self.escape_html(guidance['acceptance_test'])}</td>"
                "</tr>"
            )

        priority_rows = ""
        seen_urls: set[str] = set()
        sorted_failed_rows = sorted(
            failed_rows,
            key=lambda item: (
                -int(category_counts.get(str(item.get("mismatch_category") or "").strip(), 0)),
                str(((item.get("baseline") or {}) if isinstance(item.get("baseline"), dict) else {}).get("url") or ""),
            ),
        )
        for row in sorted_failed_rows:
            baseline = row.get("baseline", {}) if isinstance(row.get("baseline"), dict) else {}
            replay = row.get("replay", {}) if isinstance(row.get("replay"), dict) else {}
            baseline_url = str(baseline.get("url", "") or "").strip()
            if not baseline_url or baseline_url in seen_urls:
                continue
            seen_urls.add(baseline_url)
            category = str(row.get("mismatch_category") or "").strip()
            guidance = self.guidance_for_category(category)
            priority_rows += (
                "<tr>"
                f"<td>{len(seen_urls)}</td>"
                f"<td>{self.escape_html(category)}</td>"
                f"<td>{self.escape_html(str(baseline.get('event_name', '') or ''))}</td>"
                f"<td>{self.escape_html(str(baseline.get('url', '') or ''))}</td>"
                f"<td>{self.escape_html(str(replay.get('event_name', '') or ''))}</td>"
                f"<td>{self.escape_html(str(row.get('mismatch_details', '') or ''))}</td>"
                f"<td>{self.escape_html(guidance['where_to_change'])}</td>"
                f"<td>{self.escape_html(guidance['what_to_do'])}</td>"
                "</tr>"
            )
            if len(seen_urls) >= 5:
                break

        if not priority_rows:
            priority_rows = "<tr><td colspan='7'>No priority parser URLs available.</td></tr>"

        parser_recommendations: list[str] = []
        top_issues = recommendation_plan.get("top_issues", []) if isinstance(recommendation_plan, dict) else []
        for item in top_issues:
            if not isinstance(item, dict):
                continue
            issue_type = str(item.get("issue_type") or "").strip()
            if issue_type in {"domain_regression", "classifier_regression", "coverage_gap", "guardrail_violation"}:
                parser_recommendations.append(str(item.get("title") or "").strip())

        queued_actions: list[str] = []
        action_items = action_queue.get("items", []) if isinstance(action_queue, dict) else []
        for item in action_items:
            if not isinstance(item, dict):
                continue
            reason_text = " ".join(
                str(item.get(key) or "").lower() for key in ("title", "reason", "suggested_change", "acceptance_test")
            )
            if any(token in reason_text for token in ("parser", "replay", "scraper", "domain", "classifier")):
                queued_actions.append(str(item.get("title") or "").strip())

        worst_domains = []
        if isinstance(domain_evaluation_summary, dict):
            raw_worst_domains = domain_evaluation_summary.get("worst_domains", [])
            worst_domains = raw_worst_domains if isinstance(raw_worst_domains, list) else []

        worst_domain_text = ", ".join(
            self.escape_html(str(item.get("domain") or "")) for item in worst_domains[:3] if isinstance(item, dict)
        ) or "n/a"
        parser_recommendation_text = "<br>".join(
            self.escape_html(text) for text in parser_recommendations[:3] if text
        ) or "Use the mismatch category table below to drive the next fix."
        queued_action_text = "<br>".join(
            self.escape_html(text) for text in queued_actions[:3] if text
        ) or "No parser-specific action queue entries yet."

        ordered_steps_html = (
            "<table><tr><th>Order</th><th>What To Do</th><th>How To Do It</th><th>Done When</th></tr>"
            "<tr>"
            "<td>1</td>"
            "<td>Work parser/replay mismatches before runtime or cost tuning.</td>"
            f"<td>Start with the dominant mismatch category for this run: <strong>{self.escape_html(dominant_category)}</strong>. {self.escape_html(dominant_guidance['how_to_do'])}</td>"
            f"<td>{self.escape_html(dominant_guidance['acceptance_test'])}</td>"
            "</tr>"
            "<tr>"
            "<td>2</td>"
            "<td>Inspect the exact failed source URLs.</td>"
            "<td>Use the Priority URLs table below. Open the source page, compare the Original and Re-scraped rows, and decide whether the defect is in the original scraper, the DB write guard, or the replay matcher.</td>"
            "<td>You can point to the exact bad field and the exact code path that produced it.</td>"
            "</tr>"
            "<tr>"
            "<td>3</td>"
            "<td>Apply the smallest deterministic fix first.</td>"
            "<td>If the wrong event was selected from a listing page or the replay URL drifted, fix tests/validation/test_runner.py. If the original row was wrong, fix the scraper/prompt or the shared DB write guard instead of broadening model usage.</td>"
            "<td>The targeted row becomes correct without introducing broader behavior changes.</td>"
            "</tr>"
            "<tr>"
            "<td>4</td>"
            "<td>Re-run validation only.</td>"
            "<td>Run <code>set -a; source src/.env; set +a; python tests/validation/test_runner.py || true</code> and inspect the same rows again.</td>"
            "<td>The targeted row disappears or the mismatch category count decreases, and no new guardrail regression appears.</td>"
            "</tr>"
            "</table>"
        )

        return (
            "<p><strong>Recommended first self-improving subsystem:</strong> page parsing and parser routing. "
            "Use this section after each <code>pipeline.py</code> run to decide what to fix first, how to fix it, "
            "and how to verify the change before moving on.</p>"
            "<p><strong>Focus signals:</strong> worst replay domains: "
            f"{worst_domain_text}.<br><strong>Current parser-related recommendation plan items:</strong><br>{parser_recommendation_text}"
            f"<br><strong>Current queued actions:</strong><br>{queued_action_text}</p>"
            "<h3>Ordered Parser Workflow</h3>"
            f"{ordered_steps_html}"
            "<h3>Top Parser Mismatch Categories</h3>"
            "<table><tr><th>Mismatch Category</th><th>Count</th><th>What To Do</th><th>Likely Fix Location</th><th>How To Do It</th><th>Acceptance Test</th></tr>"
            f"{category_rows}</table>"
            "<h3>Priority URLs To Inspect First</h3>"
            "<table><tr><th>Priority</th><th>Mismatch Category</th><th>Original Event</th><th>Source URL</th><th>Re-scraped Event</th><th>Mismatch Details</th><th>Likely Fix Location</th><th>Next Fix</th></tr>"
            f"{priority_rows}</table>"
        )
