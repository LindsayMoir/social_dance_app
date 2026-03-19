"""
Promote URL-level classifier queue candidates into training CSV rows.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from evaluation_holdout import load_gold_holdout_urls
from page_classifier import classify_page_with_confidence


_EMAIL_INPUT_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)


def promote_training_candidates(
    *,
    queue_summary: dict[str, Any],
    training_csv_path: str | Path,
    max_promotions: int = 10,
    max_per_domain_per_archetype: int = 3,
) -> dict[str, Any]:
    """
    Append eligible, deduplicated queue candidates into the training CSV.
    """
    training_path = Path(training_csv_path)
    if not training_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_path}")

    rows = _read_csv_rows(training_path)
    if not rows:
        raise ValueError(f"Training CSV is empty: {training_path}")

    header = list(rows[0].keys())
    existing_urls = {str(row.get("url") or "").strip() for row in rows}
    holdout_urls = load_gold_holdout_urls()
    current_row_ids = [int(str(row.get("row_id") or "0") or 0) for row in rows if str(row.get("row_id") or "").strip().isdigit()]
    next_row_id = max(current_row_ids, default=0) + 1

    existing_domain_archetype_counts = Counter()
    for row in rows:
        url = str(row.get("url") or "").strip()
        archetype = str(row.get("reviewed_truth_archetype") or "").strip()
        if not url or not archetype:
            continue
        domain = (urlparse(url).netloc or "").lower()
        existing_domain_archetype_counts[(domain, archetype)] += 1

    candidates = queue_summary.get("candidates", []) if isinstance(queue_summary, dict) else []
    selected_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate in candidates:
        if len(selected_rows) >= max(0, int(max_promotions or 0)):
            break
        selected, reason = _select_candidate(
            candidate=candidate,
            existing_urls=existing_urls,
            holdout_urls=holdout_urls,
            domain_archetype_counts=existing_domain_archetype_counts,
            max_per_domain_per_archetype=max_per_domain_per_archetype,
        )
        if not selected:
            skipped.append(
                {
                    "normalized_url": str((candidate or {}).get("normalized_url") or ""),
                    "reason": reason,
                }
            )
            continue
        csv_row = _build_training_csv_row(
            row_id=next_row_id,
            candidate=candidate,
            header=header,
        )
        selected_rows.append(csv_row)
        url = str(csv_row.get("url") or "").strip()
        domain = (urlparse(url).netloc or "").lower()
        archetype = str(csv_row.get("reviewed_truth_archetype") or "").strip()
        existing_urls.add(url)
        existing_domain_archetype_counts[(domain, archetype)] += 1
        next_row_id += 1

    if selected_rows:
        with training_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            for row in selected_rows:
                writer.writerow(row)

    return {
        "promoted_count": len(selected_rows),
        "promoted_urls": [str(row.get("url") or "") for row in selected_rows],
        "skipped": skipped,
    }


def load_queue_summary(path: str | Path) -> dict[str, Any]:
    queue_path = Path(path)
    with queue_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Queue summary at {queue_path} is not a JSON object")
    return data


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_candidate(
    *,
    candidate: dict[str, Any],
    existing_urls: set[str],
    holdout_urls: set[str],
    domain_archetype_counts: Counter[tuple[str, str]],
    max_per_domain_per_archetype: int,
) -> tuple[bool, str]:
    url = str(candidate.get("normalized_url") or "").strip()
    if not url:
        return False, "missing_url"
    if _is_email_like_input(url):
        return False, "email_like_input_excluded"
    if url in holdout_urls:
        return False, "holdout_url_excluded"
    if url in existing_urls:
        return False, "duplicate_url"
    if not bool(candidate.get("training_eligible")):
        return False, "not_training_eligible"
    if str(candidate.get("status") or "") != "auto_positive_candidate":
        return False, "status_not_auto_positive"
    archetype = str(candidate.get("recommended_archetype") or "").strip()
    if not archetype:
        return False, "missing_recommended_archetype"
    domain = (urlparse(url).netloc or "").lower()
    if domain_archetype_counts[(domain, archetype)] >= int(max(1, max_per_domain_per_archetype)):
        return False, "domain_archetype_cap_reached"
    return True, ""


def _build_training_csv_row(
    *,
    row_id: int,
    candidate: dict[str, Any],
    header: list[str],
) -> dict[str, Any]:
    url = str(candidate.get("normalized_url") or "").strip()
    classification = classify_page_with_confidence(url=url)
    parsed = urlparse(url)
    notes = (
        "auto_promoted_from_replay_queue:"
        f"status={candidate.get('status','')};"
        f"match_rate_pct={candidate.get('match_rate_pct', 0)};"
        f"priority_score={candidate.get('priority_score', 0)}"
    )
    row = {column: "" for column in header}
    row.update(
        {
            "row_id": row_id,
            "url": url,
            "domain": (parsed.netloc or "").lower(),
            "url_path": parsed.path or url,
            "source_file": "replay_queue",
            "predicted_archetype": classification.classification.archetype,
            "predicted_subtype": classification.classification.subtype,
            "predicted_owner_step": classification.classification.owner_step,
            "predicted_is_event_detail": classification.classification.is_event_detail,
            "predicted_is_calendar": classification.classification.is_calendar,
            "predicted_is_social": classification.classification.is_social,
            "classifier_stage": classification.stage,
            "classifier_confidence": classification.confidence,
            "feature_event_like_links": classification.features.get("event_like_links", ""),
            "feature_listing_signal": classification.features.get("listing_signal", ""),
            "feature_repeated_date_tokens": classification.features.get("repeated_date_tokens", ""),
            "feature_listing_score": classification.features.get("listing_score", ""),
            "feature_event_signal": classification.features.get("event_signal", ""),
            "reviewed_truth_archetype": str(candidate.get("recommended_archetype") or "").strip(),
            "reviewed_truth_owner_step": str(candidate.get("recommended_owner_step") or "").strip(),
            "review_notes": notes,
        }
    )
    return row


def _is_email_like_input(value: Any) -> bool:
    return _EMAIL_INPUT_RE.match(str(value or "").strip()) is not None
