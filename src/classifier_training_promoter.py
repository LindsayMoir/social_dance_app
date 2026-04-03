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

from evaluation_holdout import load_dev_urls, load_gold_holdout_urls, normalize_evaluation_url
from page_classifier import classify_page_with_confidence


_EMAIL_INPUT_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
_TRUE_REVIEW_LABELS = {"true", "correct", "yes", "y", "1"}
_FALSE_REVIEW_LABELS = {"false", "incorrect", "no", "n", "0"}


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
    existing_urls = {normalize_evaluation_url(row.get("url")) for row in rows if normalize_evaluation_url(row.get("url"))}
    excluded_urls = load_gold_holdout_urls() | load_dev_urls()
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
            excluded_urls=excluded_urls,
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
        normalized_url = normalize_evaluation_url(url)
        if normalized_url:
            existing_urls.add(normalized_url)
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


def parse_manual_review_label(value: Any) -> bool | None:
    """Normalize a manual-review correctness label into True/False/None."""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_REVIEW_LABELS:
        return True
    if normalized in _FALSE_REVIEW_LABELS:
        return False
    return None


def summarize_manual_review_csv(path: str | Path) -> dict[str, Any]:
    """Summarize a scored manual-review CSV for gating and trend reporting."""
    review_path = Path(path)
    if not review_path.exists():
        return {
            "exists": False,
            "rows_total": 0,
            "rows_completed": 0,
            "rows_missing_label": 0,
            "rows_true": 0,
            "rows_false": 0,
            "rows_unknown_label": 0,
            "false_rows_missing_truth": 0,
            "correctness_pct": None,
        }

    rows = _read_csv_rows(review_path)
    rows_total = len(rows)
    rows_completed = 0
    rows_missing_label = 0
    rows_true = 0
    rows_false = 0
    rows_unknown_label = 0
    false_rows_missing_truth = 0
    for row in rows:
        parsed = parse_manual_review_label(row.get("human_label"))
        if parsed is None:
            label_text = str(row.get("human_label") or "").strip()
            if label_text:
                rows_completed += 1
                rows_unknown_label += 1
            else:
                rows_missing_label += 1
            continue
        rows_completed += 1
        if parsed:
            rows_true += 1
            continue
        rows_false += 1
        truth_archetype = str(row.get("human_truth_archetype") or "").strip()
        truth_owner_step = str(row.get("human_truth_owner_step") or "").strip()
        if not truth_archetype or not truth_owner_step:
            false_rows_missing_truth += 1

    labeled_rows = rows_true + rows_false
    correctness_pct = round((rows_true / labeled_rows) * 100.0, 2) if labeled_rows else None
    return {
        "exists": True,
        "rows_total": rows_total,
        "rows_completed": rows_completed,
        "rows_missing_label": rows_missing_label,
        "rows_true": rows_true,
        "rows_false": rows_false,
        "rows_unknown_label": rows_unknown_label,
        "false_rows_missing_truth": false_rows_missing_truth,
        "correctness_pct": correctness_pct,
    }


def promote_manual_review_training_rows(
    *,
    review_csv_path: str | Path,
    training_csv_path: str | Path,
    true_limit: int = 2,
    false_limit: int = 2,
) -> dict[str, Any]:
    """Append reviewed manual-audit rows into the classifier training CSV."""
    training_path = Path(training_csv_path)
    if not training_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_path}")

    rows = _read_csv_rows(training_path)
    if not rows:
        raise ValueError(f"Training CSV is empty: {training_path}")

    review_rows = _read_csv_rows(Path(review_csv_path))
    header = list(rows[0].keys())
    existing_urls = {
        normalize_evaluation_url(row.get("url"))
        for row in rows
        if normalize_evaluation_url(row.get("url"))
    }
    excluded_urls = load_gold_holdout_urls() | load_dev_urls()
    current_row_ids = [
        int(str(row.get("row_id") or "0") or 0)
        for row in rows
        if str(row.get("row_id") or "").strip().isdigit()
    ]
    next_row_id = max(current_row_ids, default=0) + 1

    selected_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    eligible_true_rows: list[dict[str, Any]] = []
    eligible_false_rows: list[dict[str, Any]] = []

    for review_row in review_rows:
        parsed = parse_manual_review_label(review_row.get("human_label"))
        if parsed is None:
            continue

        csv_row, reason = _build_training_csv_row_from_manual_review(
            row_id=next_row_id,
            review_row=review_row,
            header=header,
            existing_urls=existing_urls,
            excluded_urls=excluded_urls,
        )
        if not csv_row:
            skipped.append(
                {
                    "url": str(review_row.get("url") or "").strip(),
                    "reason": reason,
                }
            )
            continue
        if parsed:
            eligible_true_rows.append(csv_row)
        else:
            eligible_false_rows.append(csv_row)
        next_row_id += 1

    pair_limit = min(
        max(0, int(true_limit or 0)),
        max(0, int(false_limit or 0)),
        len(eligible_true_rows),
        len(eligible_false_rows),
    )
    for csv_row in eligible_true_rows[:pair_limit] + eligible_false_rows[:pair_limit]:
        selected_rows.append(csv_row)
        normalized_url = normalize_evaluation_url(csv_row.get("url"))
        if normalized_url:
            existing_urls.add(normalized_url)

    if selected_rows:
        with training_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            for row in selected_rows:
                writer.writerow(row)

    return {
        "promoted_count": len(selected_rows),
        "promoted_true_count": pair_limit,
        "promoted_false_count": pair_limit,
        "promoted_urls": [str(row.get("url") or "") for row in selected_rows],
        "skipped": skipped,
    }


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _build_training_csv_row_from_manual_review(
    *,
    row_id: int,
    review_row: dict[str, Any],
    header: list[str],
    existing_urls: set[str],
    excluded_urls: set[str],
) -> tuple[dict[str, Any] | None, str]:
    raw_url = str(review_row.get("url") or "").strip()
    if not raw_url:
        return None, "missing_url"
    if _is_email_like_input(raw_url):
        return None, "email_like_input_excluded"
    url = normalize_evaluation_url(raw_url)
    if not url:
        return None, "missing_normalized_url"
    if url in excluded_urls:
        return None, "evaluation_url_excluded"
    if url in existing_urls:
        return None, "duplicate_url"

    parsed_label = parse_manual_review_label(review_row.get("human_label"))
    if parsed_label is None:
        return None, "unusable_human_label"

    classification = classify_page_with_confidence(url=url)
    predicted_archetype = str(
        review_row.get("classifier_predicted_archetype")
        or classification.classification.archetype
        or ""
    ).strip()
    predicted_owner_step = str(
        review_row.get("classifier_predicted_owner_step")
        or classification.classification.owner_step
        or ""
    ).strip()

    truth_archetype = str(review_row.get("human_truth_archetype") or "").strip()
    truth_owner_step = str(review_row.get("human_truth_owner_step") or "").strip()
    if parsed_label:
        truth_archetype = truth_archetype or predicted_archetype
        truth_owner_step = truth_owner_step or predicted_owner_step
    elif not truth_archetype or not truth_owner_step:
        return None, "false_row_missing_truth_labels"

    parsed = urlparse(url)
    manual_notes = str(review_row.get("review_notes") or "").strip()
    sample_bucket = str(review_row.get("sample_bucket") or "").strip()
    run_id = str(review_row.get("run_id") or "").strip()
    row = {column: "" for column in header}
    row.update(
        {
            "row_id": row_id,
            "url": url,
            "domain": (parsed.netloc or "").lower(),
            "url_path": parsed.path or url,
            "source_file": "manual_review_audit",
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
            "reviewed_truth_archetype": truth_archetype,
            "reviewed_truth_owner_step": truth_owner_step,
            "review_notes": (
                "manual_review_audit:"
                f"human_label={str(review_row.get('human_label') or '').strip()};"
                f"sample_bucket={sample_bucket};"
                f"run_id={run_id};"
                f"notes={manual_notes}"
            ),
        }
    )
    return row, ""


def _select_candidate(
    *,
    candidate: dict[str, Any],
    existing_urls: set[str],
    excluded_urls: set[str],
    domain_archetype_counts: Counter[tuple[str, str]],
    max_per_domain_per_archetype: int,
) -> tuple[bool, str]:
    raw_url = str(candidate.get("normalized_url") or "").strip()
    if not raw_url:
        return False, "missing_url"
    if _is_email_like_input(raw_url):
        return False, "email_like_input_excluded"
    url = normalize_evaluation_url(raw_url)
    if url in excluded_urls:
        return False, "evaluation_url_excluded"
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
