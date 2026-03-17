"""
Utilities for turning replay validation output into URL-level classifier labeling candidates.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from page_classifier import classify_page


_CLASSIFIER_REVIEW_CATEGORIES = {
    "no_event_extracted_replay",
    "content_drift_major",
}
_NON_CLASSIFIER_CATEGORIES = {
    "wrong_date",
    "wrong_time",
    "wrong_location_or_address",
    "wrong_source",
    "duplicate_event_identity",
    "url_unreachable_replay",
    "social_platform_scraper_init_failed",
    "parser_or_llm_failure",
    "missing_url_baseline",
    "other_mismatch",
}


def load_training_label_distribution(training_csv_path: str | Path) -> dict[str, Counter[str]]:
    """
    Load current reviewed-label counts from the training CSV.
    """
    path = Path(training_csv_path)
    archetype_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    if not path.exists():
        return {"archetype": archetype_counts, "owner": owner_counts}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            archetype = str(row.get("reviewed_truth_archetype") or "").strip()
            owner = str(row.get("reviewed_truth_owner_step") or "").strip()
            if archetype:
                archetype_counts.update([archetype])
            if owner:
                owner_counts.update([owner])
    return {"archetype": archetype_counts, "owner": owner_counts}


def build_classifier_training_queue(
    replay_rows: list[dict[str, Any]],
    *,
    training_csv_path: str | Path,
    query_text: str = "",
) -> dict[str, Any]:
    """
    Aggregate replay validation rows into one classifier-label candidate per normalized URL.
    """
    label_distribution = load_training_label_distribution(training_csv_path)
    archetype_counts = label_distribution["archetype"]
    owner_counts = label_distribution["owner"]

    grouped: dict[str, dict[str, Any]] = {}
    for row in replay_rows:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
        replay = row.get("replay") if isinstance(row.get("replay"), dict) else {}
        normalized_url = _normalize_url_value(baseline.get("url"))
        if not normalized_url:
            continue
        current = grouped.setdefault(
            normalized_url,
            {
                "normalized_url": normalized_url,
                "domain": (urlparse(normalized_url).netloc or "").lower(),
                "query_text": str(query_text or "").strip(),
                "total_rows": 0,
                "true_count": 0,
                "false_count": 0,
                "mismatch_category_counts": Counter(),
                "sample_baseline": baseline,
                "sample_replay": replay,
                "baseline_event_ids": [],
            },
        )
        current["total_rows"] += 1
        if bool(row.get("is_match")):
            current["true_count"] += 1
        else:
            current["false_count"] += 1
            category = str(row.get("mismatch_category") or "").strip() or "unknown"
            current["mismatch_category_counts"].update([category])
            if not current.get("sample_replay") and replay:
                current["sample_replay"] = replay
        event_id = row.get("baseline_event_id")
        if event_id is not None:
            current["baseline_event_ids"].append(int(event_id))

    candidates: list[dict[str, Any]] = []
    for aggregated in grouped.values():
        classification = classify_page(url=aggregated["normalized_url"])
        category_counts = dict(aggregated["mismatch_category_counts"])
        status, recommended_action, training_eligible = _determine_candidate_status(
            true_count=int(aggregated["true_count"]),
            false_count=int(aggregated["false_count"]),
            category_counts=category_counts,
        )
        priority_score = _compute_priority_score(
            classification_archetype=classification.archetype,
            classification_owner=classification.owner_step,
            status=status,
            archetype_counts=archetype_counts,
            owner_counts=owner_counts,
        )
        candidates.append(
            {
                "normalized_url": aggregated["normalized_url"],
                "domain": aggregated["domain"],
                "query_text": aggregated["query_text"],
                "total_rows": int(aggregated["total_rows"]),
                "true_count": int(aggregated["true_count"]),
                "false_count": int(aggregated["false_count"]),
                "match_rate_pct": round((aggregated["true_count"] / aggregated["total_rows"]) * 100.0, 2),
                "status": status,
                "recommended_action": recommended_action,
                "training_eligible": training_eligible,
                "recommended_archetype": classification.archetype,
                "recommended_owner_step": classification.owner_step,
                "recommended_subtype": classification.subtype,
                "priority_score": priority_score,
                "mismatch_category_counts": category_counts,
                "baseline_event_ids": sorted(set(int(v) for v in aggregated["baseline_event_ids"])),
                "sample_baseline": aggregated["sample_baseline"] or {},
                "sample_replay": aggregated["sample_replay"] or {},
            }
        )

    candidates.sort(
        key=lambda item: (
            -int(item["priority_score"]),
            item["status"] != "auto_positive_candidate",
            -int(item["false_count"]),
            item["normalized_url"],
        )
    )
    status_counts = Counter(str(item.get("status") or "") for item in candidates)
    domain_counts = Counter(str(item.get("domain") or "") for item in candidates)
    training_ready = [item for item in candidates if bool(item.get("training_eligible"))]
    return {
        "query_text": str(query_text or "").strip(),
        "total_urls": len(candidates),
        "status_counts": dict(status_counts),
        "domain_counts": dict(domain_counts),
        "training_ready_count": len(training_ready),
        "review_required_count": sum(1 for item in candidates if not bool(item.get("training_eligible"))),
        "candidates": candidates,
        "training_label_distribution": {
            "archetype": dict(archetype_counts),
            "owner": dict(owner_counts),
        },
    }


def filter_classifier_review_candidates(queue_summary: dict[str, Any]) -> dict[str, Any]:
    """
    Return only candidates that require classifier-focused human review.
    """
    candidates = queue_summary.get("candidates", []) if isinstance(queue_summary, dict) else []
    filtered = [
        candidate
        for candidate in candidates
        if str(candidate.get("status") or "").strip() == "classifier_review_needed"
    ]
    return {
        "status": str((queue_summary or {}).get("status") or "OK"),
        "run_id": str((queue_summary or {}).get("run_id") or "").strip(),
        "query_text": str((queue_summary or {}).get("query_text") or "").strip(),
        "total_urls": len(filtered),
        "candidates": filtered,
    }


def _determine_candidate_status(
    *,
    true_count: int,
    false_count: int,
    category_counts: dict[str, int],
) -> tuple[str, str, bool]:
    if true_count > 0 and false_count == 0:
        return "auto_positive_candidate", "review_and_add_positive_label", True
    if false_count > 0 and true_count == 0:
        categories = set(category_counts)
        if categories and categories.issubset(_CLASSIFIER_REVIEW_CATEGORIES):
            return "classifier_review_needed", "review_for_negative_or_relabel", False
        if categories and categories.issubset(_NON_CLASSIFIER_CATEGORIES):
            return "extraction_issue", "do_not_add_to_classifier_training", False
        return "manual_review_needed", "manual_review_before_training", False
    if true_count > 0 and false_count > 0:
        return "mixed_signal", "manual_review_before_training", False
    return "manual_review_needed", "manual_review_before_training", False


def _compute_priority_score(
    *,
    classification_archetype: str,
    classification_owner: str,
    status: str,
    archetype_counts: Counter[str],
    owner_counts: Counter[str],
) -> int:
    max_arch = max(archetype_counts.values(), default=0)
    max_owner = max(owner_counts.values(), default=0)
    arch_gap = max(0, max_arch - int(archetype_counts.get(classification_archetype, 0)))
    owner_gap = max(0, max_owner - int(owner_counts.get(classification_owner, 0)))
    status_bonus = {
        "auto_positive_candidate": 100,
        "classifier_review_needed": 60,
        "mixed_signal": 30,
        "manual_review_needed": 20,
        "extraction_issue": 0,
    }.get(status, 10)
    return int(status_bonus + arch_gap * 3 + owner_gap * 2)


def _normalize_url_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        scheme = (parsed.scheme or "https").lower()
        netloc = (parsed.netloc or "").lower()
        path = (parsed.path or "").rstrip("/")
        query = parsed.query or ""
        return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")
    except Exception:
        return text.lower().rstrip("/")
