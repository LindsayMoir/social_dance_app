import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classifier_training_queue import (
    build_classifier_training_queue,
    filter_classifier_review_candidates,
)


def test_build_classifier_training_queue_deduplicates_by_url_and_marks_positive_candidate(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    with training_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reviewed_truth_archetype", "reviewed_truth_owner_step"],
        )
        writer.writeheader()
        writer.writerow({"reviewed_truth_archetype": "simple_page", "reviewed_truth_owner_step": "scraper.py"})

    replay_rows = [
        {
            "is_match": True,
            "mismatch_category": "",
            "baseline_event_id": 11,
            "baseline": {
                "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                "event_name": "Event A",
                "start_date": "2026-03-22",
                "start_time": "19:00:00",
                "source": "livevictoria.com",
                "location": "Venue A",
            },
            "replay": {},
        },
        {
            "is_match": True,
            "mismatch_category": "",
            "baseline_event_id": 12,
            "baseline": {
                "url": "https://livevictoria.com/calendar/music&month_limit=4&year_limit=2026",
                "event_name": "Event B",
                "start_date": "2026-03-23",
                "start_time": "20:00:00",
                "source": "livevictoria.com",
                "location": "Venue B",
            },
            "replay": {},
        },
    ]

    summary = build_classifier_training_queue(
        replay_rows,
        training_csv_path=training_csv,
        query_text="Where can I dance next week?",
    )

    assert summary["total_urls"] == 1
    assert summary["training_ready_count"] == 1
    candidate = summary["candidates"][0]
    assert candidate["status"] == "auto_positive_candidate"
    assert candidate["training_eligible"] is True
    assert candidate["recommended_archetype"] == "incomplete_event"
    assert candidate["recommended_owner_step"] == "scraper.py"
    assert candidate["true_count"] == 2
    assert candidate["false_count"] == 0


def test_build_classifier_training_queue_separates_extraction_issues_from_classifier_review(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    with training_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reviewed_truth_archetype", "reviewed_truth_owner_step"],
        )
        writer.writeheader()
        writer.writerow({"reviewed_truth_archetype": "google_calendar", "reviewed_truth_owner_step": "scraper.py"})

    replay_rows = [
        {
            "is_match": False,
            "mismatch_category": "wrong_date",
            "baseline_event_id": 21,
            "baseline": {"url": "https://example.org/event/one"},
            "replay": {"url": "https://example.org/event/one"},
        },
        {
            "is_match": False,
            "mismatch_category": "no_event_extracted_replay",
            "baseline_event_id": 22,
            "baseline": {"url": "https://example.org/events"},
            "replay": {},
        },
    ]

    summary = build_classifier_training_queue(
        replay_rows,
        training_csv_path=training_csv,
        query_text="Where can I dance next week?",
    )

    statuses = {candidate["normalized_url"]: candidate["status"] for candidate in summary["candidates"]}
    assert statuses["https://example.org/event/one"] == "extraction_issue"
    assert statuses["https://example.org/events"] == "classifier_review_needed"


def test_filter_classifier_review_candidates_keeps_only_classifier_review_needed() -> None:
    queue_summary = {
        "status": "OK",
        "run_id": "run-1",
        "query_text": "query",
        "candidates": [
            {"normalized_url": "https://example.org/a", "status": "classifier_review_needed"},
            {"normalized_url": "https://example.org/b", "status": "auto_positive_candidate"},
            {"normalized_url": "https://example.org/c", "status": "mixed_signal"},
        ],
    }
    filtered = filter_classifier_review_candidates(queue_summary)
    assert filtered["total_urls"] == 1
    assert filtered["candidates"][0]["normalized_url"] == "https://example.org/a"
