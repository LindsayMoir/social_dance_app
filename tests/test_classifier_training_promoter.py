import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classifier_training_promoter import (
    migrate_legacy_classifier_review_csv,
    migrate_legacy_classifier_review_row,
    promote_manual_review_training_rows,
    promote_training_candidates,
    summarize_manual_review_csv,
)


def _write_training_csv(path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "url",
                "domain",
                "url_path",
                "source_file",
                "predicted_archetype",
                "predicted_subtype",
                "predicted_owner_step",
                "predicted_is_event_detail",
                "predicted_is_calendar",
                "predicted_is_social",
                "classifier_stage",
                "classifier_confidence",
                "feature_event_like_links",
                "feature_listing_signal",
                "feature_repeated_date_tokens",
                "feature_listing_score",
                "feature_event_signal",
                "reviewed_truth_archetype",
                "reviewed_truth_owner_step",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "1",
                "url": "https://example.org/existing",
                "domain": "example.org",
                "url_path": "/existing",
                "source_file": "events_table",
                "predicted_archetype": "simple_page",
                "predicted_subtype": "event_detail",
                "predicted_owner_step": "scraper.py",
                "predicted_is_event_detail": "True",
                "predicted_is_calendar": "False",
                "predicted_is_social": "False",
                "classifier_stage": "rule",
                "classifier_confidence": "0.99",
                "feature_event_like_links": "0",
                "feature_listing_signal": "",
                "feature_repeated_date_tokens": "",
                "feature_listing_score": "",
                "feature_event_signal": "",
                "reviewed_truth_archetype": "simple_page",
                "reviewed_truth_owner_step": "scraper.py",
                "review_notes": "seed",
            }
        )


def _write_manual_review_csv(path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "sample_bucket",
                "url",
                "classifier_predicted_archetype",
                "classifier_predicted_owner_step",
                "human_label",
                "human_should_scrape",
                "human_owner_step_correct",
                "human_archetype_correct",
                "human_extraction_outcome_correct",
                "human_has_recoverable_event_data",
                "human_truth_archetype",
                "human_truth_owner_step",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run-1",
                "sample_bucket": "true_candidate",
                "url": "https://example.org/event/two",
                "classifier_predicted_archetype": "simple_page",
                "classifier_predicted_owner_step": "scraper.py",
                "human_label": "true",
                "human_should_scrape": "true",
                "human_owner_step_correct": "true",
                "human_archetype_correct": "true",
                "human_extraction_outcome_correct": "true",
                "human_has_recoverable_event_data": "true",
                "human_truth_archetype": "",
                "human_truth_owner_step": "",
                "review_notes": "good",
            }
        )
        writer.writerow(
            {
                "run_id": "run-1",
                "sample_bucket": "true_candidate",
                "url": "https://example.org/event/three",
                "classifier_predicted_archetype": "simple_page",
                "classifier_predicted_owner_step": "scraper.py",
                "human_label": "correct",
                "human_should_scrape": "true",
                "human_owner_step_correct": "true",
                "human_archetype_correct": "true",
                "human_extraction_outcome_correct": "true",
                "human_has_recoverable_event_data": "true",
                "human_truth_archetype": "",
                "human_truth_owner_step": "",
                "review_notes": "good",
            }
        )
        writer.writerow(
            {
                "run_id": "run-1",
                "sample_bucket": "false_candidate",
                "url": "https://example.org/listing/four",
                "classifier_predicted_archetype": "simple_page",
                "classifier_predicted_owner_step": "scraper.py",
                "human_label": "false",
                "human_should_scrape": "true",
                "human_owner_step_correct": "false",
                "human_archetype_correct": "false",
                "human_extraction_outcome_correct": "false",
                "human_has_recoverable_event_data": "true",
                "human_truth_archetype": "complicated_page",
                "human_truth_owner_step": "rd_ext.py",
                "review_notes": "listing page",
            }
        )
        writer.writerow(
            {
                "run_id": "run-1",
                "sample_bucket": "false_candidate",
                "url": "https://example.org/listing/five",
                "classifier_predicted_archetype": "simple_page",
                "classifier_predicted_owner_step": "scraper.py",
                "human_label": "incorrect",
                "human_should_scrape": "true",
                "human_owner_step_correct": "false",
                "human_archetype_correct": "false",
                "human_extraction_outcome_correct": "false",
                "human_has_recoverable_event_data": "true",
                "human_truth_archetype": "complicated_page",
                "human_truth_owner_step": "rd_ext.py",
                "review_notes": "listing page",
            }
        )


def test_promote_training_candidates_appends_only_auto_positive_unique_urls(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    _write_training_csv(training_csv)

    queue_summary = {
        "candidates": [
            {
                "normalized_url": "https://example.org/event/one",
                "domain": "example.org",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "simple_page",
                "recommended_owner_step": "scraper.py",
                "priority_score": 120,
                "match_rate_pct": 100.0,
            },
            {
                "normalized_url": "https://example.org/event/one",
                "domain": "example.org",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "simple_page",
                "recommended_owner_step": "scraper.py",
                "priority_score": 110,
                "match_rate_pct": 100.0,
            },
            {
                "normalized_url": "https://example.org/events",
                "domain": "example.org",
                "query_text": "query",
                "status": "mixed_signal",
                "training_eligible": False,
                "recommended_archetype": "incomplete_event",
                "recommended_owner_step": "scraper.py",
                "priority_score": 90,
                "match_rate_pct": 50.0,
            },
        ]
    }

    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=training_csv,
        max_promotions=10,
        max_per_domain_per_archetype=3,
    )

    assert result["promoted_count"] == 1
    with open(training_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[-1]["url"] == "https://example.org/event/one"
    assert rows[-1]["reviewed_truth_archetype"] == "simple_page"
    assert rows[-1]["reviewed_truth_owner_step"] == "scraper.py"


def test_promote_training_candidates_respects_domain_archetype_cap(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    _write_training_csv(training_csv)

    queue_summary = {
        "candidates": [
            {
                "normalized_url": "https://example.org/event/one",
                "domain": "example.org",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "simple_page",
                "recommended_owner_step": "scraper.py",
                "priority_score": 120,
                "match_rate_pct": 100.0,
            }
        ]
    }

    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=training_csv,
        max_promotions=10,
        max_per_domain_per_archetype=1,
    )

    assert result["promoted_count"] == 0
    assert result["skipped"][0]["reason"] == "domain_archetype_cap_reached"


def test_promote_training_candidates_excludes_email_like_inputs(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    _write_training_csv(training_csv)

    queue_summary = {
        "candidates": [
            {
                "normalized_url": "person@example.com",
                "domain": "",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "other",
                "recommended_owner_step": "emails.py",
                "priority_score": 120,
                "match_rate_pct": 100.0,
            }
        ]
    }

    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=training_csv,
        max_promotions=10,
        max_per_domain_per_archetype=3,
    )

    assert result["promoted_count"] == 0
    assert result["skipped"][0]["reason"] == "email_like_input_excluded"


def test_promote_training_candidates_excludes_holdout_urls(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    _write_training_csv(training_csv)

    queue_summary = {
        "candidates": [
            {
                "normalized_url": "https://www.redhotswing.com",
                "domain": "www.redhotswing.com",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "simple_page",
                "recommended_owner_step": "scraper.py",
                "priority_score": 120,
                "match_rate_pct": 100.0,
            }
        ]
    }

    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=training_csv,
        max_promotions=10,
        max_per_domain_per_archetype=3,
    )

    assert result["promoted_count"] == 0
    assert result["skipped"][0]["reason"] == "evaluation_url_excluded"


def test_promote_training_candidates_excludes_dev_urls(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    _write_training_csv(training_csv)

    queue_summary = {
        "candidates": [
            {
                "normalized_url": "https://www.bardandbanker.com/live-music",
                "domain": "www.bardandbanker.com",
                "query_text": "query",
                "status": "auto_positive_candidate",
                "training_eligible": True,
                "recommended_archetype": "simple_page",
                "recommended_owner_step": "scraper.py",
                "priority_score": 120,
                "match_rate_pct": 100.0,
            }
        ]
    }

    result = promote_training_candidates(
        queue_summary=queue_summary,
        training_csv_path=training_csv,
        max_promotions=10,
        max_per_domain_per_archetype=3,
    )

    assert result["promoted_count"] == 0
    assert result["skipped"][0]["reason"] == "evaluation_url_excluded"


def test_summarize_manual_review_csv_counts_false_rows_missing_truth(tmp_path) -> None:
    review_csv = tmp_path / "manual_review.csv"
    with open(review_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "url",
                "human_label",
                "human_should_scrape",
                "human_owner_step_correct",
                "human_archetype_correct",
                "human_extraction_outcome_correct",
                "human_has_recoverable_event_data",
                "human_truth_archetype",
                "human_truth_owner_step",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "url": "https://example.org/a",
            "human_label": "true",
            "human_should_scrape": "true",
            "human_owner_step_correct": "true",
            "human_archetype_correct": "true",
            "human_extraction_outcome_correct": "true",
            "human_has_recoverable_event_data": "true",
            "human_truth_archetype": "",
            "human_truth_owner_step": "",
        })
        writer.writerow({
            "url": "https://example.org/b",
            "human_label": "false",
            "human_should_scrape": "true",
            "human_owner_step_correct": "false",
            "human_archetype_correct": "false",
            "human_extraction_outcome_correct": "false",
            "human_has_recoverable_event_data": "true",
            "human_truth_archetype": "",
            "human_truth_owner_step": "",
        })

    summary = summarize_manual_review_csv(review_csv)

    assert summary["rows_true"] == 1
    assert summary["rows_false"] == 1
    assert summary["false_rows_missing_truth"] == 1
    assert summary["component_rows_missing"] == 0
    assert summary["correctness_pct"] == 50.0


def test_summarize_manual_review_csv_counts_missing_component_fields(tmp_path) -> None:
    review_csv = tmp_path / "manual_review.csv"
    with open(review_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "url",
                "human_label",
                "human_should_scrape",
                "human_owner_step_correct",
                "human_archetype_correct",
                "human_extraction_outcome_correct",
                "human_has_recoverable_event_data",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "url": "https://example.org/a",
            "human_label": "true",
            "human_should_scrape": "true",
            "human_owner_step_correct": "true",
            "human_archetype_correct": "",
            "human_extraction_outcome_correct": "true",
            "human_has_recoverable_event_data": "true",
        })

    summary = summarize_manual_review_csv(review_csv)

    assert summary["uses_richer_schema"] is True
    assert summary["component_rows_missing"] == 1


def test_migrate_legacy_classifier_review_row_preserves_old_labels_and_derives_obvious_fields() -> None:
    migrated = migrate_legacy_classifier_review_row(
        {
            "run_id": "run-1",
            "sample_bucket": "false_candidate",
            "url": "https://example.org/event",
            "classifier_predicted_owner_step": "scraper.py",
            "classifier_predicted_archetype": "google_calendar",
            "human_label": "false",
            "human_truth_archetype": "simple_page",
            "human_truth_owner_step": "scraper.py",
            "review_notes": "Not actually a calendar page.",
        }
    )

    assert migrated["human_label"] == "false"
    assert migrated["human_owner_step_correct"] == "True"
    assert migrated["human_archetype_correct"] == "False"
    assert migrated["human_should_scrape"] == ""
    assert migrated["human_truth_archetype"] == "simple_page"


def test_migrate_legacy_classifier_review_csv_writes_richer_schema_file(tmp_path) -> None:
    source_csv = tmp_path / "legacy.csv"
    target_csv = tmp_path / "migrated.csv"
    with open(source_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "sample_bucket",
                "url",
                "classifier_predicted_archetype",
                "classifier_predicted_owner_step",
                "human_label",
                "human_truth_archetype",
                "human_truth_owner_step",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "run-1",
                "sample_bucket": "true_candidate",
                "url": "https://example.org/event",
                "classifier_predicted_archetype": "simple_page",
                "classifier_predicted_owner_step": "scraper.py",
                "human_label": "true",
                "human_truth_archetype": "",
                "human_truth_owner_step": "",
                "review_notes": "good",
            }
        )

    result = migrate_legacy_classifier_review_csv(source_path=source_csv, target_path=target_csv)

    assert result["rows_written"] == 1
    with open(target_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["human_should_scrape"] == "True"
    assert rows[0]["human_archetype_correct"] == "True"


def test_promote_manual_review_training_rows_appends_two_true_and_two_false_without_duplicates(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    review_csv = tmp_path / "manual_review.csv"
    _write_training_csv(training_csv)
    _write_manual_review_csv(review_csv)

    result = promote_manual_review_training_rows(
        review_csv_path=review_csv,
        training_csv_path=training_csv,
        true_limit=2,
        false_limit=2,
    )

    assert result["promoted_count"] == 4
    assert result["promoted_true_count"] == 2
    assert result["promoted_false_count"] == 2
    with open(training_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    promoted = {row["url"]: row for row in rows[1:]}
    assert promoted["https://example.org/event/two"]["reviewed_truth_archetype"] == "simple_page"
    assert promoted["https://example.org/listing/four"]["reviewed_truth_owner_step"] == "rd_ext.py"


def test_promote_manual_review_training_rows_keeps_additions_balanced_when_false_rows_are_scarce(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    review_csv = tmp_path / "manual_review.csv"
    _write_training_csv(training_csv)
    _write_manual_review_csv(review_csv)

    with open(review_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with open(review_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows[:-1]:
            writer.writerow(row)

    result = promote_manual_review_training_rows(
        review_csv_path=review_csv,
        training_csv_path=training_csv,
        true_limit=2,
        false_limit=2,
    )

    assert result["promoted_count"] == 2
    assert result["promoted_true_count"] == 1
    assert result["promoted_false_count"] == 1
    with open(training_csv, newline="", encoding="utf-8") as handle:
        promoted_rows = list(csv.DictReader(handle))
    assert len(promoted_rows) == 3
