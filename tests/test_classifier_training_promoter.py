import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from classifier_training_promoter import promote_training_candidates


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
