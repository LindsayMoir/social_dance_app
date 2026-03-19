import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from page_classifier import classify_page_with_confidence
from page_classifier_ml import (
    predict_page_classifier_labels,
    reset_model_cache,
    train_page_classifier_models,
)


FIELDNAMES = [
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
]


def _write_training_rows(path: str) -> None:
    rows = [
        {
            "row_id": "1",
            "url": "https://calendar.example.org/events",
            "domain": "calendar.example.org",
            "url_path": "/events",
            "source_file": "events_table",
            "predicted_archetype": "incomplete_event",
            "predicted_subtype": "incomplete_event",
            "predicted_owner_step": "scraper.py",
            "predicted_is_event_detail": "False",
            "predicted_is_calendar": "False",
            "predicted_is_social": "False",
            "classifier_stage": "structural",
            "classifier_confidence": "0.85",
            "feature_event_like_links": "8",
            "feature_listing_signal": "True",
            "feature_repeated_date_tokens": "6",
            "feature_listing_score": "5",
            "feature_event_signal": "True",
            "reviewed_truth_archetype": "incomplete_event",
            "reviewed_truth_owner_step": "rd_ext.py",
            "review_notes": "",
        },
        {
            "row_id": "2",
            "url": "https://calendar.example.org/upcoming",
            "domain": "calendar.example.org",
            "url_path": "/upcoming",
            "source_file": "events_table",
            "predicted_archetype": "incomplete_event",
            "predicted_subtype": "incomplete_event",
            "predicted_owner_step": "scraper.py",
            "predicted_is_event_detail": "False",
            "predicted_is_calendar": "False",
            "predicted_is_social": "False",
            "classifier_stage": "structural",
            "classifier_confidence": "0.85",
            "feature_event_like_links": "7",
            "feature_listing_signal": "True",
            "feature_repeated_date_tokens": "5",
            "feature_listing_score": "4",
            "feature_event_signal": "True",
            "reviewed_truth_archetype": "incomplete_event",
            "reviewed_truth_owner_step": "rd_ext.py",
            "review_notes": "",
        },
        {
            "row_id": "3",
            "url": "https://detail.example.org/event/friday-social",
            "domain": "detail.example.org",
            "url_path": "/event/friday-social",
            "source_file": "events_table",
            "predicted_archetype": "simple_page",
            "predicted_subtype": "event_detail",
            "predicted_owner_step": "scraper.py",
            "predicted_is_event_detail": "True",
            "predicted_is_calendar": "False",
            "predicted_is_social": "False",
            "classifier_stage": "rule",
            "classifier_confidence": "0.99",
            "feature_event_like_links": "1",
            "feature_listing_signal": "False",
            "feature_repeated_date_tokens": "1",
            "feature_listing_score": "0",
            "feature_event_signal": "True",
            "reviewed_truth_archetype": "simple_page",
            "reviewed_truth_owner_step": "scraper.py",
            "review_notes": "",
        },
        {
            "row_id": "4",
            "url": "https://detail.example.org/show/jazz-night",
            "domain": "detail.example.org",
            "url_path": "/show/jazz-night",
            "source_file": "events_table",
            "predicted_archetype": "simple_page",
            "predicted_subtype": "event_detail",
            "predicted_owner_step": "scraper.py",
            "predicted_is_event_detail": "True",
            "predicted_is_calendar": "False",
            "predicted_is_social": "False",
            "classifier_stage": "rule",
            "classifier_confidence": "0.99",
            "feature_event_like_links": "1",
            "feature_listing_signal": "False",
            "feature_repeated_date_tokens": "1",
            "feature_listing_score": "0",
            "feature_event_signal": "True",
            "reviewed_truth_archetype": "simple_page",
            "reviewed_truth_owner_step": "scraper.py",
            "review_notes": "",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_train_and_predict_page_classifier_models(tmp_path, monkeypatch) -> None:
    training_csv = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    _write_training_rows(str(training_csv))

    result = train_page_classifier_models(
        training_csv_path=training_csv,
        output_path=model_path,
    )
    assert result["training_row_count"] == 4

    monkeypatch.setenv("PAGE_CLASSIFIER_MODEL_PATH", str(model_path))
    monkeypatch.setenv("PAGE_CLASSIFIER_ML_ENABLED", "1")
    reset_model_cache()
    prediction = predict_page_classifier_labels(
        url="https://calendar.example.org/events/april",
        page_links_count=9,
        listing_signal=True,
        repeated_date_tokens=6,
        listing_score=5,
        event_signal=True,
    )
    assert prediction is not None
    assert prediction.archetype == "incomplete_event"
    assert prediction.owner_step == "rd_ext.py"


def test_training_excludes_email_like_rows(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    _write_training_rows(str(training_csv))
    with open(training_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(
            {
                "row_id": "5",
                "url": "person@example.com",
                "domain": "",
                "url_path": "person@example.com",
                "source_file": "events_table",
                "predicted_archetype": "incomplete_event",
                "predicted_subtype": "low_confidence_fallback",
                "predicted_owner_step": "scraper.py",
                "predicted_is_event_detail": "False",
                "predicted_is_calendar": "False",
                "predicted_is_social": "False",
                "classifier_stage": "structural",
                "classifier_confidence": "0.52",
                "feature_event_like_links": "0",
                "feature_listing_signal": "False",
                "feature_repeated_date_tokens": "0",
                "feature_listing_score": "0",
                "feature_event_signal": "False",
                "reviewed_truth_archetype": "other",
                "reviewed_truth_owner_step": "emails.py",
                "review_notes": "",
            }
        )

    result = train_page_classifier_models(
        training_csv_path=training_csv,
        output_path=model_path,
    )
    assert result["training_row_count"] == 4


def test_training_excludes_holdout_rows(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    _write_training_rows(str(training_csv))
    with open(training_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writerow(
            {
                "row_id": "5",
                "url": "https://www.redhotswing.com/",
                "domain": "www.redhotswing.com",
                "url_path": "/",
                "source_file": "events_table",
                "predicted_archetype": "simple_page",
                "predicted_subtype": "event_detail",
                "predicted_owner_step": "scraper.py",
                "predicted_is_event_detail": "True",
                "predicted_is_calendar": "False",
                "predicted_is_social": "False",
                "classifier_stage": "rule",
                "classifier_confidence": "0.99",
                "feature_event_like_links": "1",
                "feature_listing_signal": "False",
                "feature_repeated_date_tokens": "1",
                "feature_listing_score": "0",
                "feature_event_signal": "True",
                "reviewed_truth_archetype": "simple_page",
                "reviewed_truth_owner_step": "scraper.py",
                "review_notes": "",
            }
        )

    result = train_page_classifier_models(
        training_csv_path=training_csv,
        output_path=model_path,
    )
    assert result["training_row_count"] == 4


def test_classify_page_with_confidence_uses_ml_for_ambiguous_pages(tmp_path, monkeypatch) -> None:
    training_csv = tmp_path / "training.csv"
    model_path = tmp_path / "model.joblib"
    _write_training_rows(str(training_csv))
    train_page_classifier_models(
        training_csv_path=training_csv,
        output_path=model_path,
    )

    monkeypatch.setenv("PAGE_CLASSIFIER_MODEL_PATH", str(model_path))
    monkeypatch.setenv("PAGE_CLASSIFIER_ML_ENABLED", "1")
    reset_model_cache()
    decision = classify_page_with_confidence(
        url="https://calendar.example.org/upcoming/april",
        page_links_count=9,
        visible_text="Upcoming events in April with many listings",
    )
    assert decision.stage == "ml"
    assert decision.classification.archetype == "incomplete_event"
    assert decision.classification.owner_step == "rd_ext.py"
    assert decision.features["ml_predicted_archetype"] == "incomplete_event"
