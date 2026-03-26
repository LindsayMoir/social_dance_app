from __future__ import annotations

import csv
import json
from pathlib import Path
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pipeline


def test_scorecard_guardrails_allow_pass(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(json.dumps({"guardrails": {"status": "PASS"}}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_guardrails_allow("copy_dev_to_prod") is True


def test_scorecard_guardrails_allow_fail(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps({"guardrails": {"status": "FAIL", "violations": [{"detail": "bad"}]}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_guardrails_allow("copy_dev_to_prod") is False


def test_scorecard_has_required_evaluation_scope_pass(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "evaluation_scope": {
                    "uses_dev_split": True,
                    "uses_holdout": True,
                    "dev_summary": {"replay_url_accuracy_pct": 81.0},
                    "holdout_summary": {"replay_url_accuracy_pct": 79.0},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_has_required_evaluation_scope("classifier_training_promotion") is True


def test_scorecard_has_required_evaluation_scope_fail(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "evaluation_scope": {
                    "uses_dev_split": True,
                    "uses_holdout": True,
                    "dev_summary": {"replay_url_accuracy_pct": None},
                    "holdout_summary": {"replay_url_accuracy_pct": 79.0},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_has_required_evaluation_scope("classifier_training_promotion") is False


def test_scorecard_evaluation_deltas_allow_pass(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "comparison_summary": {
                    "previous_run": {
                        "available": True,
                        "metric_deltas": [
                            {"metric_key": "dev_replay_url_accuracy_pct", "direction": "improved"},
                        ],
                    },
                    "holdout_baseline": {
                        "available": True,
                        "metric_deltas": [
                            {"metric_key": "holdout_replay_url_accuracy_pct", "direction": "unchanged"},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_evaluation_deltas_allow("classifier_training_promotion") is True


def test_scorecard_evaluation_deltas_allow_fail(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "comparison_summary": {
                    "previous_run": {
                        "available": True,
                        "metric_deltas": [
                            {"metric_key": "dev_replay_url_accuracy_pct", "direction": "regressed"},
                        ],
                    },
                    "holdout_baseline": {
                        "available": True,
                        "metric_deltas": [
                            {"metric_key": "holdout_replay_url_accuracy_pct", "direction": "improved"},
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    assert pipeline._scorecard_evaluation_deltas_allow("classifier_training_promotion") is False


def test_log_copy_dev_to_prod_evaluation_warnings_does_not_raise(tmp_path, monkeypatch) -> None:
    scorecard_path = tmp_path / "run_scorecard.json"
    scorecard_path.write_text(
        json.dumps(
            {
                "guardrails": {"status": "FAIL", "violations": [{"detail": "bad"}]},
                "evaluation_scope": {
                    "uses_dev_split": True,
                    "uses_holdout": True,
                    "dev_summary": {"replay_url_accuracy_pct": None},
                    "holdout_summary": {"replay_url_accuracy_pct": 79.0},
                },
                "comparison_summary": {
                    "previous_run": {
                        "available": True,
                        "metric_deltas": [
                            {"metric_key": "dev_replay_url_accuracy_pct", "direction": "regressed"},
                        ],
                    },
                    "holdout_baseline": {
                        "available": False,
                        "metric_deltas": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "RUN_SCORECARD_PATH", str(scorecard_path))

    pipeline._log_copy_dev_to_prod_evaluation_warnings()


def test_parallel_crawler_group_includes_images() -> None:
    assert pipeline.PARALLEL_CRAWLER_STEPS == {"rd_ext", "ebs", "scraper", "fb", "images"}


def test_refresh_manual_coverage_audit_csv_rewrites_file(tmp_path) -> None:
    output_path = tmp_path / "manual_coverage_audit.csv"

    class _FakeDbHandler:
        def execute_query(self, query, params=None):
            query_text = str(query)
            assert "FROM events" in query_text
            assert "CURRENT_DATE + INTERVAL '7 days'" in query_text
            assert "CURRENT_DATE + INTERVAL '21 days'" in query_text
            assert params == {"limit": 2}
            return [
                ("Source A", "https://example.com/a", "Friday Social", "2026-04-10"),
                ("Source B", "https://example.com/b", "Saturday Salsa", "2026-04-11"),
            ]

    result = pipeline.refresh_manual_coverage_audit_csv(
        sample_size=2,
        output_path=str(output_path),
        db_handler=_FakeDbHandler(),
    )

    assert result["rows_written"] == 2
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["source_name"] == "Source A"
    assert rows[0]["expected_present"] == "True"
    assert rows[0]["active"] == "True"
    assert "Auto-generated from current events table" in rows[0]["notes"]
    assert "+7d to +21d window" in rows[0]["notes"]


def test_derive_log_archive_timestamp_prefers_credential_validator_log(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "credential_validator_log.txt").write_text(
        "2026-03-24 14:30:20 - INFO - credential validation started\n"
        "2026-03-24 14:30:25 - INFO - credential validation finished\n",
        encoding="utf-8",
    )
    (logs_dir / "pipeline_log.txt").write_text(
        "2026-03-24 14:30:55 - INFO - pipeline started\n",
        encoding="utf-8",
    )

    timestamp = pipeline._derive_log_archive_timestamp(str(logs_dir))

    assert timestamp == "20260324_143020"


def test_derive_log_archive_timestamp_falls_back_to_first_available_log(tmp_path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "scraper_log.txt").write_text(
        "2026-03-22 19:36:32 - INFO - scraper started\n",
        encoding="utf-8",
    )

    timestamp = pipeline._derive_log_archive_timestamp(str(logs_dir))

    assert timestamp == "20260322_193632"


def test_pipeline_steps_refresh_manual_coverage_audit_after_copy_dev_to_prod() -> None:
    step_names = [name for name, _ in pipeline.PIPELINE_STEPS]
    copy_index = step_names.index("copy_dev_to_prod")
    assert step_names[copy_index + 1] == "refresh_manual_coverage_audit"
