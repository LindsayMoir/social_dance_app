from __future__ import annotations

import json
from pathlib import Path

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
