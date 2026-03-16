import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "validation"))

from test_runner import ValidationTestRunner


def test_snapshot_delta_uses_window_total_on_first_snapshot(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    summary = {
        "start_ts": "2026-03-13 01:00:00",
        "end_ts": "2026-03-13 03:00:00",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 12.5,
        "requests": 500,
        "tokens": 150000,
    }
    out = runner._apply_snapshot_delta_cost("openrouter", summary, str(tmp_path))
    assert out["cost_usd"] == 12.5
    assert out["requests"] == 500
    assert out["tokens"] == 150000
    assert out["cost_basis"] == "window_total_api"


def test_snapshot_delta_returns_delta_on_subsequent_snapshot(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    first = {
        "start_ts": "2030-03-13 01:00:00",
        "end_ts": "2030-03-13 03:00:00",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 12.5,
        "requests": 500,
        "tokens": 150000,
    }
    second = {
        "start_ts": "2030-03-14 01:00:00",
        "end_ts": "2030-03-14 03:00:00",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 15.0,
        "requests": 580,
        "tokens": 180000,
    }

    runner._apply_snapshot_delta_cost("openrouter", first, str(tmp_path))
    out = runner._apply_snapshot_delta_cost("openrouter", second, str(tmp_path))

    assert out["cost_usd"] == 2.5
    assert out["requests"] == 80
    assert out["tokens"] == 30000
    assert out["cost_basis"] == "delta_from_snapshot"


def test_snapshot_delta_uses_pre_run_baseline_not_latest_same_run_snapshot(tmp_path) -> None:
    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    pre_run = {
        "start_ts": "2030-03-15 17:00:00",
        "end_ts": "2030-03-15 18:00:00",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 9.8,
        "requests": 5200,
        "tokens": 30800000,
    }
    run_window = {
        "start_ts": "2030-03-15 19:11:03",
        "end_ts": "2030-03-15 22:44:02",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 11.0,
        "requests": 5600,
        "tokens": 32500000,
    }

    # First snapshot before run start.
    runner._apply_snapshot_delta_cost("openrouter", pre_run, str(tmp_path))
    # First report generation during/after run writes same-window snapshot.
    first_pass = runner._apply_snapshot_delta_cost("openrouter", dict(run_window), str(tmp_path))
    # Re-generating the same report should keep the same delta (not collapse to zero).
    second_pass = runner._apply_snapshot_delta_cost("openrouter", dict(run_window), str(tmp_path))

    assert first_pass["cost_basis"] == "delta_from_snapshot"
    assert second_pass["cost_basis"] == "delta_from_snapshot"
    assert first_pass["cost_usd"] == 1.2
    assert second_pass["cost_usd"] == 1.2
