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
        "start_ts": "2026-03-13 01:00:00",
        "end_ts": "2026-03-13 03:00:00",
        "endpoint_used": "https://openrouter.ai/api/v1/activity",
        "cost_usd": 12.5,
        "requests": 500,
        "tokens": 150000,
    }
    second = {
        "start_ts": "2026-03-14 01:00:00",
        "end_ts": "2026-03-14 03:00:00",
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
