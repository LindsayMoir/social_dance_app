import os
import sys
from datetime import datetime

sys.path.insert(0, "tests/validation")

from test_runner import ValidationTestRunner


def _ts_line(message: str) -> str:
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}\n"


def test_summarize_llm_extraction_quality_from_logs(tmp_path, monkeypatch):
    log_path = tmp_path / "llm_quality.log"
    lines = [
        _ts_line("[root] INFO: query_llm(): Querying OpenRouter model qwen/qwen3-coder"),
        _ts_line("[root] INFO: def process_llm_response: URL https://a.example/events attempt=1 llm_response_len=18"),
        _ts_line("[root] WARNING: def process_llm_response: URL https://a.example/events attempt=1 returned too-short response (18 <= 100); treating as non-fatal miss."),
        _ts_line("[root] WARNING: def process_llm_response: URL https://a.example/events attempt=1 returned non-parseable response; retrying once with strict JSON instruction."),
        _ts_line("[root] INFO: query_llm(): Querying OpenRouter model qwen/qwen3-coder"),
        _ts_line("[root] INFO: def process_llm_response: URL https://a.example/events attempt=2 llm_response_len=420"),
        _ts_line("[root] INFO: def process_llm_response: URL https://a.example/events marked as relevant with events written to the database."),
        _ts_line("[root] INFO: query_llm(): Querying OpenAI model gpt-4o-mini"),
        _ts_line("[root] INFO: def process_llm_response: URL https://b.example/classes attempt=1 llm_response_len=72"),
        _ts_line("[root] ERROR: def process_llm_response: Failed to process LLM response for URL: https://b.example/classes"),
    ]
    log_path.write_text("".join(lines), encoding="utf-8")

    runner = ValidationTestRunner.__new__(ValidationTestRunner)
    runner.validation_config = {"reporting": {"llm_activity_hours": 24}}

    monkeypatch.setattr(
        runner,
        "_get_llm_activity_log_files",
        lambda: [os.fspath(log_path)],
    )

    summary = runner._summarize_llm_extraction_quality(datetime.now().isoformat())

    assert summary["total_urls"] == 2
    assert summary["successful_urls"] == 1
    assert summary["hard_failed_urls"] == 1
    assert summary["too_short_urls"] == 0  # final URL status is success/failure, not transient
    assert summary["total_retries"] == 1
    assert summary["parse_success_rate"] == 0.5
    assert any(model_key.startswith("openrouter:") for model_key, _ in summary["models"])
