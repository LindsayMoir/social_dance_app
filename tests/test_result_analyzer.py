import json

import pandas as pd

from tests.validation import result_analyzer as result_analyzer_module


class DummyLLMHandler:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def query_llm(self, url, prompt, schema_type=None):
        self.calls.append((url, prompt, schema_type))
        return json.dumps(
            {
                "summary": {
                    "total_issues_identified": 1,
                    "critical_issues": 0,
                    "high_priority_issues": 1,
                    "medium_priority_issues": 0,
                    "low_priority_issues": 0,
                },
                "recurring_patterns": [],
                "priority_recommendations": [],
                "acceptable_issues": [],
            }
        )


def _make_raw_chatbot_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": "Dance events on Wednesday",
                "category": "day_specific",
                "sql_query": None,
                "sql_syntax_valid": False,
                "execution_success": False,
                "result_count": 0,
                "error": "query_for_interpretation",
                "timestamp": "2026-05-02T18:54:36.784443",
            },
            {
                "question": "Show me social dances tonight",
                "category": "event_type_only",
                "sql_query": None,
                "sql_syntax_valid": False,
                "execution_success": True,
                "result_count": 4,
                "error": "",
                "timestamp": "2026-05-02T18:54:46.727227",
            },
        ]
    )


def test_prepare_results_summary_handles_raw_results_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(result_analyzer_module, "LLMHandler", DummyLLMHandler)
    prompt_path = tmp_path / "result_analysis_prompt.txt"
    prompt_path.write_text("Results summary:\n{test_results_summary}\n")

    analyzer = result_analyzer_module.ResultAnalyzer(
        config_path="config/config.yaml",
        prompt_path=str(prompt_path),
    )

    summary = analyzer.prepare_results_summary(_make_raw_chatbot_results_df())

    assert "Average score: unavailable" in summary
    assert "Score data unavailable in this results file" in summary
    assert "PROBLEMATIC TESTS (1 tests):" in summary
    assert "Question: Dance events on Wednesday" in summary
    assert "Score:" not in summary


def test_analyze_results_handles_raw_results_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(result_analyzer_module, "LLMHandler", DummyLLMHandler)
    prompt_path = tmp_path / "result_analysis_prompt.txt"
    prompt_path.write_text("Results summary:\n{test_results_summary}\n")

    analyzer = result_analyzer_module.ResultAnalyzer(
        config_path="config/config.yaml",
        prompt_path=str(prompt_path),
    )
    df = _make_raw_chatbot_results_df()

    analysis = analyzer.analyze_results(df)

    assert analysis["summary"]["total_issues_identified"] == 1
    assert analysis["summary"]["high_priority_issues"] == 1
    assert analyzer.llm_handler.calls
    assert analyzer.llm_handler.calls[0][0] == "test_results_analysis"
