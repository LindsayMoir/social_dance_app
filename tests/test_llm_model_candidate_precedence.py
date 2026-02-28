import sys

sys.path.insert(0, "src")

from llm import LLMHandler


def test_provider_model_candidates_take_precedence_over_alias_map():
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {
        "llm": {
            "model_alias": "default_fast",
            "model_candidates": {
                "openrouter": {
                    "default_fast": [
                        "qwen/qwen3-coder",
                        "deepseek/deepseek-v3.2",
                    ]
                }
            },
            "openrouter_model_candidates": [
                "deepseek/deepseek-v3.2",
                "qwen/qwen3-coder",
            ],
            "openrouter_model": "openai/gpt-oss-120b:free",
        }
    }

    assert handler._provider_model_candidates("openrouter") == [
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-coder",
    ]
