"""
Local test to validate LLM tool-call normalization without network or DB.

It monkeypatches LLMHandler to avoid DB/API construction and stubs the
provider tool-call methods to simulate:
 - Iteration 1: provider returns a tool call with an invalid id and no type
 - Iteration 2: provider returns final content

Expected: _query_with_tools completes without 400 errors and returns content.
"""
import json
import sys

sys.path.insert(0, 'src')

from logging_config import setup_logging  # noqa: E402
from llm import LLMHandler  # noqa: E402
from date_calculator import CALCULATE_DATE_RANGE_TOOL  # noqa: E402


def main():
    # Configure logging
    setup_logging('tool_call_normalization')

    # Monkeypatch __init__ to avoid DB/API clients
    orig_init = LLMHandler.__init__

    def fake_init(self, config_path=None):
        self.config = {
            'llm': {
                'openai_model': 'gpt-5-mini',
                'mistral_model': 'mistral-large-2512',
                'provider': 'mistral',
            }
        }

    LLMHandler.__init__ = fake_init  # type: ignore

    try:
        handler = LLMHandler(config_path='config/config.yaml')

        calls = {"mistral": 0, "openai": 0}

        def stub_mistral_with_tools(messages, tools):
            calls["mistral"] += 1
            if calls["mistral"] == 1:
                # Return a tool call with an invalid id to exercise normalization
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_@@invalidID12345",
                            "function": {
                                "name": "calculate_date_range",
                                "arguments": json.dumps(
                                    {"temporal_phrase": "tonight", "current_date": "2026-02-01"}
                                ),
                            },
                        }
                    ],
                }
            # After tool result, provider returns final content
            return {"content": "SELECT 1"}

        def stub_openai_with_tools(messages, tools):
            calls["openai"] += 1
            return None

        # Inject stubs
        handler._query_mistral_with_tools = stub_mistral_with_tools  # type: ignore
        handler._query_openai_with_tools = stub_openai_with_tools  # type: ignore

        res = handler._query_with_tools(
            url='', prompt='test', tools=[CALCULATE_DATE_RANGE_TOOL], max_iterations=3
        )
        import logging
        logging.info('Result: %s', res)
        logging.info('Calls: %s', calls)

    finally:
        # Restore __init__
        LLMHandler.__init__ = orig_init  # type: ignore


if __name__ == '__main__':
    main()
