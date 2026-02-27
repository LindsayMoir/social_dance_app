from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm import LLMHandler


def test_chatbot_openai_promoted_every_third_request():
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {
        "llm": {
            "chatbot_provider_order": ["openrouter", "gemini", "openai"],
        }
    }
    handler.chatbot_openai_first_every_n_requests = 3
    handler.chatbot_request_counter = 0

    order_1 = handler._chatbot_provider_order()
    order_2 = handler._chatbot_provider_order()
    order_3 = handler._chatbot_provider_order()
    order_4 = handler._chatbot_provider_order()

    assert order_1 == ["openrouter", "gemini", "openai"]
    assert order_2 == ["openrouter", "gemini", "openai"]
    assert order_3 == ["openai", "openrouter", "gemini"]
    assert order_4 == ["openrouter", "gemini", "openai"]


def test_chatbot_openai_cadence_disabled_uses_configured_order():
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {
        "llm": {
            "chatbot_provider_order": ["openrouter", "openai", "gemini"],
        }
    }
    handler.chatbot_openai_first_every_n_requests = 0
    handler.chatbot_request_counter = 0

    order_1 = handler._chatbot_provider_order()
    order_2 = handler._chatbot_provider_order()

    assert order_1 == ["openrouter", "openai", "gemini"]
    assert order_2 == ["openrouter", "openai", "gemini"]
