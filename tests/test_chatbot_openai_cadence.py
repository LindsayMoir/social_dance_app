from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm import LLMHandler


def test_regular_openai_promoted_every_third_request():
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {
        "llm": {
            "fallback_enabled": True,
            "fallback_provider_order": ["mistral", "openrouter", "openai", "gemini"],
        }
    }
    handler.regular_openai_first_every_n_requests = 3
    handler.regular_request_counter = 0

    order_1 = handler._candidate_providers_for_request("https://example.com/events", "mistral")
    order_2 = handler._candidate_providers_for_request("https://example.com/events", "mistral")
    order_3 = handler._candidate_providers_for_request("https://example.com/events", "mistral")
    order_4 = handler._candidate_providers_for_request("https://example.com/events", "mistral")

    assert order_1 == ["mistral", "openrouter", "openai", "gemini"]
    assert order_2 == ["mistral", "openrouter", "openai", "gemini"]
    assert order_3 == ["openai", "mistral", "openrouter", "gemini"]
    assert order_4 == ["mistral", "openrouter", "openai", "gemini"]


def test_chatbot_order_unaffected_by_regular_cadence_setting():
    handler = LLMHandler.__new__(LLMHandler)
    handler.config = {
        "llm": {
            "chatbot_provider_order": ["openrouter", "gemini", "openai"],
            "fallback_enabled": True,
            "fallback_provider_order": ["mistral", "openrouter", "openai", "gemini"],
        }
    }
    handler.regular_openai_first_every_n_requests = 3
    handler.regular_request_counter = 0

    order_1 = handler._candidate_providers_for_request("https://chatbot.local/query", "mistral")
    order_2 = handler._candidate_providers_for_request("https://chatbot.local/query", "mistral")

    assert order_1 == ["openrouter", "gemini", "openai"]
    assert order_2 == ["openrouter", "gemini", "openai"]
