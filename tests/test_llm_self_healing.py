from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from llm import LLMHandler


def _build_minimal_handler() -> LLMHandler:
    handler = LLMHandler.__new__(LLMHandler)
    handler.self_healing_enabled = True
    handler.provider_retry_max_attempts = 2
    handler.provider_retry_base_delay_seconds = 0.0
    handler.provider_retry_jitter_seconds = 0.0
    handler.provider_retry_max_delay_seconds = 0.0
    handler._openai_in_timeout_cooldown = lambda: False
    handler._openrouter_in_cooldown = lambda: False
    handler._mistral_in_cooldown = lambda: False
    handler._gemini_in_timeout_cooldown = lambda: False
    handler._record_openai_result = lambda error=None: None
    handler._record_openrouter_result = lambda error=None: None
    handler._record_mistral_result = lambda error=None: None
    handler._record_gemini_result = lambda error=None: None
    handler._resolve_model_for_provider = lambda provider, force_refresh=False: "test-model"
    return handler


def test_openai_transient_error_retries_then_succeeds():
    handler = _build_minimal_handler()
    calls = {"count": 0}

    def fake_query_openai(prompt, model, schema_type=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("read timeout")
        return '{"ok": true}'

    handler.query_openai = fake_query_openai

    result = handler._query_provider_once("openai", "prompt", None, request_url="chatbot")

    assert result == '{"ok": true}'
    assert calls["count"] == 2


def test_openai_non_transient_error_does_not_retry():
    handler = _build_minimal_handler()
    calls = {"count": 0}

    def fake_query_openai(prompt, model, schema_type=None):
        calls["count"] += 1
        raise RuntimeError("invalid request schema")

    handler.query_openai = fake_query_openai

    try:
        handler._query_provider_once("openai", "prompt", None, request_url="chatbot")
    except RuntimeError as exc:
        assert "invalid request schema" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError to be raised")

    assert calls["count"] == 1
