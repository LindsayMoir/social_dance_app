"""Helpers for keeping chatbot UI session state coherent across reruns."""

from __future__ import annotations

from typing import Any


def ensure_pending_confirmation_message(session_state: dict[str, Any]) -> bool:
    """
    Rehydrate a missing assistant confirmation message from pending state.

    Streamlit reruns can occasionally leave the user message persisted while the
    paired assistant confirmation message is absent. When that happens, the UI
    looks like the chatbot never replied even though the backend returned a valid
    confirmation payload. This helper restores the assistant message idempotently.
    """
    pending = session_state.get("pending_confirmation")
    messages = session_state.get("messages")

    if not isinstance(pending, dict) or not isinstance(messages, list) or not messages:
        return False

    last_message = messages[-1]
    if not isinstance(last_message, dict):
        return False

    if last_message.get("role") == "assistant" and last_message.get("pending_confirmation"):
        return False

    if last_message.get("role") != "user":
        return False

    messages.append(
        {
            "role": "assistant",
            "content": pending.get("message", "Please confirm your query."),
            "events": [],
            "pending_confirmation": True,
            "interpretation": pending.get("interpretation", ""),
            "options": pending.get("options", ["yes", "clarify", "no"]),
            "timestamp": last_message.get("content", ""),
        }
    )
    return True
