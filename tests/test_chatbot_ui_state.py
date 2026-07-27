from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chatbot_ui_state import ensure_pending_confirmation_message


def test_ensure_pending_confirmation_message_rehydrates_missing_assistant_reply() -> None:
    session_state = {
        "pending_confirmation": {
            "message": "Please confirm.",
            "interpretation": "You want events tonight.",
            "options": ["yes", "clarify", "no"],
        },
        "messages": [
            {"role": "user", "content": "Where can I dance tonight?"},
        ],
    }

    changed = ensure_pending_confirmation_message(session_state)

    assert changed is True
    assert len(session_state["messages"]) == 2
    assistant = session_state["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["pending_confirmation"] is True
    assert assistant["content"] == "Please confirm."


def test_ensure_pending_confirmation_message_is_idempotent() -> None:
    session_state = {
        "pending_confirmation": {
            "message": "Please confirm.",
            "interpretation": "You want events tonight.",
            "options": ["yes", "clarify", "no"],
        },
        "messages": [
            {"role": "user", "content": "Where can I dance tonight?"},
            {"role": "assistant", "content": "Please confirm.", "pending_confirmation": True},
        ],
    }

    changed = ensure_pending_confirmation_message(session_state)

    assert changed is False
    assert len(session_state["messages"]) == 2
