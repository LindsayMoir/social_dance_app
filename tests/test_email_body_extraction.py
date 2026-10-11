"""Regression coverage for newsletter content passed to event extraction."""

import base64
import os
import sys
from email.message import EmailMessage
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emails import GmailProcessor


def fetch_message(message: EmailMessage) -> str | None:
    """Fetch a MIME fixture through the Gmail API boundary without external I/O."""
    processor = GmailProcessor.__new__(GmailProcessor)
    processor.service = Mock()
    messages = processor.service.users.return_value.messages.return_value
    messages.list.return_value.execute.return_value = {"messages": [{"id": "newsletter"}]}
    messages.get.return_value.execute.return_value = {
        "raw": base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
    }
    return processor.fetch_latest_email("newsletter@example.com")


def test_html_social_dances_survive_plain_text_class_alternative() -> None:
    message = EmailMessage()
    message.set_content("West Coast Swing class: October 8, 7:30-8:30 pm.")
    message.add_alternative(
        "<html><head><style>hidden styling</style></head><body>"
        "<p>West Coast Swing class: October 8, 7:30-8:30 pm.</p>"
        "<h2>DJ social dances</h2><p>October 10 and 24, 8-11 pm.</p>"
        "<p>Social dancing after the October 8 lesson: 8:30-10 pm.</p>"
        "<script>hidden scripting</script></body></html>",
        subtype="html",
    )
    message.add_attachment("Unrelated attachment: October 31", filename="notes.txt")

    text = fetch_message(message)

    assert text is not None
    assert "West Coast Swing class" in text
    assert "DJ social dances" in text
    assert "October 10 and 24, 8-11 pm." in text
    assert "8:30-10 pm." in text
    assert "hidden styling" not in text
    assert "hidden scripting" not in text
    assert "Unrelated attachment" not in text


def test_identical_body_alternatives_are_not_repeated() -> None:
    message = EmailMessage()
    message.set_content("Social dance: October 10, 8 pm.")
    message.add_alternative("<p>Social dance: October 10, 8 pm.</p>", subtype="html")

    assert fetch_message(message) == "Social dance: October 10, 8 pm."


@pytest.mark.parametrize("subtype", ["plain", "html"])
def test_single_part_body_respects_declared_charset(subtype: str) -> None:
    message = EmailMessage()
    content = "Social dance at the café: October 10."
    message.set_content(
        f"<p>{content}</p>" if subtype == "html" else content,
        subtype=subtype,
        charset="iso-8859-1",
    )

    assert fetch_message(message) == content


def test_empty_body_returns_none() -> None:
    message = EmailMessage()
    message.set_content("")

    assert fetch_message(message) is None


def test_unknown_charset_falls_back_to_utf8() -> None:
    message = EmailMessage()
    message.set_content("Social dance: October 10, 8 pm.")
    message.set_param("charset", "unknown-newsletter-charset")

    assert fetch_message(message) == "Social dance: October 10, 8 pm."
