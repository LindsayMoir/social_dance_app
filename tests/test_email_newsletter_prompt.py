import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from emails import _EMAIL_NEWSLETTER_PROMPT_TYPE


def test_every_configured_email_uses_the_common_newsletter_prompt() -> None:
    with open("data/other/emails.csv", encoding="utf-8-sig", newline="") as email_file:
        rows = list(csv.DictReader(email_file))

    assert rows
    assert set(rows[0]) == {"email", "source", "keywords"}
    assert _EMAIL_NEWSLETTER_PROMPT_TYPE == "email_newsletter"
