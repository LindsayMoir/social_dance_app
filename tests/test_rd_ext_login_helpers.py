import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rd_ext import _wait_for_login_completion


class _FakePage:
    def __init__(self, urls, selector_states):
        self._urls = list(urls)
        self._selector_states = list(selector_states)
        self.url = self._urls[0]
        self._index = 0

    async def query_selector(self, selector):
        state = self._selector_states[min(self._index, len(self._selector_states) - 1)]
        return object() if state.get(selector, False) else None

    async def wait_for_timeout(self, _ms):
        if self._index < len(self._urls) - 1:
            self._index += 1
            self.url = self._urls[self._index]


def test_wait_for_login_completion_succeeds_after_redirect() -> None:
    page = _FakePage(
        urls=[
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/organizer/home/",
        ],
        selector_states=[
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": False},
            {"input#email": False, "input#password": False},
        ],
    )

    result = asyncio.run(
        _wait_for_login_completion(
            page,
            login_url="https://www.eventbrite.com/signin/",
            email_selector="input#email",
            pass_selector="input#password",
            timeout_ms=5000,
        )
    )

    assert result is True


def test_wait_for_login_completion_fails_when_login_controls_persist() -> None:
    page = _FakePage(
        urls=[
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
            "https://www.eventbrite.com/signin/",
        ],
        selector_states=[
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": True},
            {"input#email": True, "input#password": True},
        ],
    )

    result = asyncio.run(
        _wait_for_login_completion(
            page,
            login_url="https://www.eventbrite.com/signin/",
            email_selector="input#email",
            pass_selector="input#password",
            timeout_ms=1500,
        )
    )

    assert result is False
