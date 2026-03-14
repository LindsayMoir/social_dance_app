import sys

import pandas as pd

sys.path.insert(0, "src")

from llm import LLMHandler


def test_resolve_effective_event_url_prefers_row_url() -> None:
    out = LLMHandler._resolve_effective_event_url(
        row_url="https://row.example/event/1",
        url="https://page.example/events",
        parent_url="https://parent.example/listing",
    )
    assert out == "https://row.example/event/1"


def test_resolve_effective_event_url_falls_back_to_page_then_parent() -> None:
    out_page = LLMHandler._resolve_effective_event_url(
        row_url="",
        url="https://page.example/events",
        parent_url="https://parent.example/listing",
    )
    assert out_page == "https://page.example/events"

    out_parent = LLMHandler._resolve_effective_event_url(
        row_url="",
        url="",
        parent_url="https://parent.example/listing",
    )
    assert out_parent == "https://parent.example/listing"


def test_apply_url_context_to_events_df_stamps_url_for_each_row() -> None:
    handler = LLMHandler.__new__(LLMHandler)
    df = pd.DataFrame(
        [
            {"event_name": "Event A", "url": ""},
            {"event_name": "Event B", "url": "https://row.example/b"},
            {"event_name": "Event C"},
        ]
    )
    out = handler._apply_url_context_to_events_df(
        events_df=df,
        url="https://page.example/events",
        parent_url="https://parent.example/listing",
    )
    assert len(out) == 3
    assert out.loc[0, "url"] == "https://page.example/events"
    assert out.loc[1, "url"] == "https://row.example/b"
    assert out.loc[2, "url"] == "https://page.example/events"
