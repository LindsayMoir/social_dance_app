import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from clean_up import CleanUp


class _StubDBHandler:
    _DANCE_STYLE_TOKENS = {
        "argentine tango",
        "tango",
        "salsa",
        "bachata",
        "kizomba",
        "semba",
        "urban kiz",
        "tarraxo",
        "tarraxa",
        "tarraxinha",
        "merengue",
        "rumba",
        "swing",
        "west coast swing",
        "wcs",
        "east coast swing",
        "lindy",
        "lindy hop",
        "balboa",
    }


def _make_clean_up() -> CleanUp:
    clean_up = CleanUp.__new__(CleanUp)
    clean_up.db_handler = _StubDBHandler()
    return clean_up


def test_update_dance_style_live_music_requires_explicit_style_evidence() -> None:
    clean_up = _make_clean_up()
    row = pd.Series(
        {
            "event_name": "Friday Night Live Music",
            "description": "Doors at 8pm. Great band and cocktails.",
            "source": "The Coda",
            "event_type": "live music",
            "dance_style": "rumba",
            "update": False,
        }
    )

    updated = clean_up.update_dance_style(row, keywords_list=["dance", "rumba", "live music"])

    assert pd.isna(updated.iloc[0])
    assert bool(updated.iloc[1]) is True


def test_update_dance_style_live_music_keeps_explicit_style_from_event_text() -> None:
    clean_up = _make_clean_up()
    row = pd.Series(
        {
            "event_name": "Salsa Night with Live Music",
            "description": "Join us for salsa dancing until midnight.",
            "source": "Venue",
            "event_type": "social dance, live music",
            "dance_style": "",
            "update": False,
        }
    )

    updated = clean_up.update_dance_style(row, keywords_list=["dance", "rumba", "live music"])

    assert updated.iloc[0] == "salsa"
    assert bool(updated.iloc[1]) is True


def test_normalize_binary_label_series_extracts_binary_values_from_messy_strings() -> None:
    labels = pd.Series(["0", "1", "博士候选人: 0", "Label=1", "invalid"])

    normalized = CleanUp._normalize_binary_label_series(labels)

    assert normalized.iloc[0] == 0
    assert normalized.iloc[1] == 1
    assert normalized.iloc[2] == 0
    assert normalized.iloc[3] == 1
    assert pd.isna(normalized.iloc[4])
