import sys
from datetime import date

import pytest

pytest.importorskip("playwright")

sys.path.insert(0, "src")
from rd_ext import ReadExtract


def test_next_weekday_date_returns_valid_weekday() -> None:
    reader = ReadExtract.__new__(ReadExtract)
    next_wed = reader._next_weekday_date("Wednesday", include_today=True)
    assert isinstance(next_wed, date)
    assert next_wed.weekday() == 2


def test_resolve_synthetic_event_base_from_inline_event() -> None:
    reader = ReadExtract.__new__(ReadExtract)
    reader.config = {}

    resolved = reader._resolve_synthetic_event_base(
        {
            "event": {
                "source": "Uvic Cuban Salsa Club",
                "event_name": "UVic Salsa Rueda",
            }
        }
    )
    assert resolved["source"] == "Uvic Cuban Salsa Club"
    assert resolved["event_name"] == "UVic Salsa Rueda"
