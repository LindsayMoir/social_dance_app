import sys

import pandas as pd

sys.path.insert(0, "src")

from db import DatabaseHandler


def test_live_music_without_explicit_style_is_null() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_type": "live music",
                "event_name": "Wednesday Night Jam",
                "description": "Doors at 7pm. Great local band.",
                "dance_style": "rumba",
            }
        ]
    )
    out = handler._enforce_live_music_dance_style_policy(df)
    assert pd.isna(out.loc[0, "dance_style"])


def test_live_music_with_explicit_style_keeps_detected_style() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_type": "live music",
                "event_name": "Latin Night",
                "description": "Live rumba and salsa set all evening.",
                "dance_style": "rumba, tango",
            }
        ]
    )
    out = handler._enforce_live_music_dance_style_policy(df)
    assert out.loc[0, "dance_style"] == "rumba, salsa"


def test_non_live_music_not_modified() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    df = pd.DataFrame(
        [
            {
                "event_type": "social dance",
                "event_name": "Friday Social",
                "description": "Beginner lesson and dance.",
                "dance_style": "west coast swing",
            }
        ]
    )
    out = handler._enforce_live_music_dance_style_policy(df)
    assert out.loc[0, "dance_style"] == "west coast swing"
