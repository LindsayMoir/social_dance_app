import sys

sys.path.insert(0, "src")

from db import DatabaseHandler


def test_meaningful_token_overlap_blocks_loft_lab_false_match() -> None:
    assert DatabaseHandler._has_meaningful_token_overlap("The Loft Victoria", "The Lab Victoria") is False


def test_meaningful_token_overlap_allows_loft_variants() -> None:
    assert DatabaseHandler._has_meaningful_token_overlap("The Loft Victoria", "The Loft Pub") is True


def test_find_address_by_building_name_skips_false_positive_when_only_lab_exists() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.execute_query = lambda *_args, **_kwargs: [(101, "The Lab Victoria")]  # type: ignore[attr-defined]

    match = handler.find_address_by_building_name("The Loft Victoria", threshold=75)
    assert match is None


def test_find_address_by_building_name_still_finds_true_loft_match() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.execute_query = lambda *_args, **_kwargs: [  # type: ignore[attr-defined]
        (101, "The Lab Victoria"),
        (202, "The Loft Victoria"),
    ]

    match = handler.find_address_by_building_name("The Loft Victoria", threshold=75)
    assert match == 202
