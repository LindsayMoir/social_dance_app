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


def test_resolve_or_insert_address_rejects_loft_to_lab_street_fuzzy_match() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    captured = {"insert_called": False}

    def _execute(query, params=None):  # type: ignore[no-untyped-def]
        q = str(query)
        if "FROM address" in q and "LOWER(street_number)" in q and "street_name_alt" in q:
            return [(101, "The Lab Victoria", "729", "Yates", "V8W 1L6")]
        if "FROM address" in q and "LOWER(city)" in q and "building_name IS NOT NULL" in q:
            return [(101, "The Lab Victoria", "Victoria", "V8W 1L6")]
        if "SELECT address_id, building_name FROM address WHERE building_name IS NOT NULL" in q:
            return [(101, "The Lab Victoria")]
        if "INSERT INTO address" in q:
            captured["insert_called"] = True
            return [(777,)]
        if "SELECT address_id FROM address WHERE full_address = :full_address" in q:
            return []
        return []

    handler.execute_query = _execute  # type: ignore[attr-defined]
    handler._find_address_alias_match = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    handler.normalize_nulls = lambda x: x  # type: ignore[attr-defined]
    handler.find_address_by_building_name = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

    parsed = {
        "building_name": "The Loft Victoria",
        "street_number": "729",
        "street_name": "Yates",
        "city": "Victoria",
        "province_or_state": "BC",
        "country_id": "CA",
        "postal_code": "V8W 1L6",
    }
    result = handler.resolve_or_insert_address(parsed)
    assert result == 777
    assert captured["insert_called"] is True


def test_lookup_raw_location_ignores_stale_loft_to_lab_cache() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)
    handler.execute_query = lambda *_args, **_kwargs: [(101, "The Lab Victoria", "729", "Yates", None, "The Lab Victoria, 729 Yates, Victoria, BC, CA")]  # type: ignore[attr-defined]
    handler._stale_raw_location_warnings = set()  # type: ignore[attr-defined]

    result = handler.lookup_raw_location("The Loft Victoria, Fort, Victoria, BC, CA")
    assert result is None


def test_build_full_address_preserves_direction_token() -> None:
    handler = DatabaseHandler.__new__(DatabaseHandler)

    result = handler.build_full_address(
        building_name="The Relocation Experiment",
        street_number="154",
        direction="E",
        street_name="10th",
        city="Vancouver",
        province_or_state="BC",
        postal_code="V5T 1Z4",
        country_id="CA",
    )

    assert result == "The Relocation Experiment, 154 E 10th, Vancouver, BC V5T 1Z4, CA"


def test_address_text_supports_candidate_rejects_mismatched_quick_lookup() -> None:
    assert DatabaseHandler._address_text_supports_candidate(
        "Slovenian Society, 5762 Sprott Street Burnaby, BC V5G 1X5",
        {
            "building_name": "Slovenian Society",
            "street_number": "5751",
            "street_name": "Forest",
            "direction": None,
            "full_address": "5751 Forest, Burnaby, BC V5G 1X5, CA",
        },
    ) is False
