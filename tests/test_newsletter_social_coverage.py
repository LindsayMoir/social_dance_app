"""Independent source inventory and partial social extraction regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from config_runtime import load_config
from llm import LLMHandler

SOURCE = (
    'October 8 and 15: Thursday classes at Riverside Hall.\n'
    '7:30pm - 8:30pm Class; 8:30pm - 9:45pm Social Dancing.\n'
    'October 15: Social Dancing 8:30pm - 9:45pm at Uptown Hall.\n'
    'Resources: J&J Social Dancing, dates to be announced.'
)


def slot(day: int, venue: str = 'Riverside Hall') -> dict:
    return {
        'start_date': f'2026-10-{day:02d}', 'start_time': '20:30', 'end_time': '21:45',
        'location': venue, 'separate': True,
        'evidence': [
            'October 8 and 15: Thursday classes at Riverside Hall.' if venue == 'Riverside Hall'
            else 'October 15: Social Dancing 8:30pm - 9:45pm at Uptown Hall.',
            '8:30pm - 9:45pm Social Dancing.' if venue == 'Riverside Hall'
            else 'Social Dancing 8:30pm - 9:45pm at Uptown Hall.',
        ],
    }


def event(day: int, venue: str = 'Riverside Hall', **overrides) -> dict:
    return dict({
        'event_name': 'A different social title', 'event_type': 'social dance',
        'start_date': f'2026-10-{day:02d}', 'start_time': '8:30 PM', 'end_time': '9:45 PM',
        'location': venue,
    }, **overrides)


@pytest.fixture
def handler() -> LLMHandler:
    instance = LLMHandler.__new__(LLMHandler)
    instance.config = load_config()
    instance.db_handler = SimpleNamespace(write_events_to_db=Mock(side_effect=lambda df, *a, **kw: len(df)))
    return instance


def run_extraction(handler: LLMHandler) -> object:
    return handler.process_llm_response(
        'newsletter@example.com', 'email inbox', SOURCE, 'Another organizer', [], 'email_newsletter'
    )


def test_partial_extraction_recovers_only_missing_slots(handler, monkeypatch) -> None:
    inventory = [slot(8), slot(15), slot(15, 'Uptown Hall')]
    responses = [json.dumps(inventory), json.dumps([event(8)]), json.dumps([
        event(8), event(15), event(15), event(15, 'Uptown Hall'), event(22),
    ])]
    query = Mock(side_effect=[(response, {'provider': 'test', 'model': 'fake'}) for response in responses])
    monkeypatch.setattr(handler, 'query_llm', query)

    result = run_extraction(handler)

    assert result.success
    assert result.events_written == 3
    assert query.call_count == 3
    inventory_prompt = query.call_args_list[0].args[1]
    assert 'independent inventory' in inventory_prompt
    assert SOURCE in inventory_prompt
    assert 'A different social title' not in inventory_prompt
    recovery_prompt = query.call_args_list[2].args[1]
    missing = json.loads(recovery_prompt.split('Missing slots:\n', 1)[1])
    assert {(s['start_date'], s['location']) for s in missing} == {
        ('2026-10-15', 'Riverside Hall'), ('2026-10-15', 'Uptown Hall'),
    }
    written = handler.db_handler.write_events_to_db.call_args.args[0]
    assert len(written) == 3
    assert '2026-10-22' not in set(written.start_date)
    assert 'evidence' not in written.columns


def test_five_advertised_three_extracted_recovers_two_missing_dates(handler, monkeypatch) -> None:
    source = 'October 1, 8, 15, 22, 29: Social Dancing 8:30pm - 9:45pm at Riverside Hall.'
    inventory = [dict(slot(day), evidence=[source]) for day in (1, 8, 15, 22, 29)]
    query = Mock(side_effect=[
        (json.dumps(inventory), {}),
        (json.dumps([event(day) for day in (1, 8, 15)]), {}),
        (json.dumps([event(day) for day in (22, 29)]), {}),
    ])
    monkeypatch.setattr(handler, 'query_llm', query)
    result = handler.process_llm_response('another@example.com', 'email inbox', source, 'Other source', [], 'email_newsletter')
    assert result.events_written == 5
    missing = json.loads(query.call_args.args[1].split('Missing slots:\n', 1)[1])
    assert {item['start_date'] for item in missing} == {'2026-10-22', '2026-10-29'}


@pytest.mark.parametrize('recovery', [[], [event(15)], [event(15, start_time='19:30')]])
def test_incomplete_recovery_never_writes_partial_batch(handler, monkeypatch, recovery, caplog) -> None:
    query = Mock(side_effect=[
        (json.dumps([slot(8), slot(15), slot(15, 'Uptown Hall')]), {}),
        (json.dumps([event(8)]), {}), (json.dumps(recovery), {}),
    ])
    monkeypatch.setattr(handler, 'query_llm', query)

    result = run_extraction(handler)

    assert not result
    assert result.decision_reason == 'newsletter_social_coverage_missing'
    handler.db_handler.write_events_to_db.assert_not_called()
    assert 'unresolved social slots' in caplog.text
    assert 'Uptown Hall' in caplog.text
    assert 'evidence' in caplog.text


@pytest.mark.parametrize('start_time,end_time,venue', [
    ('20:30', '21:45', 'RIVERSIDE HALL'),
    ('20:30:00', '21:45:00', 'Riverside Hall, 123 River Road'),
    ('8:30 p.m.', '9:45 p.m.', 'Riverside Hall'),
])
def test_complete_extraction_skips_recovery(handler, monkeypatch, start_time, end_time, venue) -> None:
    query = Mock(side_effect=[
        (json.dumps([slot(8)]), {}),
        (json.dumps([event(8, venue, start_time=start_time, end_time=end_time)]), {}),
    ])
    monkeypatch.setattr(handler, 'query_llm', query)
    assert run_extraction(handler).success
    assert query.call_count == 2


@pytest.mark.parametrize('response', [None, '', 'not json', '{}', '[null]', '[{}]'])
def test_invalid_inventory_does_not_write_or_extract(handler, monkeypatch, response) -> None:
    query = Mock(return_value=(response, {}))
    monkeypatch.setattr(handler, 'query_llm', query)
    result = run_extraction(handler)
    assert result.decision_reason == 'newsletter_social_inventory_failed'
    assert query.call_count == 1
    handler.db_handler.write_events_to_db.assert_not_called()


@pytest.mark.parametrize('change', [
    {'evidence': ['Invented source quote']}, {'evidence': []}, {'evidence': 'Social Dancing'},
    {'start_date': '2026-02-30'}, {'start_time': '25:00'}, {'end_time': 'sometime'},
    {'separate': 'true'}, {'location': 'Imaginary ballroom'},
])
def test_invalid_inventory_evidence_or_slot_fails_closed(handler, monkeypatch, change) -> None:
    monkeypatch.setattr(handler, 'query_llm', Mock(return_value=(json.dumps([dict(slot(8), **change)]), {})))
    assert run_extraction(handler).decision_reason == 'newsletter_social_inventory_failed'
    handler.db_handler.write_events_to_db.assert_not_called()


def test_duplicate_inventory_slots_do_not_duplicate_events(handler, monkeypatch) -> None:
    monkeypatch.setattr(handler, 'query_llm', Mock(side_effect=[
        (json.dumps([slot(8), slot(8)]), {}), (json.dumps([event(8)]), {}),
    ]))
    assert run_extraction(handler).events_written == 1


def test_no_dated_socials_does_not_trigger_recovery(handler, monkeypatch) -> None:
    query = Mock(side_effect=[
        ('[]', {}), (json.dumps([event(8, event_type='class')]), {}),
    ])
    monkeypatch.setattr(handler, 'query_llm', query)
    assert run_extraction(handler).success
    assert query.call_count == 2


def test_single_combined_event_is_valid_when_source_has_no_separate_time(handler, monkeypatch) -> None:
    combined_slot = dict(slot(8), separate=False, start_time='19:30')
    query = Mock(side_effect=[
        (json.dumps([combined_slot]), {}),
        (json.dumps([event(8, event_type='class, social dance', start_time='19:30')]), {}),
    ])
    monkeypatch.setattr(handler, 'query_llm', query)
    assert run_extraction(handler).events_written == 1
    assert query.call_count == 2


def test_normalization_cannot_silently_drop_recovered_social(handler, monkeypatch) -> None:
    monkeypatch.setattr(handler, 'query_llm', Mock(side_effect=[
        (json.dumps([slot(8), slot(15)]), {}), (json.dumps([event(8)]), {}),
        (json.dumps([event(15)]), {}),
    ]))
    monkeypatch.setattr(handler, '_normalize_event_rows_before_write', lambda **kwargs: pd.DataFrame([event(8)]))
    assert run_extraction(handler).decision_reason == 'newsletter_social_coverage_missing'
    handler.db_handler.write_events_to_db.assert_not_called()


def test_empty_primary_array_recovers_expected_slots(handler, monkeypatch) -> None:
    monkeypatch.setattr(handler, 'query_llm', Mock(side_effect=[
        (json.dumps([slot(8)]), {}), ('[]', {}), (json.dumps([event(8)]), {}),
    ]))
    assert run_extraction(handler).events_written == 1


def test_inventory_never_silently_truncates_source(handler, monkeypatch) -> None:
    handler.config['crawling']['prompt_max_length'] = 100
    query = Mock()
    monkeypatch.setattr(handler, 'query_llm', query)
    assert run_extraction(handler).decision_reason == 'newsletter_social_inventory_failed'
    query.assert_not_called()


def test_one_record_cannot_cover_two_expected_slots(handler) -> None:
    unknown_venue = dict(slot(8), location='')
    assert handler._missing_newsletter_socials(
        [unknown_venue, slot(8)], [event(8)]
    ) == [unknown_venue]


def test_generic_venue_does_not_cover_distinct_halls(handler) -> None:
    assert handler._missing_newsletter_socials(
        [slot(8)], [event(8, 'Hall')]
    ) == [slot(8)]


def test_inventory_transport_failure_is_reported(handler, monkeypatch) -> None:
    monkeypatch.setattr(handler, 'query_llm', Mock(return_value=None))
    assert run_extraction(handler).decision_reason == 'newsletter_social_inventory_failed'
    handler.db_handler.write_events_to_db.assert_not_called()
