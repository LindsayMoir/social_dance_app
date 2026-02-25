import sys

sys.path.insert(0, "tests/validation")

from chatbot_evaluator import ChatbotScorer


class _DummyLLM:
    pass


def test_default_social_dance_filter_not_penalized_for_tonight_query():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Show me lindy hop events tonight",
        "sql_query": (
            "SELECT * FROM events WHERE dance_style ILIKE '%lindy hop%' "
            "AND start_date = '2026-02-24' AND start_time >= '18:00:00' "
            "AND event_type ILIKE '%social dance%'"
        ),
    }
    eval_result = {
        "score": 82,
        "reasoning": "It may be too restrictive by forcing event_type to social dance.",
        "criteria_matched": ["dance_style", "timeframe"],
        "criteria_missed": ["event_type"],
        "sql_issues": ["event_type too restrictive"],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "default_event_type_policy" in out["criteria_matched"]
    assert "event_type" not in [c.lower() for c in out["criteria_missed"]]
    assert all("event_type" not in issue.lower() for issue in out["sql_issues"])


def test_rehearsal_query_not_penalized_for_missing_extra_dance_related_filter():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Show me dance rehearsals next month",
        "sql_query": (
            "SELECT * FROM events WHERE event_type ILIKE '%rehearsal%' "
            "AND start_date >= '2026-03-01' AND start_date <= '2026-03-31'"
        ),
    }
    eval_result = {
        "score": 86,
        "reasoning": "The query correctly filters rehearsals but doesn't explicitly ensure events are dance-related.",
        "criteria_matched": ["event_type", "timeframe"],
        "criteria_missed": [],
        "sql_issues": ["doesn't explicitly ensure events are dance-related"],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "rehearsal_event_type_policy" in out["criteria_matched"]
    assert all("dance-related" not in issue.lower() for issue in out["sql_issues"])


def test_workshop_query_not_penalized_for_class_workshop_or_filter():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Find workshops for west coast swing this week",
        "sql_query": (
            "SELECT * FROM events WHERE "
            "(dance_style ILIKE '%west coast swing%' OR dance_style ILIKE '%wcs%') "
            "AND start_date >= '2026-02-24' AND start_date <= '2026-02-28' "
            "AND (event_type ILIKE '%class%' OR event_type ILIKE '%workshop%')"
        ),
    }
    eval_result = {
        "score": 88,
        "reasoning": "It allows general 'class' records and is not strictly workshop-only.",
        "criteria_matched": ["dance_style", "timeframe"],
        "criteria_missed": [],
        "sql_issues": ["allows general class records"],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "workshop_learning_synonym_policy" in out["criteria_matched"]
    assert all("class" not in issue.lower() for issue in out["sql_issues"])
