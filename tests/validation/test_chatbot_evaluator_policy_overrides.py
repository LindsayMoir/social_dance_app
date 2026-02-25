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
