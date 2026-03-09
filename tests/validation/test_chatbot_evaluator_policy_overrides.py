import sys

sys.path.insert(0, "tests/validation")

from chatbot_evaluator import ChatbotScorer, generate_chatbot_report


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


def test_weekday_on_monday_single_date_not_penalized():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Dance events on Monday",
        "sql_query": (
            "SELECT * FROM events WHERE start_date = '2026-03-02' "
            "AND (event_type ILIKE '%social dance%' OR event_type ILIKE '%live music%')"
        ),
    }
    eval_result = {
        "score": 84,
        "reasoning": (
            "The query incorrectly restricts to a single start_date instead of "
            "filtering for events occurring on Mondays."
        ),
        "criteria_matched": ["event_type"],
        "criteria_missed": ["day_of_week"],
        "sql_issues": ["should filter all Mondays, not single date"],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "weekday_single_date_policy" in out["criteria_matched"]
    assert all("monday" not in issue.lower() for issue in out["sql_issues"])


def test_next_week_sunday_to_saturday_window_not_penalized():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Show me lindy hop events next week",
        "sql_query": (
            "SELECT * FROM events WHERE (dance_style ILIKE '%lindy hop%' OR dance_style ILIKE '%lindy%') "
            "AND start_date >= '2026-03-01' AND start_date <= '2026-03-07' "
            "AND event_type ILIKE '%social dance%'"
        ),
        "current_date_used": "2026-02-24",
    }
    eval_result = {
        "score": 85,
        "reasoning": (
            'The date range does not match the definition of "next week" '
            "(it starts on 2026-03-01 instead of 2026-02-27)."
        ),
        "criteria_matched": ["dance_style", "event_type"],
        "criteria_missed": ["timeframe"],
        "sql_issues": ['Date range does not match definition of "next week"'],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "next_week_window_policy" in out["criteria_matched"]
    assert "timeframe" in out["criteria_matched"]
    assert "timeframe" not in [c.lower() for c in out["criteria_missed"]]
    assert all("next week" not in issue.lower() for issue in out["sql_issues"])


def test_weekend_evening_filter_not_penalized_when_friday_to_sunday_window_is_correct():
    scorer = ChatbotScorer(_DummyLLM())

    test_result = {
        "question": "Show me tango events this weekend",
        "sql_query": (
            "SELECT * FROM events WHERE dance_style ILIKE '%tango%' "
            "AND start_date >= '2026-03-06' AND start_date <= '2026-03-08' "
            "AND start_time >= '18:00:00' "
            "AND event_type ILIKE '%social dance%'"
        ),
    }
    eval_result = {
        "score": 84,
        "reasoning": (
            "Weekend date range is correct but incorrectly adds a start_time filter "
            "that restricts to evening events."
        ),
        "criteria_matched": ["dance_style", "timeframe"],
        "criteria_missed": ["timeframe"],
        "sql_issues": ["Incorrect evening time filter in weekend query"],
    }

    out = scorer._apply_policy_overrides(test_result, eval_result)

    assert out["score"] == 100
    assert "weekend_evening_filter_policy" in out["criteria_matched"]
    assert all("time" not in c.lower() for c in out["criteria_missed"])
    assert all("evening" not in issue.lower() for issue in out["sql_issues"])


def test_problem_category_does_not_label_weekend_without_weekend_definition_error(tmp_path):
    scored_results = [
        {
            "question": "Show me tango events this weekend",
            "category": "static",
            "execution_success": True,
            "sql_query": (
                "SELECT * FROM events WHERE dance_style ILIKE '%tango%' "
                "AND start_date >= '2026-03-06' AND start_date <= '2026-03-08' "
                "AND start_time >= '18:00:00'"
            ),
            "results_count": 5,
            "evaluation": {
                "score": 80,
                "reasoning": "Weekend range is correct; query adds an evening start_time filter.",
                "criteria_matched": ["dance_style", "timeframe"],
                "criteria_missed": ["timeframe"],
                "sql_issues": ["Evening start_time filter"],
                "interpretation_evaluation": {"score": 100},
            },
        }
    ]

    report = generate_chatbot_report(scored_results, output_dir=str(tmp_path))
    categories = report.get("problem_categories", [])
    assert categories
    assert categories[0]["name"] != "Weekend Calculation"


def test_problem_category_does_not_label_event_type_defaults_for_valid_class_filter(tmp_path):
    scored_results = [
        {
            "question": "Show me tango classes this weekend",
            "category": "static",
            "execution_success": True,
            "sql_query": (
                "SELECT * FROM events WHERE dance_style ILIKE '%tango%' "
                "AND start_date >= '2026-03-06' AND start_date <= '2026-03-08' "
                "AND (event_type ILIKE '%class%' OR event_type ILIKE '%workshop%')"
            ),
            "results_count": 5,
            "evaluation": {
                "score": 80,
                "reasoning": "Query is mostly correct and uses class/workshop event_type filters.",
                "criteria_matched": ["dance_style", "timeframe", "event_type"],
                "criteria_missed": [],
                "sql_issues": [],
                "interpretation_evaluation": {"score": 100},
            },
        }
    ]

    report = generate_chatbot_report(scored_results, output_dir=str(tmp_path))
    categories = report.get("problem_categories", [])
    assert categories
    assert categories[0]["name"] != "Event Type Defaults"


def test_problem_category_regression_cases_are_emitted(tmp_path):
    scored_results = [
        {
            "question": "Show me lindy hop events tonight",
            "category": "static",
            "execution_success": True,
            "sql_query": (
                "SELECT * FROM events WHERE dance_style ILIKE '%lindy hop%' "
                "AND start_date = '2026-03-09' AND start_time >= '18:00:00'"
            ),
            "results_count": 2,
            "evaluation": {
                "score": 80,
                "reasoning": "Tonight query has a restrictive start_time filter.",
                "criteria_matched": ["dance_style", "timeframe"],
                "criteria_missed": [],
                "sql_issues": ["time filter too restrictive"],
                "interpretation_evaluation": {"score": 100},
            },
        }
    ]

    report = generate_chatbot_report(scored_results, output_dir=str(tmp_path))
    cases = report.get("problem_category_regression_cases", [])
    assert cases
    assert cases[0]["regression_id"] == "CAT-001"
    assert cases[0]["category"]
    assert cases[0]["question"]
