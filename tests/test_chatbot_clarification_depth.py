import random
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tests/validation")

from chatbot_evaluator import ChatbotTestExecutor


class _DummyDB:
    def execute_query(self, _sql):
        return []


class _DummyLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def query_llm(self, *_args, **_kwargs):
        if not self._responses:
            return None
        return self._responses.pop(0)


def _build_executor(responses, chatbot_cfg):
    executor = ChatbotTestExecutor.__new__(ChatbotTestExecutor)
    executor.config = {
        "testing": {
            "validation": {
                "chatbot": chatbot_cfg,
            }
        }
    }
    executor.db_handler = _DummyDB()
    executor.llm_handler = _DummyLLM(responses)
    executor.sql_prompt_template = (
        "Conversation History:\n{conversation_history}\n"
        "Current Date: {current_date}\n"
        "Current Day: {current_day_of_week}\n"
        "Intent: {intent}\n"
    )
    executor.interpretation_prompt_template = "Interpret: {user_query}"
    executor.generate_interpretation = lambda **_kwargs: "test interpretation"
    return executor


def test_clarification_depth_target_two_builds_chain():
    random.seed(7)
    executor = _build_executor(
        responses=[
            "CLARIFICATION: Which date range should I use?",
            "CLARIFICATION: Which dance style should I prioritize?",
            "SELECT event_name FROM events WHERE 1=1",
        ],
        chatbot_cfg={
            "clarification_min_depth": 2,
            "clarification_max_depth": 2,
            "max_clarification_turns": 4,
        },
    )
    result = executor.execute_test_question(
        {
            "question": "Where can I dance?",
            "category": "natural_language",
            "expected_criteria": {},
        }
    )

    assert result["is_clarification"] is True
    assert result["clarification_depth_target"] == 2
    assert result["clarification_depth_achieved"] == 2
    chain = result.get("clarification_chain", [])
    assert len(chain) == 2
    assert "synthetic_user_reply" in chain[0]


def test_clarification_depth_range_randomized_and_sql_path_keeps_metadata():
    random.seed(3)
    executor = _build_executor(
        responses=[
            "CLARIFICATION: Please confirm timeframe.",
            "SELECT event_name FROM events WHERE start_date >= CURRENT_DATE",
        ],
        chatbot_cfg={
            "clarification_min_depth": 2,
            "clarification_max_depth": 2,
            "max_clarification_turns": 4,
        },
    )
    result = executor.execute_test_question(
        {
            "question": "Show me social dances",
            "category": "event_type_only",
            "expected_criteria": {},
        }
    )

    assert result.get("is_clarification", False) is False
    assert int(result["clarification_depth_target"]) == 2
    assert int(result["clarification_depth_achieved"]) == 1
    assert "clarification_chain" in result
