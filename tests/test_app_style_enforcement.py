import os
import sys
import types
import importlib
import logging

# Ensure env vars for LLM init don't crash import
os.environ.setdefault('OPENAI_API_KEY', 'test-key')
os.environ.setdefault('MISTRAL_API_KEY', 'test-key')

# Pre-stub the db module to avoid real DB connections during import
mod = types.ModuleType('db')
class DummyMeta:
    tables = {}
class DatabaseHandler:
    def __init__(self, config=None):
        self.metadata = DummyMeta()
    def set_llm_handler(self, h):
        self.llm = h
    def execute_query(self, *a, **k):
        return []
mod.DatabaseHandler = DatabaseHandler
mod.ensure_chatbot_metrics_schema = lambda *_a, **_k: None
sys.modules['db'] = mod

# Use src on path like other tests
sys.path.insert(0, 'src')

from utils.sql_filters import enforce_dance_style  # sanity import

# Import main app
app_main = importlib.import_module('main')

# Provide a fallback SQL generator for any test path that still reaches SQL generation.
BASE_SQL = (
    "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
    "start_time, end_time, source, url, price, description, location "
    "FROM events WHERE start_date >= '2026-02-16' AND start_date <= '2026-02-22' "
    "ORDER BY start_date, start_time LIMIT 30"
)
app_main._query_llm_timed = lambda *_a, **_k: BASE_SQL


class DummyConversationManager:
    def __init__(self, pending=None, context=None, intent='search'):
        self.pending = pending
        self.context = context or {}
        self.intent = intent
        self.stored_pending = None
        self.updated_context = None

    def create_or_get_conversation(self, _token):
        return 'conv-1'

    def get_pending_query(self, _cid):
        return self.pending

    def clear_pending_query(self, _cid):
        return None

    def add_message(self, **_kwargs):
        return 'msg-1'

    def get_recent_messages(self, _cid, limit=3):
        return []

    def get_conversation_context(self, _cid):
        return dict(self.context)

    def update_conversation_context(self, _cid, context_update):
        self.context.update(context_update)
        self.updated_context = context_update

    def classify_intent(self, _user_input, _context, _recent_messages):
        return self.intent

    def extract_entities(self, _user_input, _context):
        return {}

    def store_pending_query(self, **kwargs):
        self.stored_pending = kwargs


class DummyDBHandler:
    def execute_query(self, *_args, **_kwargs):
        return []


def test_process_query_adds_style_non_context_wcs():
    req = app_main.QueryRequest(user_input="Where can I dance west coast swing this week?")
    resp = app_main.process_query(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike '%west coast swing%'" in sql
    assert "dance_style ilike '%wcs%'" in sql


def test_process_query_no_style_non_context():
    req = app_main.QueryRequest(user_input="Where can I dance this week?")
    resp = app_main.process_query(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike" not in sql


def test_process_query_multiple_styles_non_context():
    req = app_main.QueryRequest(user_input="Show me salsa or bachata this week")
    resp = app_main.process_query(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike '%salsa%'" in sql
    assert "dance_style ilike '%bachata%'" in sql


def test_process_query_uses_deterministic_sql_for_supported_compound_query(monkeypatch):
    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("LLM SQL generation should not run for supported deterministic queries")

    monkeypatch.setattr(app_main, "_query_llm_timed", _raise_if_called)
    monkeypatch.setattr(
        app_main,
        "generate_interpretation",
        lambda *_args, **_kwargs: "Searching Tuesdays in Victoria for West Coast Swing.",
    )

    req = app_main.QueryRequest(user_input="Where can I dance on Tuesdays in Victoria for West Coast Swing?")
    resp = app_main.process_query(req)
    sql = (resp.get("sql_query") or "").lower()
    assert "extract(dow from start_date) in (1)" in sql
    assert "location ilike '%victoria%'" in sql
    assert "source ilike '%victoria%'" in sql
    assert "url ilike '%victoria%'" in sql
    assert "dance_style ilike '%west coast swing%'" in sql
    assert "dance_style ilike '%wcs%'" in sql
    assert "tuesdays" in (resp.get("interpretation") or "").lower()
    assert "victoria" in (resp.get("interpretation") or "").lower()


def test_process_query_returns_targeted_clarification_for_ambiguous_weekday(monkeypatch, caplog):
    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("LLM should not run when parser already knows clarification is required")

    monkeypatch.setattr(app_main, "_query_llm_timed", _raise_if_called)
    monkeypatch.setattr(app_main, "generate_interpretation", _raise_if_called)

    req = app_main.QueryRequest(user_input="Where can I dance on Tuesday in Victoria for salsa?")
    with caplog.at_level(logging.INFO):
        resp = app_main.process_query(req)
    assert resp.get("confirmation_required") is False
    assert resp.get("clarification_required") is True
    assert resp.get("message") == "Do you mean next Tuesday or every Tuesday?"
    assert "chatbot_trace_decision:" in caplog.text
    assert "decision=clarification_required" in caplog.text
    assert "reasons=weekday_scope" in caplog.text


def test_process_query_refinement_merges_constraints_structurally(monkeypatch, caplog):
    conversation_manager = DummyConversationManager(
        context={
            "last_search_query": "show me salsa events in Victoria this week",
            "last_search_constraints": {
                "include_styles": ["salsa"],
                "location_terms": ["victoria"],
                "city_terms": ["victoria"],
                "start_date": "2026-03-09",
                "end_date": "2026-03-14",
            },
            "concatenation_count": 1,
        },
        intent="refinement",
    )
    monkeypatch.setattr(app_main, "get_conversation_manager", lambda: conversation_manager)
    monkeypatch.setattr(app_main, "generate_interpretation", lambda *_args, **_kwargs: "unused")
    monkeypatch.setattr(app_main, "_query_llm_timed", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LLM should not run")))

    req = app_main.QueryRequest(user_input="also bachata", session_token="tok-1")
    with caplog.at_level(logging.INFO):
        resp = app_main.process_query(req)
    sql = (resp.get("sql_query") or "").lower()
    interpretation = (resp.get("interpretation") or "").lower()
    assert "dance_style ilike '%salsa%'" in sql
    assert "dance_style ilike '%bachata%'" in sql
    assert "victoria" in sql
    assert "salsa" in interpretation
    assert "bachata" in interpretation
    assert "decision=refinement_merge" in caplog.text
    assert "decision=parse_path mode=refinement_merge" in caplog.text
    assert "decision=sql_source source=deterministic_constraints" in caplog.text


def test_confirmation_yes_enforces_style_on_execution(monkeypatch):
    # Simulate a pending query without style filter
    pending = {
        'combined_query': 'Where can I dance west coast swing this week?',
        'user_input': 'Where can I dance west coast swing this week?',
        'sql_query': BASE_SQL,
        'intent': 'search'
    }
    # Monkeypatch conversation manager to return a fixed conversation and pending query
    monkeypatch.setattr(app_main, 'get_conversation_manager', lambda: DummyConversationManager(pending))
    monkeypatch.setattr(app_main, 'get_db_handler', lambda: DummyDBHandler())
    monkeypatch.setattr(app_main, '_validate_sql_select', lambda *_a, **_k: (True, ""))

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike '%west coast swing%'" in sql
    assert "dance_style ilike '%wcs%'" in sql


def test_confirmation_yes_no_style_no_filter_added(monkeypatch):
    pending = {
        'combined_query': 'Where can I dance this week?',
        'user_input': 'Where can I dance this week?',
        'sql_query': BASE_SQL,
        'intent': 'search'
    }
    monkeypatch.setattr(app_main, 'get_conversation_manager', lambda: DummyConversationManager(pending))
    monkeypatch.setattr(app_main, 'get_db_handler', lambda: DummyDBHandler())
    monkeypatch.setattr(app_main, '_validate_sql_select', lambda *_a, **_k: (True, ""))

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike" not in sql


def test_confirmation_yes_zero_results_returns_typed_response(monkeypatch):
    pending = {
        'combined_query': 'Where can I dance this week?',
        'user_input': 'Where can I dance this week?',
        'sql_query': BASE_SQL,
        'intent': 'search'
    }
    monkeypatch.setattr(app_main, 'get_conversation_manager', lambda: DummyConversationManager(pending))
    monkeypatch.setattr(app_main, 'get_db_handler', lambda: DummyDBHandler())
    monkeypatch.setattr(app_main, '_validate_sql_select', lambda *_a, **_k: (True, ""))

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    assert resp.get('confirmed') is True
    assert resp.get('failure_type') == 'zero_results'
    assert "found no matching events" in (resp.get('message') or '').lower()
    assert "any salsa this weekend?" in (resp.get('message') or '').lower()
    assert "only add location" in (resp.get('message') or '').lower()


def test_confirmation_yes_execution_failure_returns_typed_response(monkeypatch):
    pending = {
        'combined_query': 'Where can I dance this week?',
        'user_input': 'Where can I dance this week?',
        'sql_query': BASE_SQL,
        'intent': 'search'
    }

    class NoneDbHandler:
        def execute_query(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(app_main, 'get_conversation_manager', lambda: DummyConversationManager(pending))
    monkeypatch.setattr(app_main, 'get_db_handler', lambda: NoneDbHandler())
    monkeypatch.setattr(app_main, '_validate_sql_select', lambda *_a, **_k: (True, ""))

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    assert resp.get('confirmed') is False
    assert resp.get('failure_type') == 'execution_error'
    assert resp.get('retry_recommended') is True
    assert "salsa this weekend" in (resp.get('message') or '').lower()
    assert "live music tonight" in (resp.get('message') or '').lower()


def test_process_query_timeout_returns_typed_failure(monkeypatch):
    monkeypatch.setattr(
        app_main,
        "_query_llm_timed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(app_main.ChatbotLLMTimeoutError("timeout")),
    )
    monkeypatch.setattr(
        app_main,
        "build_sql_from_constraints",
        lambda *_args, **_kwargs: None,
    )

    req = app_main.QueryRequest(user_input="Recommend something surprising")
    resp = app_main.process_query(req)
    assert resp.get("confirmation_required") is False
    assert resp.get("failure_type") == "timeout"
    assert resp.get("retry_recommended") is True
    assert "where can i dance tomorrow night?" in (resp.get("message") or "").lower()
    assert "any live music tonight?" in (resp.get("message") or "").lower()


def test_failure_message_helpers_are_actionable() -> None:
    assert "any salsa this weekend?" in app_main._chatbot_unsupported_query_message().lower()
    assert "style or event type plus time" in app_main._chatbot_invalid_query_message().lower()


def test_enforce_default_event_type_skips_filter_for_all_event_types_request():
    sql_in = (
        "SELECT event_name FROM events "
        "WHERE start_date = '2026-02-26' AND (event_type ILIKE '%social dance%') "
        "ORDER BY start_date LIMIT 30"
    )
    out = app_main._enforce_default_event_type(sql_in, "show me all events today, all event types")
    assert "event_type ilike '%social dance%'" not in out.lower()


def test_confirmation_yes_all_event_types_removes_social_dance_filter(monkeypatch):
    pending = {
        'combined_query': 'show me all events today, all event types',
        'user_input': 'show me all events today, all event types',
        'sql_query': (
            "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
            "start_time, end_time, source, url, price, description, location "
            "FROM events WHERE start_date = '2026-02-26' AND (event_type ILIKE '%social dance%') "
            "ORDER BY start_date, start_time LIMIT 30"
        ),
        'intent': 'search'
    }
    monkeypatch.setattr(app_main, 'get_conversation_manager', lambda: DummyConversationManager(pending))
    monkeypatch.setattr(app_main, 'get_db_handler', lambda: DummyDBHandler())
    monkeypatch.setattr(app_main, '_validate_sql_select', lambda *_a, **_k: (True, ""))

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "event_type ilike '%social dance%'" not in sql


def test_enforce_default_event_type_only_live_music():
    sql_in = (
        "SELECT event_name FROM events "
        "WHERE start_date = '2026-02-26' "
        "ORDER BY start_date LIMIT 30"
    )
    out = app_main._enforce_default_event_type(sql_in, "show me only live music events today")
    low = out.lower()
    assert "event_type ilike '%live music%'" in low
    assert "event_type ilike '%social dance%'" not in low


def test_enforce_default_event_type_without_social_dance():
    sql_in = (
        "SELECT event_name FROM events "
        "WHERE start_date = '2026-02-26' "
        "ORDER BY start_date LIMIT 30"
    )
    out = app_main._enforce_default_event_type(sql_in, "events today without social dance")
    low = out.lower()
    assert "event_type ilike '%social dance%'" not in low
