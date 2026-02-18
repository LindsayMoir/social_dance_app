import os
import sys
import types
import importlib

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
sys.modules['db'] = mod

# Use src on path like other tests
sys.path.insert(0, 'src')

from utils.sql_filters import enforce_dance_style  # sanity import

# Import main app
app_main = importlib.import_module('main')

# Patch LLM to return base SQL missing style filter
BASE_SQL = (
    "SELECT event_name, event_type, dance_style, day_of_week, start_date, end_date, "
    "start_time, end_time, source, url, price, description, location "
    "FROM events WHERE start_date >= '2026-02-16' AND start_date <= '2026-02-22' "
    "ORDER BY start_date, start_time LIMIT 30"
)
app_main.llm_handler.query_llm = lambda *_a, **_k: BASE_SQL


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


def test_confirmation_yes_enforces_style_on_execution(monkeypatch):
    # Simulate a pending query without style filter
    pending = {
        'combined_query': 'Where can I dance west coast swing this week?',
        'user_input': 'Where can I dance west coast swing this week?',
        'sql_query': BASE_SQL,
        'intent': 'search'
    }
    # Monkeypatch conversation manager to return a fixed conversation and pending query
    monkeypatch.setattr(app_main.conversation_manager, 'create_or_get_conversation', lambda token: 'conv-1')
    monkeypatch.setattr(app_main.conversation_manager, 'get_pending_query', lambda cid: pending)
    monkeypatch.setattr(app_main.conversation_manager, 'clear_pending_query', lambda cid: None)
    # Ensure DB returns no data for execute_query
    monkeypatch.setattr(app_main.db_handler, 'execute_query', lambda *a, **k: [])
    monkeypatch.setattr(app_main.conversation_manager, 'add_message', lambda **kwargs: 'msg-1')
    monkeypatch.setattr(app_main.conversation_manager, 'get_recent_messages', lambda cid, limit=3: [])
    monkeypatch.setattr(app_main.conversation_manager, 'get_conversation_context', lambda cid: {})

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
    monkeypatch.setattr(app_main.conversation_manager, 'create_or_get_conversation', lambda token: 'conv-1')
    monkeypatch.setattr(app_main.conversation_manager, 'get_pending_query', lambda cid: pending)
    monkeypatch.setattr(app_main.conversation_manager, 'clear_pending_query', lambda cid: None)
    monkeypatch.setattr(app_main.db_handler, 'execute_query', lambda *a, **k: [])
    monkeypatch.setattr(app_main.conversation_manager, 'add_message', lambda **kwargs: 'msg-1')
    monkeypatch.setattr(app_main.conversation_manager, 'get_recent_messages', lambda cid, limit=3: [])
    monkeypatch.setattr(app_main.conversation_manager, 'get_conversation_context', lambda cid: {})

    req = app_main.ConfirmationRequest(confirmation='yes', session_token='tok-1', clarification="")
    resp = app_main.process_confirmation(req)
    sql = (resp.get('sql_query') or '').lower()
    assert "dance_style ilike" not in sql
