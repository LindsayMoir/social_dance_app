# main.py
"""
This module sets up a FastAPI application for a Social Dance Chatbot API.
It handles:
- Loading environment variables and configuration.
- Initializing the LLMHandler.
- Constructing and sanitizing SQL queries from user input.
- Executing SQL queries on the database.
- Returning results via an API endpoint.
"""

import os
import sys
import logging
import yaml
import uuid
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta
from time import perf_counter
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import re
from typing import Tuple

# Set up sys.path so that modules in src/ are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
sys.path.append(parent_dir)
from utils.sql_filters import enforce_dance_style, detect_styles_in_text

print("Updated sys.path:", sys.path)
print("Current working directory:", os.getcwd())

from llm import LLMHandler  # Import the LLMHandler module
from db import DatabaseHandler  # Import DatabaseHandler for conversation management
from conversation_manager import ConversationManager  # Import ConversationManager
from query_constraints import (
    build_sql_from_constraints,
    constraints_to_query_text,
    derive_constraints_from_text,
)

# Load environment variables
load_dotenv()

# Setup centralized logging
from logging_config import setup_logging
setup_logging('main')
logging.info("main.py starting...")

# Calculate the base directory and config path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')

# Load YAML configuration
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    raise e

logging.info("main.py: Configuration loaded.")

# Initialize the LLMHandler and DatabaseHandler
llm_handler = LLMHandler(config_path=config_path)
db_handler = DatabaseHandler(config)
conversation_manager = ConversationManager(db_handler)

# Get the DATABASE_URL from environment variables
if os.getenv("RENDER"):
    DATABASE_URL = os.getenv("RENDER_EXTERNAL_DB_URL")
    print("Running on Render...")
else:
    DATABASE_URL = os.getenv("DATABASE_CONNECTION_STRING")
    print("Running locally...")
    
if DATABASE_URL:
    logging.info(f"DATABASE_URL / database connection string is set")
else:
    raise ValueError("DATABASE_URL / database connections string is not set.")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
logging.info("main.py: SQLAlchemy engine created.")

SLOW_CHATBOT_RESPONSE_SECONDS = 20.0
CHATBOT_HEDGE_OPENAI_DELAY_SECONDS = 15.0
CHATBOT_HEDGE_OPENROUTER_DELAY_SECONDS = 15.0
CHATBOT_HEDGE_TOTAL_TIMEOUT_SECONDS = 45.0
CHATBOT_WAIT_NOTICE = "Most requests complete within about 15 seconds, but traffic spikes can take longer."


class ChatbotLLMTimeoutError(RuntimeError):
    """Raised when the chatbot hedged provider chain exceeds max wait time."""


def _chatbot_traffic_timeout_message() -> str:
    """User-facing fallback when all chatbot provider windows time out."""
    return (
        "I'm experiencing heavy traffic right now and couldn't complete your request within 45 seconds. "
        "Please try again in a moment. "
        f"{CHATBOT_WAIT_NOTICE}"
    )


def _ensure_chatbot_metrics_tables() -> None:
    """Create chatbot performance metric tables if they do not exist."""
    create_requests = """
    CREATE TABLE IF NOT EXISTS chatbot_request_metrics (
        id SERIAL PRIMARY KEY,
        request_id TEXT UNIQUE NOT NULL,
        endpoint TEXT NOT NULL,
        session_suffix TEXT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        duration_ms DOUBLE PRECISION,
        result_type TEXT,
        user_input TEXT,
        sql_snippet TEXT,
        has_response BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_stages = """
    CREATE TABLE IF NOT EXISTS chatbot_stage_metrics (
        id SERIAL PRIMARY KEY,
        request_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        stage TEXT NOT NULL,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        duration_ms DOUBLE PRECISION,
        metadata_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    create_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_chatbot_request_metrics_started_at ON chatbot_request_metrics(started_at);",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_request_metrics_endpoint ON chatbot_request_metrics(endpoint);",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_started_at ON chatbot_stage_metrics(started_at);",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_stage ON chatbot_stage_metrics(stage);",
        "CREATE INDEX IF NOT EXISTS idx_chatbot_stage_metrics_request_id ON chatbot_stage_metrics(request_id);",
    ]
    try:
        with engine.begin() as conn:
            conn.execute(text(create_requests))
            conn.execute(text(create_stages))
            for idx_sql in create_indexes:
                conn.execute(text(idx_sql))
        logging.info("chatbot_metrics_db: ensured chatbot performance metric tables exist")
    except Exception as e:
        logging.warning("chatbot_metrics_db: failed to ensure metric tables: %s", e)


def _persist_request_start(request_id: str, endpoint: str, session_suffix: str | None) -> None:
    """Persist request start marker for duration tracking across process restarts."""
    sql = """
    INSERT INTO chatbot_request_metrics (request_id, endpoint, session_suffix, started_at, updated_at)
    VALUES (:request_id, :endpoint, :session_suffix, :started_at, :updated_at)
    ON CONFLICT (request_id)
    DO UPDATE SET
      endpoint = EXCLUDED.endpoint,
      session_suffix = COALESCE(EXCLUDED.session_suffix, chatbot_request_metrics.session_suffix),
      started_at = COALESCE(chatbot_request_metrics.started_at, EXCLUDED.started_at),
      updated_at = EXCLUDED.updated_at;
    """
    now_ts = datetime.now()
    params = {
        "request_id": request_id,
        "endpoint": endpoint,
        "session_suffix": session_suffix,
        "started_at": now_ts,
        "updated_at": now_ts,
    }
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:
        logging.warning("chatbot_metrics_db: failed to persist request start (%s): %s", request_id, e)


def _persist_request_end(request_id: str, endpoint: str, duration_ms: float, result_type: str | None) -> None:
    """Persist request completion metrics."""
    sql = """
    INSERT INTO chatbot_request_metrics (request_id, endpoint, finished_at, duration_ms, result_type, updated_at)
    VALUES (:request_id, :endpoint, :finished_at, :duration_ms, :result_type, :updated_at)
    ON CONFLICT (request_id)
    DO UPDATE SET
      endpoint = EXCLUDED.endpoint,
      finished_at = EXCLUDED.finished_at,
      duration_ms = EXCLUDED.duration_ms,
      result_type = EXCLUDED.result_type,
      updated_at = EXCLUDED.updated_at;
    """
    now_ts = datetime.now()
    params = {
        "request_id": request_id,
        "endpoint": endpoint,
        "finished_at": now_ts,
        "duration_ms": float(duration_ms),
        "result_type": result_type,
        "updated_at": now_ts,
    }
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:
        logging.warning("chatbot_metrics_db: failed to persist request end (%s): %s", request_id, e)


def _persist_stage_metric(
    request_id: str,
    endpoint: str,
    stage: str,
    duration_ms: float,
    fields: dict,
) -> None:
    """Persist stage-level timing events for hotspot analysis."""
    finished_at = datetime.now()
    started_at = finished_at - timedelta(milliseconds=float(duration_ms))
    sql = """
    INSERT INTO chatbot_stage_metrics (request_id, endpoint, stage, started_at, finished_at, duration_ms, metadata_json)
    VALUES (:request_id, :endpoint, :stage, :started_at, :finished_at, :duration_ms, :metadata_json);
    """
    params = {
        "request_id": request_id,
        "endpoint": endpoint,
        "stage": stage,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": float(duration_ms),
        "metadata_json": json.dumps(fields or {}),
    }
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), params)
    except Exception as e:
        logging.warning("chatbot_metrics_db: failed to persist stage metric (%s/%s): %s", request_id, stage, e)


def _persist_request_trace(request_id: str, user_input: str | None = None, sql_snippet: str | None = None) -> None:
    """Persist question/sql trace to correlate slow requests with generated SQL."""
    updates: list[str] = []
    params: dict = {"request_id": request_id, "updated_at": datetime.now()}
    if user_input is not None:
        updates.append("user_input = :user_input")
        params["user_input"] = user_input
    if sql_snippet is not None:
        updates.append("sql_snippet = :sql_snippet")
        params["sql_snippet"] = sql_snippet
    if not updates:
        return
    sql = (
        "UPDATE chatbot_request_metrics SET "
        + ", ".join(updates)
        + ", updated_at = :updated_at WHERE request_id = :request_id"
    )
    try:
        with engine.begin() as conn:
            result = conn.execute(text(sql), params)
            if int(getattr(result, "rowcount", 0) or 0) <= 0:
                insert_sql = """
                INSERT INTO chatbot_request_metrics (
                    request_id, endpoint, user_input, sql_snippet, started_at, updated_at
                ) VALUES (
                    :request_id, :endpoint, :user_input, :sql_snippet, :started_at, :updated_at
                ) ON CONFLICT (request_id) DO NOTHING;
                """
                conn.execute(
                    text(insert_sql),
                    {
                        "request_id": request_id,
                        "endpoint": "/query",
                        "user_input": params.get("user_input"),
                        "sql_snippet": params.get("sql_snippet"),
                        "started_at": datetime.now(),
                        "updated_at": datetime.now(),
                    },
                )
    except Exception as e:
        logging.warning("chatbot_metrics_db: failed to persist trace (%s): %s", request_id, e)


def _new_request_id(prefix: str) -> str:
    """Generate a short request id for correlating timing logs."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _session_suffix(session_token: str | None) -> str:
    """Return non-sensitive session token suffix for log correlation."""
    if not session_token:
        return "none"
    return str(session_token)[-8:]


def _log_timing_start(request_id: str, endpoint: str, stage: str, **fields) -> float:
    """Log timing start and return a monotonic timer start marker."""
    started = perf_counter()
    if fields:
        detail = " ".join(f"{k}={v}" for k, v in fields.items())
        logging.info(
            "chatbot_timing_start: request_id=%s endpoint=%s stage=%s %s",
            request_id,
            endpoint,
            stage,
            detail,
        )
    else:
        logging.info(
            "chatbot_timing_start: request_id=%s endpoint=%s stage=%s",
            request_id,
            endpoint,
            stage,
        )
    if stage == "request_total":
        _persist_request_start(request_id, endpoint, fields.get("session_suffix"))
    return started


def _log_timing_end(request_id: str, endpoint: str, stage: str, started: float, **fields) -> None:
    """Log timing end with elapsed milliseconds and optional detail fields."""
    elapsed_ms = (perf_counter() - started) * 1000.0
    level = logging.WARNING if (elapsed_ms / 1000.0) >= SLOW_CHATBOT_RESPONSE_SECONDS else logging.INFO
    if fields:
        detail = " ".join(f"{k}={v}" for k, v in fields.items())
        logging.log(
            level,
            "chatbot_timing_end: request_id=%s endpoint=%s stage=%s duration_ms=%.1f %s",
            request_id,
            endpoint,
            stage,
            elapsed_ms,
            detail,
        )
    else:
        logging.log(
            level,
            "chatbot_timing_end: request_id=%s endpoint=%s stage=%s duration_ms=%.1f",
            request_id,
            endpoint,
            stage,
            elapsed_ms,
        )
    _persist_stage_metric(request_id, endpoint, stage, elapsed_ms, fields)
    if stage == "request_total":
        _persist_request_end(request_id, endpoint, elapsed_ms, str(fields.get("result_type", "") or ""))


def _query_llm_timed(
    request_id: str,
    endpoint: str,
    stage: str,
    request_url: str,
    prompt: str,
    tools=None,
):
    """Run llm_handler.query_llm with start/end timing logs."""
    started = _log_timing_start(
        request_id,
        endpoint,
        stage,
        request_url=request_url,
        prompt_len=len(prompt or ""),
    )
    try:
        if str(request_url or "").lower() == "chatbot":
            response, winner = _query_llm_chatbot_hedged(
                request_id=request_id,
                endpoint=endpoint,
                stage=stage,
                request_url=request_url,
                prompt=prompt,
                tools=tools,
            )
        else:
            response = llm_handler.query_llm(request_url, prompt, tools=tools)
            winner = "standard"
        _log_timing_end(
            request_id,
            endpoint,
            stage,
            started,
            has_response=bool(response),
            response_len=len(response) if isinstance(response, str) else 0,
            llm_winner=winner,
        )
        return response
    except ChatbotLLMTimeoutError:
        _log_timing_end(request_id, endpoint, stage, started, error="timeout", llm_winner="none")
        raise
    except Exception:
        _log_timing_end(request_id, endpoint, stage, started, error="exception")
        raise


def _query_llm_chatbot_hedged(
    request_id: str,
    endpoint: str,
    stage: str,
    request_url: str,
    prompt: str,
    tools=None,
) -> Tuple[str | None, str]:
    """
    Staged hedged chatbot strategy:
    - T+0s: OpenAI
    - T+15s: OpenRouter (DeepSeek)
    - T+30s: Gemini
    - T+45s: fail request
    Returns first non-empty response and winning provider.
    """
    plan = [
        ("openai", 0.0),
        ("openrouter", CHATBOT_HEDGE_OPENAI_DELAY_SECONDS),
        (
            "gemini",
            CHATBOT_HEDGE_OPENAI_DELAY_SECONDS + CHATBOT_HEDGE_OPENROUTER_DELAY_SECONDS,
        ),
    ]
    hard_deadline_at = perf_counter() + CHATBOT_HEDGE_TOTAL_TIMEOUT_SECONDS
    pending: dict = {}
    launched: set[str] = set()

    def _run_provider_lane(provider: str) -> str | None:
        lane_stage = f"{stage}_hedge_{provider}"
        lane_started = _log_timing_start(
            request_id,
            endpoint,
            lane_stage,
            hedged=True,
            provider=provider,
            disable_fallback=True,
        )
        try:
            response = llm_handler.query_llm(
                request_url,
                prompt,
                tools=tools,
                provider_override=provider,
                disable_fallback=True,
            )
            _log_timing_end(
                request_id,
                endpoint,
                lane_stage,
                lane_started,
                has_response=bool(response),
                response_len=len(response) if isinstance(response, str) else 0,
            )
            return response
        except Exception:
            _log_timing_end(request_id, endpoint, lane_stage, lane_started, error="exception")
            return None

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="chatbot-hedge") as executor:
        while perf_counter() < hard_deadline_at:
            elapsed = CHATBOT_HEDGE_TOTAL_TIMEOUT_SECONDS - (hard_deadline_at - perf_counter())
            for provider, start_at in plan:
                if provider in launched:
                    continue
                if elapsed >= start_at:
                    launched.add(provider)
                    pending[executor.submit(_run_provider_lane, provider)] = provider

            if not pending:
                continue

            next_start_offsets = [start_at for _, start_at in plan if start_at > elapsed]
            time_to_next_start = (
                min(next_start_offsets) - elapsed if next_start_offsets else None
            )
            remaining_total = max(0.0, hard_deadline_at - perf_counter())
            wait_timeout = remaining_total
            if time_to_next_start is not None:
                wait_timeout = max(0.0, min(wait_timeout, time_to_next_start))

            done, _ = wait(
                set(pending.keys()),
                timeout=wait_timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            for fut in done:
                provider = pending.pop(fut, "unknown")
                try:
                    response = fut.result()
                except Exception:
                    response = None
                if response:
                    for loser in list(pending.keys()):
                        loser.cancel()
                    return response, provider

    raise ChatbotLLMTimeoutError("chatbot_hedged_timeout_exceeded_45s")


def _execute_query_timed(request_id: str, endpoint: str, stage: str, sql_query: str):
    """Run db_handler.execute_query with timing logs."""
    started = _log_timing_start(
        request_id,
        endpoint,
        stage,
        sql_len=len(sql_query or ""),
    )
    rows = db_handler.execute_query(sql_query)
    row_count = 0 if not rows else len(rows)
    _log_timing_end(request_id, endpoint, stage, started, rows=row_count)
    return rows


def _safe_log_snippet(value: str, max_len: int = 220) -> str:
    """Return a single-line safe snippet for structured logs."""
    text = (value or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


_ensure_chatbot_metrics_tables()

# SQL preflight validation for date arithmetic and common pitfalls
def _sql_has_illegal_date_arithmetic(sql: str) -> bool:
    if not sql:
        return False
    s = sql.upper()
    patterns = [
        r"\bCURRENT_DATE\s*[\+\-]\s*\d+\b",
        r"\bCURRENT_TIMESTAMP\s*[\+\-]\s*\d+\b",
        r"'\d{4}-\d{2}-\d{2}'\s*[\+\-]",  # adding/subtracting to a string literal date
    ]
    return any(re.search(p, s) for p in patterns)


def _repair_common_sql_issues(sql: str) -> str:
    """
    Apply safe, minimal repairs for common malformed SQL generated by the LLM.

    Current repairs:
    - Collapse duplicate top-level WHERE clauses into AND for simple flat queries.
    """
    if not sql:
        return sql
    s = re.sub(r"\s+", " ", sql.strip())
    where_matches = list(re.finditer(r"(?i)\bWHERE\b", s))
    if len(where_matches) <= 1:
        return s

    # Only repair simple flat queries (avoid rewriting subqueries/CTEs).
    first_where_end = where_matches[0].end()
    tail = s[first_where_end:]
    if re.search(r"(?i)\(\s*SELECT\b", tail):
        return s

    repaired_tail = re.sub(r"(?i)\bWHERE\b", "AND", tail)
    repaired = s[:first_where_end] + repaired_tail
    return repaired


def _validate_sql_select(sql: str) -> Tuple[bool, str]:
    """
    Validate that SQL is executable and syntactically valid by running EXPLAIN.
    """
    if not sql or not sql.strip().upper().startswith("SELECT"):
        return False, "Query is not a SELECT statement."
    candidate = sql.strip().rstrip(";")
    try:
        with engine.connect() as conn:
            conn.execute(text(f"EXPLAIN {candidate}"))
        return True, ""
    except Exception as e:
        return False, str(e)


def _format_sql_for_display(sql: str) -> str:
    """
    Lightweight SQL formatter for display in UI.

    - Puts each SELECT column on its own line
    - Breaks before FROM, WHERE, GROUP BY, ORDER BY, LIMIT, HAVING
    - Indents AND/OR conditions
    """
    if not sql:
        return sql
    s = re.sub(r"\s+", " ", sql.strip())

    # Extract columns between SELECT and FROM
    m = re.search(r"(?is)\bselect\s+(.*?)\s+from\s+", s)
    if m:
        cols_raw = m.group(1)
        cols = [c.strip() for c in cols_raw.split(',')]
        select_block = "SELECT\n    " + ",\n    ".join(cols)
        rest = s[m.end():]  # starts after 'from '
        formatted = select_block + "\nFROM " + rest
    else:
        formatted = s

    # Section breaks
    formatted = re.sub(r"(?i)\s+where\s+", "\nWHERE\n    ", formatted)
    formatted = re.sub(r"(?i)\s+group\s+by\s+", "\nGROUP BY\n    ", formatted)
    formatted = re.sub(r"(?i)\s+order\s+by\s+", "\nORDER BY\n    ", formatted)
    formatted = re.sub(r"(?i)\s+having\s+", "\nHAVING\n    ", formatted)
    formatted = re.sub(r"(?i)\s+limit\s+", "\nLIMIT ", formatted)

    # Condition line breaks
    formatted = formatted.replace(" AND ", "\n    AND ")
    formatted = formatted.replace(" OR ", "\n    OR ")

    return formatted.strip()


def _is_all_event_types_request(user_text: str) -> bool:
    """
    Return True when user explicitly requests all event types (no event_type filtering).
    """
    ut = (user_text or "").lower()
    patterns = [
        "all event types",
        "all events",
        "any event type",
        "any events",
        "include all event types",
        "include all events",
        "not just social dance",
        "not only social dance",
        "not just social dances",
        "not only social dances",
    ]
    return any(p in ut for p in patterns)


def _canonical_event_types_from_text(user_text: str) -> set[str]:
    """
    Detect canonical event types referenced in user text.
    """
    ut = (user_text or "").lower()
    detected: set[str] = set()

    patterns = {
        "social dance": [r"\bsocial dance(s|ing)?\b"],
        "class": [r"\bclass(es)?\b", r"\blesson(s)?\b"],
        "workshop": [r"\bworkshop(s)?\b"],
        "live music": [r"\blive music\b"],
        "rehearsal": [r"\brehearsal(s)?\b"],
        "other": [r"\bother\b"],
    }
    for canonical, pats in patterns.items():
        if any(re.search(p, ut, flags=re.IGNORECASE) for p in pats):
            detected.add(canonical)
    return detected


def _detected_only_event_types(user_text: str) -> set[str]:
    """
    Detect event types explicitly constrained with "only"/"just"/"<type> only".
    """
    ut = (user_text or "").lower()
    aliases = {
        "social dance": [r"social dance(?:s|ing)?"],
        "class": [r"class(?:es)?", r"lesson(?:s)?"],
        "workshop": [r"workshop(?:s)?"],
        "live music": [r"live music"],
        "rehearsal": [r"rehearsal(?:s)?"],
        "other": [r"other"],
    }
    out: set[str] = set()
    for canonical, pats in aliases.items():
        for pat in pats:
            if (
                re.search(rf"\b(?:only|just)\s+{pat}\b", ut, flags=re.IGNORECASE)
                or re.search(rf"\b{pat}\s+only\b", ut, flags=re.IGNORECASE)
            ):
                out.add(canonical)
    return out


def _detected_excluded_event_types(user_text: str) -> set[str]:
    """
    Detect event types explicitly excluded from results.
    """
    ut = (user_text or "").lower()
    aliases = {
        "social dance": [r"social dance(?:s|ing)?"],
        "class": [r"class(?:es)?", r"lesson(?:s)?"],
        "workshop": [r"workshop(?:s)?"],
        "live music": [r"live music"],
        "rehearsal": [r"rehearsal(?:s)?"],
        "other": [r"other"],
    }
    out: set[str] = set()
    for canonical, pats in aliases.items():
        for pat in pats:
            if (
                re.search(rf"\b(?:no|without|exclude|excluding)\s+{pat}\b", ut, flags=re.IGNORECASE)
                or re.search(rf"\bexcept\s+{pat}\b", ut, flags=re.IGNORECASE)
            ):
                out.add(canonical)
    return out


def _build_event_type_clause(selected_types: set[str]) -> str:
    """
    Build SQL event_type clause from canonical event types.
    """
    order = ["social dance", "class", "workshop", "live music", "rehearsal", "other"]
    selected = [t for t in order if t in selected_types]
    if not selected:
        return ""

    parts = [f"event_type ILIKE '%{t}%'" for t in selected]
    return "( " + " OR ".join(parts) + " )"


def _derive_event_type_policy_clause(user_text: str) -> str:
    """
    Derive the desired event_type SQL clause from user intent.

    Returns:
        str: SQL clause, or empty string for "no event_type filtering".
    """
    if _is_all_event_types_request(user_text):
        return ""

    explicit_types = _canonical_event_types_from_text(user_text)
    only_types = _detected_only_event_types(user_text)
    excluded_types = _detected_excluded_event_types(user_text)

    # Strongest signal: explicit "only" constraints.
    if only_types:
        selected = set(only_types) - excluded_types
        return _build_event_type_clause(selected)

    # If user names types, include those; preserve legacy default social dance unless explicitly excluded.
    if explicit_types:
        selected = set(explicit_types)
        if "social dance" not in selected and "social dance" not in excluded_types:
            selected.add("social dance")
        selected -= excluded_types
        return _build_event_type_clause(selected)

    # No explicit type mention: legacy default social dance unless excluded.
    default_selected = {"social dance"} - excluded_types
    return _build_event_type_clause(default_selected)


def _apply_event_type_clause(sql: str, clause: str) -> str:
    """
    Apply event_type clause to SQL after removing existing event_type predicates.
    """
    if not sql:
        return sql
    s = _remove_event_type_filters(sql)
    if not clause:
        return s

    # Append before ORDER BY / GROUP BY / HAVING / LIMIT if present.
    keywords = [r"\bORDER\s+BY\b", r"\bGROUP\s+BY\b", r"\bHAVING\b", r"\bLIMIT\b"]
    insert_pt = len(s)
    for kw in keywords:
        m = re.search(kw, s, flags=re.IGNORECASE)
        if m:
            insert_pt = min(insert_pt, m.start())

    prefix = s[:insert_pt].rstrip()
    suffix = s[insert_pt:]
    if re.search(r"\bWHERE\b", prefix, flags=re.IGNORECASE):
        prefix += f" AND {clause}"
    else:
        prefix += f" WHERE {clause}"
    return (prefix + " " + suffix.lstrip()).strip()


def _remove_event_type_filters(sql: str) -> str:
    """
    Remove common event_type WHERE predicates from generated SQL.

    This is intentionally conservative and targets the patterns generated by this app.
    """
    if not sql:
        return sql
    s = sql

    # Remove grouped or single event_type predicates preceded by AND/OR.
    s = re.sub(
        r"(?is)\s+(AND|OR)\s+\(\s*event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'\s*(?:OR\s+event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'\s*)*\)",
        "",
        s,
    )
    s = re.sub(
        r"(?is)\s+(AND|OR)\s+event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'",
        "",
        s,
    )

    # Remove WHERE-leading event_type predicates.
    s = re.sub(
        r"(?is)\bWHERE\s+\(\s*event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'\s*(?:OR\s+event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'\s*)*\)\s*(AND\s+)?",
        lambda m: "WHERE " if m.group(1) else "",
        s,
    )
    s = re.sub(
        r"(?is)\bWHERE\s+event_type\s*(?:ILIKE|LIKE|=)\s*'[^']+'\s*(AND\s+)?",
        lambda m: "WHERE " if m.group(1) else "",
        s,
    )

    # Clean up residual WHERE/AND artifacts and whitespace.
    s = re.sub(r"(?is)\bWHERE\s+(ORDER\s+BY|GROUP\s+BY|LIMIT|HAVING)\b", r"\1", s)
    s = re.sub(r"(?is)\bWHERE\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _enforce_default_event_type(sql: str, user_text: str) -> str:
    """
    Ensure SQL includes social dance by default unless the user explicitly restricts event_type.

    Rules:
    - If user_text indicates "only" classes or "only" live music, do not add social dance.
    - If SQL already contains social dance, leave as-is.
    - If SQL contains an event_type condition but not social dance, wrap the first event_type condition as
      (event_type ILIKE '%social dance%' OR <original_event_type_condition>).
    - If SQL has no event_type condition, append AND (event_type ILIKE '%social dance%') before ORDER BY/LIMIT.
    """
    if not sql:
        return sql
    clause = _derive_event_type_policy_clause(user_text)
    return _apply_event_type_clause(sql, clause)


def _force_style_in_interpretation(text: str, user_text: str) -> str:
    """
    Ensure the natural language interpretation reflects any explicit dance style from user_text.

    - If a style is detected and not already present in the text, replace the first occurrence of
      "dance events" or "social dance events" with "<style> events".
    - If no suitable phrase is found, append a concise style note.
    """
    try:
        styles = detect_styles_in_text(user_text)
    except Exception:
        styles = []
    if not styles or not text:
        return text
    lower_text = text.lower()
    if any(st in lower_text for st in styles):
        return text
    style_phrase = ", ".join(styles) + " events"
    import re as _re
    new = _re.sub(r"(?i)\b(social\s+)?dance events\b", style_phrase, text, count=1)
    if new != text:
        return new
    # Fallback: append style note
    return text.rstrip('.') + f" (style: {', '.join(styles)})."

def generate_interpretation(user_query: str, config: dict, request_id: str | None = None) -> str:
    """
    Generate a natural language interpretation of the user's search intent.
    
    Args:
        user_query: The user's input query
        config: Configuration dictionary containing location settings
        
    Returns:
        str: Natural language interpretation of the search intent
    """
    # Helper: deterministic fallback using local date_calculator
    def _format_date(d: str) -> str:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            s = dt.strftime("%A, %B %d, %Y")
            return s.replace(" 0", " ")
        except Exception:
            return d

    def _fallback_interpretation(uq: str) -> str:
        try:
            from date_calculator import calculate_date_range, extract_temporal_phrase
            uq_l = uq.lower()
            tz_abbr = current_time.split()[-1]
            include_live = ("live music" in uq_l) or ("music" in uq_l)
            include_classes = any(w in uq_l for w in ["class", "classes", "workshop", "workshops", "lesson", "lessons"])
            temporal = extract_temporal_phrase(uq)

            if not temporal:
                return f"My understanding is that you want to see dance events available in the {default_city} area."

            rng = calculate_date_range(temporal, current_date)
            sd, ed = rng.get("start_date"), rng.get("end_date")
            time_filter = rng.get("time_filter")

            if temporal == "tonight" or temporal == "tomorrow night":
                when = "tonight" if temporal == "tonight" else "tomorrow night"
                date_txt = _format_date(sd)
                after_txt = f" after {time_filter[:5]} {tz_abbr}" if time_filter else ""
                parts = ["social dance events"]
                if include_classes:
                    parts.append("classes")
                if include_live:
                    parts.append("live music events")
                event_phrase = " and ".join([", ".join(parts[:-1])] + [parts[-1]]) if len(parts) > 1 else parts[0]
                return (
                    f"My understanding is that you want to see all {event_phrase} available in the {default_city} area {when}. "
                    f"That would be {('today, ' if temporal=='tonight' else '')}{date_txt}{after_txt}."
                )

            if temporal in ("this weekend", "next weekend"):
                # List Fri, Sat, Sun
                from datetime import timedelta
                d0 = datetime.strptime(sd, "%Y-%m-%d")
                days = [d0 + timedelta(days=i) for i in range(3)]
                days_txt = ", ".join([day.strftime("%A, %B %d").replace(" 0"," ") for day in days[:2]]) + \
                           f", and {days[2].strftime('%A, %B %d, %Y').replace(' 0',' ')}"
                parts = ["social dance events"]
                if include_classes:
                    parts.append("classes")
                if include_live:
                    parts.append("live music events")
                event_phrase = " and ".join([", ".join(parts[:-1])] + [parts[-1]]) if len(parts) > 1 else parts[0]
                return (
                    f"My understanding is that you want to see all {event_phrase} available in the {default_city} area {temporal}. "
                    f"That would be {days_txt}."
                )

            if temporal in ("this week", "next week"):
                parts = ["social dance events"]
                if include_classes:
                    parts.append("classes")
                if include_live:
                    parts.append("live music events")
                event_phrase = " and ".join([", ".join(parts[:-1])] + [parts[-1]]) if len(parts) > 1 else parts[0]
                return (
                    f"My understanding is that you want to see all {event_phrase} available in the {default_city} area {temporal}. "
                    f"That would be from {_format_date(sd)} to {_format_date(ed)}."
                )

            # Specific day
            parts = ["social dance events"]
            if include_classes:
                parts.append("classes")
            if include_live:
                parts.append("live music events")
            event_phrase = " and ".join([", ".join(parts[:-1])] + [parts[-1]]) if len(parts) > 1 else parts[0]
            return (
                f"My understanding is that you want to see all {event_phrase} available in the {default_city} area on {_format_date(sd)}."
            )
        except Exception:
            return f"My understanding is that you want to see dance events available in the {default_city} area."

    # Load interpretation prompt
    interpretation_prompt_path = os.path.join(base_dir, 'prompts', 'interpretation_prompt.txt')
    try:
        with open(interpretation_prompt_path, "r") as file:
            interpretation_template = file.read()
    except Exception as e:
        logging.error(f"Error reading interpretation prompt: {e}")
        return f"My understanding is that you want to search for: {user_query}"
    
    # Get current context in Pacific timezone
    pacific_tz = ZoneInfo("America/Los_Angeles")
    now_pacific = datetime.now(pacific_tz)
    current_date = now_pacific.strftime("%Y-%m-%d")
    current_day_of_week = now_pacific.strftime("%A")
    # Use %Z to automatically get PST or PDT based on daylight saving time
    current_time = now_pacific.strftime("%H:%M %Z")
    default_city = config.get('location', {}).get('epicentre', 'your area')
    
    # If the query contains a tool-recognized temporal phrase, prefer deterministic tool-based fallback
    from date_calculator import extract_temporal_phrase
    if extract_temporal_phrase(user_query):
        return _fallback_interpretation(user_query)

    # Format the interpretation prompt
    formatted_prompt = interpretation_template.format(
        current_date=current_date,
        current_day_of_week=current_day_of_week,
        current_time=current_time,
        default_city=default_city,
        user_query=user_query
    )

    # Query LLM for interpretation WITH date calculator tool
    from date_calculator import CALCULATE_DATE_RANGE_TOOL
    llm_request_id = request_id or _new_request_id("interp")
    interpretation = _query_llm_timed(
        request_id=llm_request_id,
        endpoint="/query",
        stage="interpretation_llm",
        request_url='chatbot',
        prompt=formatted_prompt,
        tools=[CALCULATE_DATE_RANGE_TOOL],
    )

    if interpretation:
        text = interpretation.strip()
        # If the user asked for "tonight" or "tomorrow night", ensure time filter is present
        uq_l = user_query.lower()
        if ("tonight" in uq_l or "tomorrow night" in uq_l):
            try:
                from date_calculator import calculate_date_range
                temporal = "tonight" if "tonight" in uq_l else "tomorrow night"
                rng = calculate_date_range(temporal, current_date)
                tf = rng.get("time_filter")
                if tf:
                    # Append a friendly time hint if not already present
                    hhmm = tf[:5]
                    tz_abbr = current_time.split()[-1]
                    if hhmm not in text and "after" not in text.lower():
                        text = text.rstrip('.') + f" after {hhmm} {tz_abbr}."
            except Exception:
                pass

        # Heuristic: ensure it includes default city; otherwise fallback to deterministic version
        if default_city.split(',')[0].split()[0].lower() in text.lower():
            return text

    # Deterministic fallback using local tool
    return _fallback_interpretation(user_query)

# Initialize the FastAPI app
app = FastAPI(title="Social Dance Chatbot API")

# Define the query request models
class QueryRequest(BaseModel):
    user_input: str
    session_token: str = None  # Optional session token for conversation context

class ConfirmationRequest(BaseModel):
    confirmation: str  # "yes", "clarify", or "no"
    session_token: str  # Required for confirmation
    clarification: str = None  # Optional clarification text for "clarify" option

@app.get("/")
def read_root():
    return {"message": "Welcome to the Social Dance Chatbot API!"}

@app.post("/confirm")
def process_confirmation(request: ConfirmationRequest):
    """
    Handle user confirmations for pending queries.
    """
    request_id = _new_request_id("confirm")
    request_started = _log_timing_start(
        request_id,
        "/confirm",
        "request_total",
        session_suffix=_session_suffix(request.session_token),
    )
    result_type = "unknown"
    def _finish_confirmation_timing() -> None:
        _log_timing_end(
            request_id,
            "/confirm",
            "request_total",
            request_started,
            result_type=result_type,
        )
    confirmation = request.confirmation.lower().strip()
    session_token = request.session_token
    clarification = request.clarification
    
    if not session_token:
        result_type = "invalid_missing_session"
        _finish_confirmation_timing()
        raise HTTPException(status_code=400, detail="Session token is required for confirmations.")
    
    # Get conversation and pending query
    conversation_id = conversation_manager.create_or_get_conversation(session_token)
    pending_query = conversation_manager.get_pending_query(conversation_id)
    
    if not pending_query:
        result_type = "invalid_missing_pending_query"
        _finish_confirmation_timing()
        raise HTTPException(status_code=400, detail="No pending query found for confirmation.")
    
    if confirmation == "yes":
        # Execute the pending SQL query (regenerate if missing/invalid)
        try:
            pacific_tz = ZoneInfo("America/Los_Angeles")
            now_pacific = datetime.now(pacific_tz)
            current_date = now_pacific.strftime("%Y-%m-%d")

            sanitized_query = pending_query.get("sql_query") or ""
            pending_constraints = pending_query.get("constraints") or {}
            if pending_constraints:
                constraint_sql = build_sql_from_constraints(pending_constraints)
                if constraint_sql:
                    sanitized_query = constraint_sql

            if not sanitized_query.upper().startswith("SELECT"):
                # Rebuild prompt and regenerate SQL now that the user confirmed intent
                prompts_cfg = config.get('prompts', {})
                contextual_cfg = prompts_cfg.get('contextual_sql', {})
                contextual_path_rel = (
                    contextual_cfg.get('file') if isinstance(contextual_cfg, dict)
                    else 'prompts/contextual_sql_prompt.txt'
                )
                prompt_file_path = os.path.join(base_dir, contextual_path_rel)
                with open(prompt_file_path, "r") as f:
                    base_prompt = f.read()

                ctx = conversation_manager.get_conversation_context(conversation_id)
                recent = conversation_manager.get_recent_messages(conversation_id, limit=3)
                history_text = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent])

                current_day = now_pacific.isoweekday()

                prompt = base_prompt.format(
                    context_info=str(ctx),
                    conversation_history=history_text,
                    intent=pending_query.get('intent','search'),
                    entities="{}",
                    current_date=current_date,
                    current_day_of_week=current_day
                )
                combined_q = pending_query.get('combined_query') or pending_query.get('user_input')
                if combined_q:
                    prompt += f"\n\nCurrent User Question: \"{combined_q}\""

                from date_calculator import CALCULATE_DATE_RANGE_TOOL
                sql_raw = _query_llm_timed(
                    request_id=request_id,
                    endpoint="/confirm",
                    stage="confirm_sql_generation_primary",
                    request_url='chatbot',
                    prompt=prompt,
                    tools=[CALCULATE_DATE_RANGE_TOOL],
                )
                if sql_raw:
                    s = sql_raw.replace("```sql", "").replace("```", "").strip()
                    si = s.upper().find("SELECT")
                    if si != -1:
                        s = s[si:]
                    s = s.split(";")[0]
                    if s.upper().startswith('SELECT') and not _sql_has_illegal_date_arithmetic(s):
                        sanitized_query = _enforce_default_event_type(s, combined_q)
                        sanitized_query = enforce_dance_style(sanitized_query, combined_q)
                    else:
                        # Strict SQL-only retry
                        sql_only_suffix = (
                            "\n\nSTRICT FIX: Return ONLY a raw SQL SELECT statement (no tool calls, no JSON, no explanations). "
                            "Call calculate_date_range internally and embed the dates directly in WHERE clauses."
                        )
                        strict_prompt = f"{prompt}\n{sql_only_suffix}"
                        sql_raw2 = _query_llm_timed(
                            request_id=request_id,
                            endpoint="/confirm",
                            stage="confirm_sql_generation_retry",
                            request_url='chatbot',
                            prompt=strict_prompt,
                            tools=[CALCULATE_DATE_RANGE_TOOL],
                        )
                        if sql_raw2:
                            s2 = sql_raw2.replace("```sql", "").replace("```", "").strip()
                            si2 = s2.upper().find("SELECT")
                            if si2 != -1:
                                s2 = s2[si2:]
                            s2 = s2.split(";")[0]
                            if s2.upper().startswith('SELECT') and not _sql_has_illegal_date_arithmetic(s2):
                                sanitized_query = _enforce_default_event_type(s2, combined_q)
            # Pre-execution repair + validation to prevent malformed SQL from reaching users.
            sanitized_query = _repair_common_sql_issues(sanitized_query)
            validate_1_started = _log_timing_start(request_id, "/confirm", "sql_validate_pre_fallback")
            is_valid_sql, validation_error = _validate_sql_select(sanitized_query)
            _log_timing_end(
                request_id,
                "/confirm",
                "sql_validate_pre_fallback",
                validate_1_started,
                is_valid=is_valid_sql,
            )

            if not is_valid_sql:
                logging.warning("CONFIRMATION: SQL validation failed, using deterministic fallback. Error: %s", validation_error)
                # Deterministic SQL fallback using date_calculator
                from date_calculator import calculate_date_range, extract_temporal_phrase
                cq = pending_query.get('combined_query') or pending_query.get('user_input') or ''
                temporal = extract_temporal_phrase(cq)
                if temporal:
                    rng = calculate_date_range(temporal, current_date)
                    sd, ed = rng.get('start_date'), rng.get('end_date')
                    tf = rng.get('time_filter')
                    cols = (
                        "event_name, event_type, dance_style, day_of_week, start_date, end_date, "
                        "start_time, end_time, source, url, price, description, location"
                    )
                    # Base filters: date range (and time filter if provided)
                    filters = [f"start_date >= '{sd}'", f"start_date <= '{ed}'"]
                    if tf:
                        filters.append(f"start_time >= '{tf}'")

                    # Detect explicit event_type in user text and apply centralized policy when absent.
                    import re as _re_detect
                    explicit_in_text = bool(_re_detect.search(r"(?i)\bevent_type\s*(=|ILIKE|LIKE|>=|<=|>|<|:)", cq))
                    has_explicit_event_type = explicit_in_text or any('event_type' in f.lower() for f in filters)
                    if not has_explicit_event_type:
                        policy_clause = _derive_event_type_policy_clause(cq)
                        if policy_clause:
                            filters.append(policy_clause)

                    # Optional: parse simple column filters from the confirmed question (safe subset)
                    # Pattern: <column> <op> '<value>' where column is a known column and op in =, ILIKE, LIKE, >=, <=, >, <
                    try:
                        # Dynamically discover valid columns from the events table
                        try:
                            events_table = db_handler.metadata.tables.get('events')
                            allowed_cols = set([c.name for c in events_table.columns]) if events_table else set()
                        except Exception:
                            allowed_cols = set()

                        import re as _re
                        # Pattern 1: SQL-like conditions with quoted values
                        for m in _re.finditer(r"(?i)\b([a-z_]+)\s*(=|ILIKE|LIKE|>=|<=|>|<)\s*'([^']*)'", cq):
                            col, op, val = m.group(1).lower(), m.group(2).upper(), m.group(3)
                            if allowed_cols and col not in allowed_cols:
                                continue
                            safe_val = val.replace("'", "")
                            # Prefer ILIKE with wildcards for textual comparisons when '=' is used
                            if op == '=':
                                op = 'ILIKE'
                            if op in ('LIKE','ILIKE') and '%' not in safe_val:
                                safe_val = f"%{safe_val}%"
                            filters.append(f"{col} {op} '{safe_val}'")

                        # Pattern 2: shorthand "col: value" or "col=value" without quotes → ILIKE '%value%'
                        for m in _re.finditer(r"(?i)\b([a-z_]+)\s*[:=]\s*([\w\-\s%\./]+)", cq):
                            col, val = m.group(1).lower(), m.group(2).strip()
                            if allowed_cols and col not in allowed_cols:
                                continue
                            if not val:
                                continue
                            safe_val = val.replace("'", "")
                            if '%' not in safe_val:
                                safe_val = f"%{safe_val}%"
                            filters.append(f"{col} ILIKE '{safe_val}'")
                    except Exception:
                        pass
                    sanitized_query = (
                        f"SELECT {cols} FROM events WHERE " + " AND ".join(filters) +
                        " ORDER BY start_date, start_time LIMIT 30"
                    )
                    logging.info("CONFIRMATION: Using deterministic SQL fallback.")
                else:
                    raise ValueError("Could not generate a valid SQL query from confirmation.")

            # Final enforcement: ensure default social dance is included unless explicitly restricted
            cq_final = pending_query.get('combined_query') or pending_query.get('user_input') or ''
            sanitized_query = _enforce_default_event_type(sanitized_query, cq_final)
            sanitized_query = enforce_dance_style(sanitized_query, cq_final)
            sanitized_query = _repair_common_sql_issues(sanitized_query)

            validate_2_started = _log_timing_start(request_id, "/confirm", "sql_validate_final")
            is_valid_sql, validation_error = _validate_sql_select(sanitized_query)
            _log_timing_end(
                request_id,
                "/confirm",
                "sql_validate_final",
                validate_2_started,
                is_valid=is_valid_sql,
            )
            if not is_valid_sql:
                raise ValueError(f"Unable to build a valid query for this request. Validation: {validation_error}")

            display_sql = _format_sql_for_display(sanitized_query)
            logging.info(f"CONFIRMATION: Executing confirmed query: {sanitized_query}")
            logging.info(
                "chatbot_trace_sql: request_id=%s endpoint=/confirm stage=confirmed_sql sql=%s",
                request_id,
                _safe_log_snippet(sanitized_query, max_len=320),
            )
            _persist_request_trace(request_id=request_id, sql_snippet=_safe_log_snippet(sanitized_query, max_len=1000))

            rows = _execute_query_timed(
                request_id=request_id,
                endpoint="/confirm",
                stage="execute_confirmed_sql",
                sql_query=sanitized_query,
            )
            if rows is None:
                raise ValueError("Query execution failed before result retrieval.")
            if not rows:
                data = []
            else:
                columns = [
                    'event_name', 'event_type', 'dance_style', 'day_of_week',
                    'start_date', 'end_date', 'start_time', 'end_time', 'source',
                    'url', 'price', 'description', 'location'
                ]
                data = [dict(zip(columns, row)) for row in rows]
            
            # Add assistant message and clear pending query
            conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=f"Found {len(data)} events",
                sql_query=display_sql,
                result_count=len(data)
            )
            
            conversation_manager.clear_pending_query(conversation_id)
            
            result_type = "confirmed_results"
            _finish_confirmation_timing()
            return {
                "sql_query": display_sql,
                "data": data,
                "message": "Here are the results from your confirmed query.",
                "conversation_id": conversation_id,
                "confirmed": True
            }
            
        except ChatbotLLMTimeoutError:
            logging.error("CONFIRMATION: LLM timeout while resolving confirmed query.")
            conversation_manager.clear_pending_query(conversation_id)
            result_type = "confirm_llm_timeout"
            _finish_confirmation_timing()
            raise HTTPException(
                status_code=503,
                detail=_chatbot_traffic_timeout_message(),
            )
        except Exception as db_err:
            logging.error("CONFIRMATION: Query execution pipeline failed: %s", db_err)
            conversation_manager.clear_pending_query(conversation_id)
            result_type = "confirm_error"
            _finish_confirmation_timing()
            raise HTTPException(
                status_code=500,
                detail="I couldn't run that search right now. Please try again with a slightly reworded request."
            )
    
    elif confirmation == "clarify":
        # Handle clarification.
        # Apply clarification as structured constraint updates instead of string concatenation.
        if not clarification:
            raise HTTPException(status_code=400, detail="Clarification text is required when selecting 'clarify' option.")

        clarification_input = clarification.strip()
        pacific_tz = ZoneInfo("America/Los_Angeles")
        now_pacific = datetime.now(pacific_tz)
        current_date = now_pacific.strftime("%Y-%m-%d")
        default_base_query = pending_query.get("combined_query") or pending_query.get("user_input") or ""
        base_constraints = pending_query.get("constraints") or derive_constraints_from_text(
            default_base_query,
            current_date,
        )
        try:
            updated_constraints = derive_constraints_from_text(
                clarification_input,
                current_date,
                base_constraints=base_constraints,
                is_clarification=True,
            )
        except Exception:
            updated_constraints = base_constraints

        rewritten_query = constraints_to_query_text(updated_constraints, fallback_text=clarification_input)
        interpretation = generate_interpretation(rewritten_query, config, request_id=request_id)
        interpretation = _force_style_in_interpretation(interpretation, rewritten_query)

        sql_from_constraints = build_sql_from_constraints(updated_constraints)
        if sql_from_constraints:
            sql_from_constraints = _enforce_default_event_type(sql_from_constraints, rewritten_query)
            sql_from_constraints = enforce_dance_style(sql_from_constraints, rewritten_query)

        # Replace pending query with the clarified structured query.
        conversation_manager.clear_pending_query(conversation_id)
        conversation_manager.store_pending_query(
            conversation_id=conversation_id,
            user_input=clarification_input,
            combined_query=rewritten_query,
            interpretation=interpretation,
            sql_query=sql_from_constraints if sql_from_constraints else None,
            constraints=updated_constraints,
        )
        conversation_manager.update_conversation_context(
            conversation_id,
            {"last_search_query": "", "concatenation_count": 1},
        )
        result_type = "clarify_updated"
        _finish_confirmation_timing()
        return {
            "interpretation": interpretation,
            "confirmation_required": True,
            "conversation_id": conversation_id,
            "message": f"{interpretation}\n\nIf that is correct, please confirm using the buttons below:",
            "options": ["yes", "clarify", "no"],
            "sql_query": sql_from_constraints if sql_from_constraints else None,
        }
    
    elif confirmation == "no":
        # User rejected the interpretation - clear pending query
        conversation_manager.clear_pending_query(conversation_id)
        
        result_type = "cancelled"
        _finish_confirmation_timing()
        return {
            "message": "Query cancelled. Please provide a new search request.",
            "conversation_id": conversation_id,
            "cancelled": True
        }
    
    else:
        result_type = "invalid_confirmation_option"
        _finish_confirmation_timing()
        raise HTTPException(status_code=400, detail="Invalid confirmation option. Use 'yes', 'clarify', or 'no'.")

@app.post("/query")
def process_query(request: QueryRequest):
    request_id = _new_request_id("query")
    request_started = _log_timing_start(
        request_id,
        "/query",
        "request_total",
        session_suffix=_session_suffix(request.session_token),
        input_len=len((request.user_input or "").strip()),
    )
    result_type = "unknown"
    def _finish_query_timing() -> None:
        _log_timing_end(
            request_id,
            "/query",
            "request_total",
            request_started,
            result_type=result_type,
        )
    user_input = request.user_input.strip()
    if not user_input:
        result_type = "invalid_input"
        _finish_query_timing()
        raise HTTPException(status_code=400, detail="User input is empty.")
    logging.info(
        "chatbot_trace_request: request_id=%s endpoint=/query user_input=%s",
        request_id,
        _safe_log_snippet(user_input),
    )
    _persist_request_trace(request_id=request_id, user_input=_safe_log_snippet(user_input, max_len=500))
    
    # Handle session-based conversation context
    session_token = request.session_token
    use_contextual_prompt = session_token is not None
    
    # Initialize variables for both contextual and non-contextual paths
    conversation_id = None
    intent = None
    entities = {}
    
    if use_contextual_prompt:
        context_started = _log_timing_start(
            request_id,
            "/query",
            "context_preparation",
            has_session=True,
        )
        try:
            # Get or create conversation
            conversation_id = conversation_manager.create_or_get_conversation(session_token)
            
            # Get conversation context and recent messages
            context = conversation_manager.get_conversation_context(conversation_id)
            recent_messages = conversation_manager.get_recent_messages(conversation_id, limit=5)
            
            # Classify intent and extract entities
            intent = conversation_manager.classify_intent(user_input, context, recent_messages)
            entities = conversation_manager.extract_entities(user_input, context)
            
            # Add user message to conversation
            conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user", 
                content=user_input,
                intent=intent,
                entities=entities
            )
            
            # Get updated recent messages INCLUDING the current user message
            recent_messages_updated = conversation_manager.get_recent_messages(conversation_id, limit=5)
            
            # Use contextual prompt template
            prompt_file_path = os.path.join(base_dir, 'prompts', 'contextual_sql_prompt.txt')
            with open(prompt_file_path, "r") as file:
                base_prompt = file.read()
            
            # Format conversation history for prompt (include current user message)
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_messages_updated[-3:]  # Last 3 messages for context
            ])
            
            # Get current date context in Pacific timezone
            pacific_tz = ZoneInfo("America/Los_Angeles")
            now_pacific = datetime.now(pacific_tz)
            current_date = now_pacific.strftime("%Y-%m-%d")
            current_day_of_week = now_pacific.isoweekday()  # Monday=1, Sunday=7
            
            # Handle query concatenation for refinements (up to 5 parts)
            if intent == 'refinement':
                # Get the current combined query and concatenation count from context
                current_combined_query = context.get('last_search_query', '')
                concatenation_count = context.get('concatenation_count', 1)
                
                if current_combined_query and concatenation_count < 5:
                    # Concatenate current combined query with new input
                    combined_query = f"{current_combined_query} {user_input}"
                    concatenation_count += 1
                    logging.info(f"REFINEMENT #{concatenation_count}: Combining '{current_combined_query}' + '{user_input}' = '{combined_query}'")
                elif concatenation_count >= 5:
                    # Max concatenations reached, treat as new search
                    combined_query = user_input
                    concatenation_count = 1
                    logging.info(f"REFINEMENT: Max concatenations (5) reached, treating as new search: '{user_input}'")
                else:
                    combined_query = user_input
                    concatenation_count = 1
                    logging.warning("REFINEMENT: No original query found in context, using current input only")
                
                # Update context with the new combined query and count
                context['last_search_query'] = combined_query
                context['concatenation_count'] = concatenation_count
            else:
                # New search - use input as-is and reset concatenation count
                combined_query = user_input
                # Store the query and reset concatenation count for future refinements
                context['last_search_query'] = user_input
                context['concatenation_count'] = 1
                logging.info(f"NEW SEARCH: Storing query: '{user_input}' (concatenation count reset to 1)")
            
            # Construct contextual prompt
            prompt = base_prompt.format(
                context_info=str(context),
                conversation_history=history_text,
                intent=intent,
                entities=str(entities),
                current_date=current_date,
                current_day_of_week=current_day_of_week
            )
            prompt += f"\n\nCurrent User Question: \"{combined_query}\""
            
            # DEBUG: Log the full prompt to see what's being sent to LLM
            logging.info("=== FULL PROMPT BEING SENT TO LLM ===")
            logging.info(prompt)
            logging.info("=== END PROMPT ===")
            
        except Exception as e:
            logging.error(f"Error with contextual conversation: {e}")
            # Fall back to non-contextual mode
            use_contextual_prompt = False
        finally:
            _log_timing_end(
                request_id,
                "/query",
                "context_preparation",
                context_started,
                contextual_used=use_contextual_prompt,
            )
    
    if not use_contextual_prompt:
        fallback_prompt_started = _log_timing_start(
            request_id,
            "/query",
            "fallback_prompt_preparation",
            has_session=False,
        )
        # Use contextual SQL prompt even in fallback (no history/context)
        prompts_cfg = config.get('prompts', {})
        contextual_cfg = prompts_cfg.get('contextual_sql', {})
        contextual_path_rel = (
            contextual_cfg.get('file') if isinstance(contextual_cfg, dict)
            else 'prompts/contextual_sql_prompt.txt'
        )
        prompt_file_path = os.path.join(base_dir, contextual_path_rel)

        try:
            with open(prompt_file_path, "r") as file:
                base_prompt = file.read()
        except Exception as e:
            logging.error(f"Error reading contextual SQL prompt file: {e}")
            result_type = "prompt_file_error"
            _finish_query_timing()
            raise HTTPException(status_code=500, detail="Error reading contextual SQL prompt file.")

        # Minimal context for non-session queries
        pacific_tz = ZoneInfo("America/Los_Angeles")
        now_pacific = datetime.now(pacific_tz)
        current_date = now_pacific.strftime("%Y-%m-%d")
        current_day_of_week = now_pacific.isoweekday()  # Monday=1, Sunday=7

        prompt = base_prompt.format(
            context_info="",
            conversation_history="",
            intent="search",
            entities="{}",
            current_date=current_date,
            current_day_of_week=current_day_of_week
        )
        prompt += f"\n\nCurrent User Question: \"{user_input}\""

        # DEBUG: Log that we're using contextual prompt in fallback
        logging.info("=== USING CONTEXTUAL PROMPT (FALLBACK) ===")
        logging.info("=== FULL PROMPT BEING SENT TO LLM ===")
        logging.info(prompt)
        logging.info("=== END PROMPT ===")
        _log_timing_end(request_id, "/query", "fallback_prompt_preparation", fallback_prompt_started)
    
    logging.info(f"Constructed Prompt: {prompt}")

    # Query the language model for a raw SQL query with date calculator tool support
    from date_calculator import CALCULATE_DATE_RANGE_TOOL
    try:
        sql_query = _query_llm_timed(
            request_id=request_id,
            endpoint="/query",
            stage="sql_generation_primary",
            request_url='chatbot',
            prompt=prompt,
            tools=[CALCULATE_DATE_RANGE_TOOL],
        )
    except ChatbotLLMTimeoutError:
        result_type = "llm_timeout"
        _finish_query_timing()
        return {
            "message": _chatbot_traffic_timeout_message(),
            "confirmation_required": False,
            "retry_recommended": True,
        }
    logging.info(f"Raw SQL Query: {sql_query}")

    # Always generate interpretation and confirmation, even if SQL didn't come back yet
    sanitized_query = None
    if sql_query:
        sanitized_query = sql_query.replace("```sql", "").replace("```", "").strip()
        select_index = sanitized_query.find("SELECT")
        if (select_index != -1):
            sanitized_query = sanitized_query[select_index:]
        sanitized_query = sanitized_query.split(";")[0]
        logging.info(f"Sanitized SQL Query: {sanitized_query}")
        logging.info(
            "chatbot_trace_sql: request_id=%s endpoint=/query stage=sql_sanitized sql=%s",
            request_id,
            _safe_log_snippet(sanitized_query, max_len=320),
        )
        _persist_request_trace(request_id=request_id, sql_snippet=_safe_log_snippet(sanitized_query, max_len=1000))

        # Preflight: reject illegal date arithmetic and re-query with strict reminder
        if _sql_has_illegal_date_arithmetic(sanitized_query):
            logging.warning("Preflight: Detected illegal date arithmetic in SQL. Re-querying with strict date rules.")
            strict_suffix = (
                "\n\nSTRICT FIX: You MUST call calculate_date_range for ANY temporal expression and use ONLY the returned dates. "
                "Never add/subtract integers to dates (e.g., CURRENT_DATE + 7). If referencing CURRENT_DATE, use INTERVAL syntax only, "
                "but prefer explicit dates from the tool. Return ONLY SQL."
            )
            strict_prompt = f"{prompt}\n{strict_suffix}"
            sql_query2 = _query_llm_timed(
                request_id=request_id,
                endpoint="/query",
                stage="sql_generation_strict_retry",
                request_url='chatbot',
                prompt=strict_prompt,
                tools=[CALCULATE_DATE_RANGE_TOOL],
            )
            if sql_query2:
                sanitized_query2 = sql_query2.replace("```sql", "").replace("```", "").strip()
                select_index2 = sanitized_query2.find("SELECT")
                if select_index2 != -1:
                    sanitized_query2 = sanitized_query2[select_index2:]
                sanitized_query2 = sanitized_query2.split(";")[0]
                if not _sql_has_illegal_date_arithmetic(sanitized_query2):
                    sanitized_query = sanitized_query2
                    logging.info("Preflight: Successfully regenerated SQL without illegal date arithmetic.")
                    logging.info(
                        "chatbot_trace_sql: request_id=%s endpoint=/query stage=sql_strict_retry sql=%s",
                        request_id,
                        _safe_log_snippet(sanitized_query, max_len=320),
                    )
                    _persist_request_trace(request_id=request_id, sql_snippet=_safe_log_snippet(sanitized_query, max_len=1000))
                else:
                    logging.warning("Preflight: Regenerated SQL still contains illegal date arithmetic; proceeding with interpretation but execution may fail.")

        # Ensure we actually have a SELECT statement; otherwise re-query with stricter instructions
        if sanitized_query and not sanitized_query.upper().startswith("SELECT"):
            # Prefer deterministic SQL synthesis from parsed constraints before retrying LLM.
            constraint_source_query = combined_query if use_contextual_prompt else user_input
            synthesized_constraints = derive_constraints_from_text(
                constraint_source_query,
                current_date,
            )
            deterministic_sql = build_sql_from_constraints(synthesized_constraints)
            if deterministic_sql:
                sanitized_query = deterministic_sql
                logging.info("Preflight: Built deterministic SQL from constraints after non-SELECT LLM output.")
                logging.info(
                    "chatbot_trace_sql: request_id=%s endpoint=/query stage=sql_constraints_fallback sql=%s",
                    request_id,
                    _safe_log_snippet(sanitized_query, max_len=320),
                )
                _persist_request_trace(request_id=request_id, sql_snippet=_safe_log_snippet(sanitized_query, max_len=1000))
            else:
                logging.warning("Preflight: No valid SELECT found. Re-querying with explicit SQL-only instruction.")
                sql_only_suffix = (
                    "\n\nSTRICT FIX: Return ONLY a raw SQL SELECT statement (no tool calls, no JSON, no explanations). "
                    "Call calculate_date_range internally and embed the dates directly in WHERE clauses."
                )
                sql_only_prompt = f"{prompt}\n{sql_only_suffix}"
                try:
                    sql_query3 = _query_llm_timed(
                        request_id=request_id,
                        endpoint="/query",
                        stage="sql_generation_select_retry",
                        request_url='chatbot',
                        prompt=sql_only_prompt,
                        tools=[CALCULATE_DATE_RANGE_TOOL],
                    )
                except ChatbotLLMTimeoutError:
                    result_type = "llm_timeout"
                    _finish_query_timing()
                    return {
                        "message": _chatbot_traffic_timeout_message(),
                        "confirmation_required": False,
                        "retry_recommended": True,
                    }

                if sql_query3:
                    s3 = sql_query3.replace("```sql", "").replace("```", "").strip()
                    si3 = s3.upper().find("SELECT")
                    if si3 != -1:
                        s3 = s3[si3:]
                    s3 = s3.split(";")[0]
                    if s3.upper().startswith("SELECT") and not _sql_has_illegal_date_arithmetic(s3):
                        sanitized_query = s3
                        logging.info("Preflight: Successfully regenerated a valid SELECT SQL.")
                        logging.info(
                            "chatbot_trace_sql: request_id=%s endpoint=/query stage=sql_select_retry sql=%s",
                            request_id,
                            _safe_log_snippet(sanitized_query, max_len=320),
                        )
                        _persist_request_trace(request_id=request_id, sql_snippet=_safe_log_snippet(sanitized_query, max_len=1000))

    # Generate natural language interpretation and always confirm intent
    try:
        if use_contextual_prompt:
            query_for_interpretation = combined_query
        else:
            query_for_interpretation = user_input

        interpretation = generate_interpretation(
            query_for_interpretation,
            config,
            request_id=request_id,
        )
        interpretation = _force_style_in_interpretation(interpretation, query_for_interpretation)
        logging.info(f"Generated interpretation: {interpretation}")

        if use_contextual_prompt and session_token:
            try:
                pending_constraints = derive_constraints_from_text(
                    query_for_interpretation,
                    current_date,
                )
                if pending_constraints:
                    deterministic_sql = build_sql_from_constraints(pending_constraints)
                    if deterministic_sql:
                        sanitized_query = deterministic_sql

                conversation_manager.store_pending_query(
                    conversation_id=conversation_id,
                    user_input=user_input,
                    combined_query=query_for_interpretation,
                    interpretation=interpretation,
                    sql_query=sanitized_query if (sanitized_query and sanitized_query.upper().startswith('SELECT')) else None,
                    constraints=pending_constraints,
                )
                search_context = {
                    "last_search_query": context.get('last_search_query', combined_query),
                    "concatenation_count": context.get('concatenation_count', 1)
                }
                conversation_manager.update_conversation_context(conversation_id, search_context)
            except Exception as e:
                logging.error(f"Error storing pending query: {e}")
                raise HTTPException(status_code=500, detail=f"Error storing query for confirmation: {e}")

        # Ensure dance_style filter is present when the user explicitly requested a style
        try:
            qfe = combined_query if use_contextual_prompt else user_input
            if sanitized_query and sanitized_query.upper().startswith('SELECT'):
                sanitized_query = enforce_dance_style(sanitized_query, qfe)
        except Exception:
            pass

        result_type = "confirmation_required"
        return {
            "interpretation": interpretation,
            "confirmation_required": True,
            "conversation_id": conversation_id if use_contextual_prompt else None,
            "intent": intent if use_contextual_prompt else None,
            "message": f"{interpretation}\n\nIf that is correct, please confirm using the buttons below:",
            "options": ["yes", "clarify", "no"],
            "sql_query": sanitized_query if (sanitized_query and sanitized_query.upper().startswith('SELECT')) else None
        }

    except ChatbotLLMTimeoutError:
        result_type = "llm_timeout"
        return {
            "message": _chatbot_traffic_timeout_message(),
            "confirmation_required": False,
            "retry_recommended": True,
        }
    except Exception as e:
        logging.error(f"Error generating interpretation: {e}")
        result_type = "confirmation_fallback"
        return {
            "message": f"I understand you want to search for: {user_input}. Please confirm if this is correct.",
            "confirmation_required": True,
            "simple_confirmation": True
        }
    finally:
        _finish_query_timing()
