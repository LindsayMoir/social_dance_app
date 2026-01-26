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
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import re

# Set up sys.path so that modules in src/ are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
sys.path.append(parent_dir)
print("Updated sys.path:", sys.path)
print("Current working directory:", os.getcwd())

from llm import LLMHandler  # Import the LLMHandler module
from db import DatabaseHandler  # Import DatabaseHandler for conversation management
from conversation_manager import ConversationManager  # Import ConversationManager

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

def generate_interpretation(user_query: str, config: dict) -> str:
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
            from date_calculator import calculate_date_range
            uq_l = uq.lower()
            tz_abbr = current_time.split()[-1]
            include_live = ("live music" in uq_l) or ("music" in uq_l)
            include_classes = any(w in uq_l for w in ["class", "classes", "workshop", "workshops", "lesson", "lessons"])

            # Map simple phrases
            phrases = [
                "tonight", "tomorrow night", "tomorrow",
                "this weekend", "next weekend",
                "this week", "next week",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
            ]
            temporal = next((p for p in phrases if p in uq_l), None)

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
    
    # If the query contains a clear temporal phrase, prefer deterministic tool-based fallback
    uq_l_pref = user_query.lower()
    if any(p in uq_l_pref for p in [
        "tonight", "tomorrow night", "tomorrow", "this weekend", "next weekend",
        "this week", "next week", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ]):
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
    interpretation = llm_handler.query_llm('', formatted_prompt, tools=[CALCULATE_DATE_RANGE_TOOL])

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
    confirmation = request.confirmation.lower().strip()
    session_token = request.session_token
    clarification = request.clarification
    
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token is required for confirmations.")
    
    # Get conversation and pending query
    conversation_id = conversation_manager.create_or_get_conversation(session_token)
    pending_query = conversation_manager.get_pending_query(conversation_id)
    
    if not pending_query:
        raise HTTPException(status_code=400, detail="No pending query found for confirmation.")
    
    if confirmation == "yes":
        # Execute the pending SQL query (regenerate if missing/invalid)
        try:
            sanitized_query = pending_query.get("sql_query") or ""
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

                pacific_tz = ZoneInfo("America/Los_Angeles")
                now_pacific = datetime.now(pacific_tz)
                current_date = now_pacific.strftime("%Y-%m-%d")
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
                sql_raw = llm_handler.query_llm('', prompt, tools=[CALCULATE_DATE_RANGE_TOOL])
                if sql_raw:
                    s = sql_raw.replace("```sql", "").replace("```", "").strip()
                    si = s.upper().find("SELECT")
                    if si != -1:
                        s = s[si:]
                    s = s.split(";")[0]
                    if s.upper().startswith('SELECT') and not _sql_has_illegal_date_arithmetic(s):
                        sanitized_query = s
                    else:
                        # Strict SQL-only retry
                        sql_only_suffix = (
                            "\n\nSTRICT FIX: Return ONLY a raw SQL SELECT statement (no tool calls, no JSON, no explanations). "
                            "Call calculate_date_range internally and embed the dates directly in WHERE clauses."
                        )
                        strict_prompt = f"{prompt}\n{sql_only_suffix}"
                        sql_raw2 = llm_handler.query_llm('', strict_prompt, tools=[CALCULATE_DATE_RANGE_TOOL])
                        if sql_raw2:
                            s2 = sql_raw2.replace("```sql", "").replace("```", "").strip()
                            si2 = s2.upper().find("SELECT")
                            if si2 != -1:
                                s2 = s2[si2:]
                            s2 = s2.split(";")[0]
                            if s2.upper().startswith('SELECT') and not _sql_has_illegal_date_arithmetic(s2):
                                sanitized_query = s2
            if not sanitized_query.upper().startswith('SELECT'):
                # Deterministic SQL fallback using date_calculator
                from date_calculator import calculate_date_range
                cq = pending_query.get('combined_query') or pending_query.get('user_input') or ''
                cq_l = cq.lower()
                phrases = [
                    "tonight", "tomorrow night", "tomorrow",
                    "this weekend", "next weekend",
                    "this week", "next week",
                    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
                ]
                temporal = next((p for p in phrases if p in cq_l), None)
                if temporal:
                    rng = calculate_date_range(temporal, current_date)
                    sd, ed = rng.get('start_date'), rng.get('end_date')
                    tf = rng.get('time_filter')
                    cols = (
                        "event_name, event_type, dance_style, day_of_week, start_date, end_date, "
                        "start_time, end_time, source, url, price, description, location"
                    )
                    # Keep fallback minimally restrictive: only date range (and time filter if provided).
                    filters = [f"start_date >= '{sd}'", f"start_date <= '{ed}'"]
                    # If user asked to include classes/live, add those too (without excluding others)
                    include_live = ("live music" in cq_l) or ("music" in cq_l)
                    include_classes = any(w in cq_l for w in ["class", "classes", "workshop", "workshops", "lesson", "lessons"])
                    or_clauses = []
                    if include_classes:
                        or_clauses.append("(event_type ILIKE '%class%' OR event_type ILIKE '%workshop%')")
                    if include_live:
                        or_clauses.append("event_type ILIKE '%live music%'")
                    if or_clauses:
                        filters.append("( " + " OR ".join(or_clauses) + " )")
                    if tf:
                        filters.append(f"start_time >= '{tf}'")

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

            display_sql = _format_sql_for_display(sanitized_query)
            logging.info(f"CONFIRMATION: Executing confirmed query: {sanitized_query}")

            rows = db_handler.execute_query(sanitized_query)
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
            
            return {
                "sql_query": display_sql,
                "data": data,
                "message": "Here are the results from your confirmed query.",
                "conversation_id": conversation_id,
                "confirmed": True
            }
            
        except Exception as db_err:
            error_message = f"Database Error: {db_err}"
            logging.error(error_message)
            conversation_manager.clear_pending_query(conversation_id)
            raise HTTPException(status_code=500, detail=error_message)
    
    elif confirmation == "clarify":
        # Handle clarification — merge with prior combined query for better context
        if not clarification:
            raise HTTPException(status_code=400, detail="Clarification text is required when selecting 'clarify' option.")

        prior = pending_query.get('combined_query') or pending_query.get('user_input') or ''
        merged = (prior + ' ' + clarification).strip()

        # Clear pending and re-run with merged input
        conversation_manager.clear_pending_query(conversation_id)

        clarification_request = QueryRequest(user_input=merged, session_token=session_token)
        return process_query(clarification_request)
    
    elif confirmation == "no":
        # User rejected the interpretation - clear pending query
        conversation_manager.clear_pending_query(conversation_id)
        
        return {
            "message": "Query cancelled. Please provide a new search request.",
            "conversation_id": conversation_id,
            "cancelled": True
        }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid confirmation option. Use 'yes', 'clarify', or 'no'.")

@app.post("/query")
def process_query(request: QueryRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is empty.")
    
    # Handle session-based conversation context
    session_token = request.session_token
    use_contextual_prompt = session_token is not None
    
    # Initialize variables for both contextual and non-contextual paths
    conversation_id = None
    intent = None
    entities = {}
    
    if use_contextual_prompt:
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
    
    if not use_contextual_prompt:
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
    
    logging.info(f"Constructed Prompt: {prompt}")

    # Query the language model for a raw SQL query with date calculator tool support
    from date_calculator import CALCULATE_DATE_RANGE_TOOL
    sql_query = llm_handler.query_llm('', prompt, tools=[CALCULATE_DATE_RANGE_TOOL])
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

        # Preflight: reject illegal date arithmetic and re-query with strict reminder
        if _sql_has_illegal_date_arithmetic(sanitized_query):
            logging.warning("Preflight: Detected illegal date arithmetic in SQL. Re-querying with strict date rules.")
            strict_suffix = (
                "\n\nSTRICT FIX: You MUST call calculate_date_range for ANY temporal expression and use ONLY the returned dates. "
                "Never add/subtract integers to dates (e.g., CURRENT_DATE + 7). If referencing CURRENT_DATE, use INTERVAL syntax only, "
                "but prefer explicit dates from the tool. Return ONLY SQL."
            )
            strict_prompt = f"{prompt}\n{strict_suffix}"
            sql_query2 = llm_handler.query_llm('', strict_prompt, tools=[CALCULATE_DATE_RANGE_TOOL])
            if sql_query2:
                sanitized_query2 = sql_query2.replace("```sql", "").replace("```", "").strip()
                select_index2 = sanitized_query2.find("SELECT")
                if select_index2 != -1:
                    sanitized_query2 = sanitized_query2[select_index2:]
                sanitized_query2 = sanitized_query2.split(";")[0]
                if not _sql_has_illegal_date_arithmetic(sanitized_query2):
                    sanitized_query = sanitized_query2
                    logging.info("Preflight: Successfully regenerated SQL without illegal date arithmetic.")
                else:
                    logging.warning("Preflight: Regenerated SQL still contains illegal date arithmetic; proceeding with interpretation but execution may fail.")

        # Ensure we actually have a SELECT statement; otherwise re-query with stricter instructions
        if sanitized_query and not sanitized_query.upper().startswith("SELECT"):
            logging.warning("Preflight: No valid SELECT found. Re-querying with explicit SQL-only instruction.")
            sql_only_suffix = (
                "\n\nSTRICT FIX: Return ONLY a raw SQL SELECT statement (no tool calls, no JSON, no explanations). "
                "Call calculate_date_range internally and embed the dates directly in WHERE clauses."
            )
            sql_only_prompt = f"{prompt}\n{sql_only_suffix}"
            sql_query3 = llm_handler.query_llm('', sql_only_prompt, tools=[CALCULATE_DATE_RANGE_TOOL])
            if sql_query3:
                s3 = sql_query3.replace("```sql", "").replace("```", "").strip()
                si3 = s3.upper().find("SELECT")
                if si3 != -1:
                    s3 = s3[si3:]
                s3 = s3.split(";")[0]
                if s3.upper().startswith("SELECT") and not _sql_has_illegal_date_arithmetic(s3):
                    sanitized_query = s3
                    logging.info("Preflight: Successfully regenerated a valid SELECT SQL.")

    # Generate natural language interpretation and always confirm intent
    try:
        if use_contextual_prompt:
            query_for_interpretation = combined_query
        else:
            query_for_interpretation = user_input

        interpretation = generate_interpretation(query_for_interpretation, config)
        logging.info(f"Generated interpretation: {interpretation}")

        if use_contextual_prompt and session_token:
            try:
                conversation_manager.store_pending_query(
                    conversation_id=conversation_id,
                    user_input=user_input,
                    combined_query=query_for_interpretation,
                    interpretation=interpretation,
                    sql_query=sanitized_query if (sanitized_query and sanitized_query.upper().startswith('SELECT')) else None
                )
                search_context = {
                    "last_search_query": context.get('last_search_query', combined_query),
                    "concatenation_count": context.get('concatenation_count', 1)
                }
                conversation_manager.update_conversation_context(conversation_id, search_context)
            except Exception as e:
                logging.error(f"Error storing pending query: {e}")
                raise HTTPException(status_code=500, detail=f"Error storing query for confirmation: {e}")

        return {
            "interpretation": interpretation,
            "confirmation_required": True,
            "conversation_id": conversation_id if use_contextual_prompt else None,
            "intent": intent if use_contextual_prompt else None,
            "message": f"{interpretation}\n\nIf that is correct, please confirm using the buttons below:",
            "options": ["yes", "clarify", "no"],
            "sql_query": sanitized_query if (sanitized_query and sanitized_query.upper().startswith('SELECT')) else None
        }

    except Exception as e:
        logging.error(f"Error generating interpretation: {e}")
        return {
            "message": f"I understand you want to search for: {user_input}. Please confirm if this is correct.",
            "confirmation_required": True,
            "simple_confirmation": True
        }
