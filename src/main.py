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

def generate_interpretation(user_query: str, config: dict) -> str:
    """
    Generate a natural language interpretation of the user's search intent.
    
    Args:
        user_query: The user's input query
        config: Configuration dictionary containing location settings
        
    Returns:
        str: Natural language interpretation of the search intent
    """
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

    if not interpretation:
        return f"My understanding is that you want to search for dance events related to: {user_query}"

    return interpretation.strip()

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
        # Execute the pending SQL query
        try:
            sanitized_query = pending_query["sql_query"]
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
                sql_query=sanitized_query,
                result_count=len(data)
            )
            
            conversation_manager.clear_pending_query(conversation_id)
            
            return {
                "sql_query": sanitized_query,
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
        # Handle clarification - treat as new query with clarification text
        if not clarification:
            raise HTTPException(status_code=400, detail="Clarification text is required when selecting 'clarify' option.")
        
        # Clear pending query and process clarification as new query
        conversation_manager.clear_pending_query(conversation_id)
        
        # Create a new QueryRequest and process it
        clarification_request = QueryRequest(user_input=clarification, session_token=session_token)
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

    if sql_query:
        # Sanitize the SQL query by removing markdown formatting
        sanitized_query = sql_query.replace("```sql", "").replace("```", "").strip()
        # Optionally, trim everything before the first "SELECT" and after the first ';'
        select_index = sanitized_query.find("SELECT")
        if (select_index != -1):
            sanitized_query = sanitized_query[select_index:]
        sanitized_query = sanitized_query.split(";")[0]
        logging.info(f"Sanitized SQL Query: {sanitized_query}")
        
        # Generate natural language interpretation instead of executing immediately
        try:
            # Get the combined query for interpretation
            if use_contextual_prompt:
                query_for_interpretation = combined_query
            else:
                query_for_interpretation = user_input
            
            interpretation = generate_interpretation(query_for_interpretation, config)
            logging.info(f"Generated interpretation: {interpretation}")
            
            # Store pending query for confirmation if using sessions
            if use_contextual_prompt and session_token:
                try:
                    conversation_manager.store_pending_query(
                        conversation_id=conversation_id,
                        user_input=user_input,
                        combined_query=query_for_interpretation,
                        interpretation=interpretation,
                        sql_query=sanitized_query
                    )
                    
                    # Update context with concatenation info for next refinement
                    search_context = {
                        "last_search_query": context.get('last_search_query', combined_query),
                        "concatenation_count": context.get('concatenation_count', 1)
                    }
                    conversation_manager.update_conversation_context(conversation_id, search_context)
                    
                except Exception as e:
                    logging.error(f"Error storing pending query: {e}")
                    raise HTTPException(status_code=500, detail=f"Error storing query for confirmation: {e}")
            
            # Return interpretation with confirmation options
            return {
                "interpretation": interpretation,
                "confirmation_required": True,
                "conversation_id": conversation_id if use_contextual_prompt else None,
                "intent": intent if use_contextual_prompt else None,
                "message": f"{interpretation}\n\nIf that is correct, please confirm using the buttons below:",
                "options": ["yes", "clarify", "no"]
            }
            
        except Exception as e:
            logging.error(f"Error generating interpretation: {e}")
            # Fallback for non-contextual queries - just return a simple message
            return {
                "message": f"I understand you want to search for: {user_input}. Please confirm if this is correct.",
                "confirmation_required": True,
                "simple_confirmation": True
            }
    else:
        logging.warning("LLM did not return a valid SQL query.")
        return {
            "message": "The language model could not generate a valid SQL query from your input. Please try rephrasing your question."
        }
