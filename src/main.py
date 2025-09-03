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

# Set up basic logging
logging.basicConfig(level=logging.INFO)
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

# Initialize the FastAPI app
app = FastAPI(title="Social Dance Chatbot API")

# Define the query request models
class QueryRequest(BaseModel):
    user_input: str
    session_token: str = None  # Optional session token for conversation context

@app.get("/")
def read_root():
    return {"message": "Welcome to the Social Dance Chatbot API!"}

@app.post("/query")
def process_query(request: QueryRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is empty.")
    
    # Handle session-based conversation context
    session_token = request.session_token
    use_contextual_prompt = session_token is not None
    
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
            
            # Use contextual prompt template
            prompt_file_path = os.path.join(base_dir, 'prompts', 'contextual_sql_prompt.txt')
            with open(prompt_file_path, "r") as file:
                base_prompt = file.read()
            
            # Format conversation history for prompt
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in recent_messages[-3:]  # Last 3 messages for context
            ])
            
            # Construct contextual prompt
            prompt = base_prompt.format(
                context_info=str(context),
                conversation_history=history_text,
                intent=intent,
                entities=str(entities)
            )
            prompt += f"\n\nCurrent User Question: \"{user_input}\""
            
        except Exception as e:
            logging.error(f"Error with contextual conversation: {e}")
            # Fall back to non-contextual mode
            use_contextual_prompt = False
    
    if not use_contextual_prompt:
        # Use original SQL prompt template (backward compatibility)
        prompt_config = config['prompts']['sql']
        if isinstance(prompt_config, dict):
            prompt_file_path = os.path.join(base_dir, prompt_config['file'])
        else:
            prompt_file_path = os.path.join(base_dir, prompt_config)
        
        try:
            with open(prompt_file_path, "r") as file:
                base_prompt = file.read()
        except Exception as e:
            logging.error(f"Error reading prompt file: {e}")
            raise HTTPException(status_code=500, detail="Error reading prompt file.")
        
        prompt = f"{base_prompt}\n\nUser Question:\n\n\"{user_input}\""
    
    logging.info(f"Constructed Prompt: {prompt}")
    
    # Query the language model for a raw SQL query
    sql_query = llm_handler.query_llm('', prompt)
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
        
        try:
            # Execute the SQL query using DatabaseHandler pattern
            rows = db_handler.execute_query(sanitized_query)
            if not rows:
                data = []
            else:
                # Get column names from the query result
                # For SELECT queries, we need to parse the columns from the SQL
                columns = [
                    'event_name', 'event_type', 'dance_style', 'day_of_week',
                    'start_date', 'end_date', 'start_time', 'end_time', 'source',
                    'url', 'price', 'description', 'location'
                ]
                data = [dict(zip(columns, row)) for row in rows]
        except Exception as db_err:
            error_message = f"Database Error: {db_err}"
            logging.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
        
        # Update conversation context if using sessions
        if use_contextual_prompt and session_token:
            try:
                # Add assistant message to conversation
                conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=f"Found {len(data)} events",
                    sql_query=sanitized_query,
                    result_count=len(data)
                )
                
                # Update context with last search criteria for future refinements
                search_context = {
                    "last_search_criteria": entities,
                    "last_query": sanitized_query,
                    "last_result_count": len(data)
                }
                conversation_manager.update_conversation_context(conversation_id, search_context)
                
            except Exception as e:
                logging.error(f"Error updating conversation context: {e}")
        
        return {
            "sql_query": sanitized_query,
            "data": data,
            "message": "Here are the results from your query.",
            "conversation_id": conversation_id if use_contextual_prompt else None,
            "intent": intent if use_contextual_prompt else None
        }
    else:
        logging.warning("LLM did not return a valid SQL query.")
        return {
            "message": "The language model could not generate a valid SQL query from your input. Please try rephrasing your question."
        }
