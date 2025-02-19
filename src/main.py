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

# Check and see if we are running local or remote on Render
if 'render/project' in sys.path[-1]:
    local = False
    print("Running on Render...")
else:
    local = True
    print("Running locally...")

# Load environment variables
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)

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

# Initialize the LLMHandler
llm_handler = LLMHandler(config_path=config_path)

# Get the DATABASE_URL from environment variables
if local:
    DATABASE_URL = os.getenv("DATABASE_CONNECTION_STRING")
else:
    DATABASE_URL = os.getenv("RENDER_EXTERNAL_DB_URL")

if DATABASE_URL:
    logging.info(f"DATABASE_URL / database connection string is set")
else:
    raise ValueError("DATABASE_URL / database connections string is not set.")

print(f"DATABASE_URL: {DATABASE_URL}")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
logging.info("main.py: SQLAlchemy engine created.")

# Initialize the FastAPI app
app = FastAPI(title="Social Dance Chatbot API")

# Define the query request model
class QueryRequest(BaseModel):
    user_input: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Social Dance Chatbot API!"}

@app.post("/query")
def process_query(request: QueryRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is empty.")
    
    # Load the SQL prompt template from the YAML config
    prompt_file_path = os.path.join(base_dir, config['prompts']['sql'])
    try:
        with open(prompt_file_path, "r") as file:
            base_prompt = file.read()
    except Exception as e:
        logging.error(f"Error reading prompt file: {e}")
        raise HTTPException(status_code=500, detail="Error reading prompt file.")
    
    # Construct the full prompt for the language model
    prompt = f"{base_prompt}\n\nUser Question:\n\n\"{user_input}\""
    logging.info(f"Constructed Prompt: {prompt}")
    
    # Query the language model for a raw SQL query
    sql_query = llm_handler.query_llm(prompt)
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
            # Execute the SQL query using SQLAlchemy
            with engine.connect() as conn:
                result = conn.execute(text(sanitized_query))
                rows = result.fetchall()
            columns = result.keys()
            data = [dict(zip(columns, row)) for row in rows]
        except Exception as db_err:
            error_message = f"Database Error: {db_err}"
            logging.error(error_message)
            raise HTTPException(status_code=500, detail=error_message)
        
        return {
            "sql_query": sanitized_query,
            "data": data,
            "message": "Here are the results from your query."
        }
    else:
        logging.warning("LLM did not return a valid SQL query.")
        return {
            "message": "The language model could not generate a valid SQL query from your input. Please try rephrasing your question."
        }
