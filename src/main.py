# main.py
"""
This module sets up a FastAPI application for a Social Dance Chatbot API. It includes the following functionalities:
- Loads environment variables from a .env file.
- Loads configuration from a YAML file.
- Initializes a custom LLMHandler for processing language model queries.
- Connects to a database using SQLAlchemy.
- Defines a FastAPI application with endpoints for querying the chatbot.
Classes:
    QueryRequest(BaseModel): A Pydantic model for the incoming query requests.
Functions:
    read_root(): A simple root endpoint that returns a welcome message.
    process_query(request: QueryRequest): A POST endpoint that processes the user's query, constructs a prompt for the language model, executes the resulting SQL query, and returns the results.
Configuration:
    - Environment variables are loaded from a .env file.
    - YAML configuration is loaded from a file specified by the config_path.
    - Database URL is retrieved from the environment variable RENDER_EXTERNAL_DB_URL.
Endpoints:
    - GET /: Returns a welcome message.
    - POST /query: Processes the user's query and returns the results from the database.
Error Handling:
    - Logs errors related to loading configuration, reading prompt files, and database operations.
    - Raises HTTP exceptions with appropriate status codes and error messages for client and server errors.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
import yaml
import logging

import os
print("Current working directory:", os.getcwd())

# Import your custom LLMHandler
from llm import LLMHandler

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Calculate base directory assuming a similar structure to your original project
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')

# Load YAML configuration
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    raise e

# Initialize the LLMHandler using the configuration path
llm_handler = LLMHandler(config_path=config_path)

# Get the database URL from the environment variable
DATABASE_URL = os.getenv("RENDER_EXTERNAL_DB_URL")
if not DATABASE_URL:
    raise ValueError("The environment variable RENDER_EXTERNAL_DB_URL is not set.")

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create the FastAPI app instance
app = FastAPI(title="Social Dance Chatbot API")

# Define a Pydantic model for the incoming query requests
class QueryRequest(BaseModel):
    user_input: str

# A simple root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Social Dance Chatbot API!"}

# POST endpoint to process the user's query
@app.post("/query")
def process_query(request: QueryRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is empty.")
    
    # Load the SQL prompt from file specified in the YAML config
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
    
    # Query the language model to get a raw SQL query
    sql_query = llm_handler.query_llm(prompt)
    logging.info(f"Raw SQL Query: {sql_query}")
    
    # Sanitize the SQL query (removing markdown syntax if present)
    sanitized_query = sql_query.replace("```sql", "").replace("```", "").strip()
    logging.info(f"Sanitized SQL Query: {sanitized_query}")
    
    try:
        # Execute the SQL query using SQLAlchemy
        with engine.connect() as conn:
            result = conn.execute(text(sanitized_query))
            rows = result.fetchall()
        
        # Convert result rows to a list of dictionaries
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in rows]
    except Exception as db_err:
        error_message = f"Database Error: {db_err}"
        logging.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
    
    # Return the sanitized SQL query and the data as JSON
    return {
        "sql_query": sanitized_query,
        "data": data,
        "message": "Here are the results from your query."
    }
