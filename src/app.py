# app.py
"""
This script sets up a Streamlit application for a social dance chatbot that answers questions about dance events in Greater Victoria.
It uses OpenAI's language model to generate SQL queries based on user input and retrieves data from a database.

Modules:
- dotenv: Loads environment variables from a .env file.
- logging: Provides logging capabilities.
- streamlit: Creates the web application interface.
- openai: Interacts with OpenAI's language model.
- pandas: Handles data manipulation and analysis.
- os: Interacts with the operating system.
- sqlalchemy: Manages database connections and queries.
- yaml: Parses YAML configuration files.

Classes:
- LLMHandler: Custom class to handle interactions with the language model.

Functions:
- load_dotenv: Loads environment variables from a .env file.
- create_engine: Creates a SQLAlchemy engine for database connections.
- text: Prepares SQL queries for execution.

Streamlit Components:
- st.set_page_config: Configures the Streamlit page layout.
- st.title: Sets the title of the Streamlit app.
- st.text_input: Creates a text input box for user queries.
- st.button: Creates a button to submit user queries.
- st.dataframe: Displays data in a table format.
- st.markdown: Renders Markdown text.

Workflow:
1. Load environment variables from a .env file.
2. Load configuration settings from a YAML file.
3. Initialize the LLMHandler with the configuration path.
4. Create a SQLAlchemy engine using the database URL from environment variables.
5. Set up the Streamlit interface with a title, text input, and button.
6. Handle user input by generating and executing SQL queries using the language model.
7. Display query results and the conversation history in the Streamlit app.
"""
from dotenv import load_dotenv
import logging
import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine, text
import yaml

from llm import LLMHandler

# Set Streamlit page configuration for wide layout
st.set_page_config(layout="wide")

# Calculate the path to config.yaml
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')

# Pass the config path to the LLMHandler
llm_handler = LLMHandler(config_path=config_path)

# 1) Load .env to populate environment variables
load_dotenv()  # looks for .env in the current directory

# 2) Load YAML config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

print("Starting the application...")

# set RENDER_EXTERNAL_DB_URL
DATABASE_URL = os.getenv("RENDER_EXTERNAL_DB_URL")

# 4) Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

st.markdown("# Let's Dance! 🕺💃")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Read the chatbot instructions from the specified file
instructions_path = os.path.join(base_dir, config['prompts']['chatbot_instructions'])
with open(instructions_path, "r") as file:
    chatbot_instructions = file.read()

# Display the instructions with Markdown
st.markdown(chatbot_instructions)

user_input = st.text_input("Ask a question, then click Send:")

if st.button("Send"):
    if user_input.strip():
        # 1) Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # 2) Read the prompt from sql_prompt.txt and append the user input
        prompt_path = os.path.join(base_dir, config['prompts']['sql'])
        with open(prompt_path, "r") as file:
            base_prompt = file.read()

        # Read the chatbot instructions from the specified file
        instructions_path = os.path.join(base_dir, config['prompts']['chatbot_instructions'])
        with open(instructions_path, "r") as file:
            chatbot_instructions = file.read()

        prompt = f"{base_prompt}\n\nUser Question:\n\n\"{user_input}\""
        print(f"Prompt: {prompt}")

        sql_query = llm_handler.query_llm(prompt)
        print(f"Raw SQL Query: \n{sql_query}")

        # 3) Sanitize the SQL Query
        # Remove triple backticks, extra whitespace, or markdown syntax
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        # Remove everything before the SELECT keyword
        sql_query = sql_query[sql_query.find("SELECT"):]

        # Remove eerything after ';' character
        sql_query = sql_query.split(";")[0]
        
        print(f"Sanitized SQL Query: \n{sql_query}")

        try:
            # Execute the SQL query
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))  # Use `text()` for proper handling
                rows = result.fetchall()

            # Convert rows to a DataFrame
            df = pd.DataFrame(rows, columns=result.keys())

            # If 'url' column exists, configure it to be a clickable link
            if "url" in df.columns:
                st.dataframe(
                    df,
                    column_config={
                        "url": st.column_config.LinkColumn("Event Link"),
                    },
                    height=600,
                    width=1800
                )
            else:
                st.dataframe(df, height=600, width=1800)

            # Show the SQL query
            st.markdown(f"**SQL Query**:\n```\n{sql_query}\n```")

            st.session_state["messages"].append({"role": "assistant", "content": "Here are the results from your query."})

        except Exception as db_err:
            error_message = f"Database Error: {db_err}"
            logging.error(error_message)
            st.session_state["messages"].append({"role": "assistant", "content": error_message})

    else:
        st.write("Please enter a message")
else:
    st.write("Please enter a message")

# 4) Render the conversation
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"**User**: {message['content']}")
    else:
        st.markdown(f"**Assistant**:\n{message['content']}")
