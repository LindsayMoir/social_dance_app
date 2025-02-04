from dotenv import load_dotenv
import logging
import streamlit as st
import openai
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

st.title("Social Dance Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("I can answer questions about what dance events are happening in Greater Victoria now and in the future. What would you like to know?")

if st.button("Send"):
    if user_input.strip():
        # 1) Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # 2) Read the prompt from sql_prompt.txt and append the user input
        prompt_path = os.path.join(base_dir, config['prompts']['sql'])
        with open(prompt_path, "r") as file:
            base_prompt = file.read()

        prompt = f"{base_prompt}\n\nUser Question:\n\n\"{user_input}\""
        print(f"Prompt: {prompt}")

        sql_query = llm_handler.query_llm(prompt)
        print(f"Raw SQL Query: \n{sql_query}")

        # 3) Sanitize the SQL Query
        # Remove triple backticks, extra whitespace, or markdown syntax
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        print(f"Sanitized SQL Query: \n{sql_query}")

        try:
            # 4) Sanitize the SQL Query
            # Remove triple backticks, extra whitespace, or markdown syntax
            sanitized_sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            print(f"Sanitized SQL Query: \n{sanitized_sql_query}")

            # 5) Execute the SQL with SQLAlchemy
            with engine.connect() as conn:
                result = conn.execute(text(sanitized_sql_query))  # Use `text()` for proper handling
                rows = result.fetchall()

            # Convert rows to a DataFrame
            df = pd.DataFrame(rows, columns=result.keys())

            # Display DataFrame with increased width and height
            st.dataframe(df, height=600, width=1200)

            # Show the SQL query
            st.markdown(f"**SQL Query**:\n```\n{sanitized_sql_query}\n```")

            # Display the DataFrame in Streamlit
            st.dataframe(df)

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
