# app.py
"""
This script sets up a Streamlit application for the Social Dance Chatbot UI.
It collects user queries and sends them to the FastAPI backend for processing.
The results (SQL query and data) are then displayed in the interface.
"""

import streamlit as st
import requests
import pandas as pd
import os
import yaml
import logging
from dotenv import load_dotenv
import sys

# Set up sys.path so that modules in src/ are accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(current_dir)
sys.path.append(parent_dir)
print("Updated sys.path:", sys.path)
print("Current working directory:", os.getcwd())

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logging.info("app.py: Streamlit app starting...")

# Load environment variables and configuration
load_dotenv()

# Calculate the base directory and config path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')

# Load YAML configuration
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
logging.info("app.py: config completed.")

# See whether you are running local or remote
if config['testing']['local']:
    # Set the FastAPI backend URL
    FASTAPI_API_URL = config['testing']['fast_api_url']
else:    
    FASTAPI_API_URL = os.getenv("FAST_API_URL")

if not FASTAPI_API_URL:
    raise ValueError("The environment variable FAST_API_URL is not set.")
    
logging.info("app.py: FAST_API_URL is: {FASTAPI_API_URL}")

st.set_page_config(layout="wide")
st.markdown("# Let's Dance! 🕺💃")

# Initialize the chat message history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Load chatbot instructions from a file specified in the YAML config
instructions_path = os.path.join(base_dir, config['prompts']['chatbot_instructions'])
with open(instructions_path, "r") as file:
    chatbot_instructions = file.read()

st.markdown(chatbot_instructions)

user_input = st.text_input("Ask a question, then click Send:")

if st.button("Send"):
    if user_input.strip():
        # Display the user's message in the chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})
        logging.info("app.py: About to send user input to FastAPI backend.")
        try:
            # Send the query to the FastAPI backend
            response = requests.post(FASTAPI_API_URL, json={"user_input": user_input})
            response.raise_for_status()
            data = response.json()
            
            # Create the DataFrame from the data
            df = pd.DataFrame(data["data"])

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

            # Display the SQL query
            st.markdown(f"**SQL Query**:\n```\n{data['sql_query']}\n```")

            # Display the history of the conversation
            st.session_state["messages"].append({"role": "assistant", "content": data["message"]})
            
        except Exception as e:
            error_message = f"Error: {e}"
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
    else:
        st.write("Please enter a message")
else:
    st.write("Please enter a message")

# Render the conversation history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"**User**: {message['content']}")
    else:
        st.markdown(f"**Assistant**: {message['content']}")
