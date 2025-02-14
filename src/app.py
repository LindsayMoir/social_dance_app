# app.py

import streamlit as st
import requests
import os
import yaml
import logging
from dotenv import load_dotenv
import sys

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

# Check whether we are running locally or on Render
FASTAPI_API_URL = os.getenv("FASTAPI_API_URL", "https://social-dance-app-ws-main.onrender.com/query")

if not FASTAPI_API_URL:
    raise ValueError("The environment variable FASTAPI_API_URL is not set.")

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

            # Get the event data from the response
            events = data["data"]
            if events:
                # Create a scrollable container to hold the events
                with st.container():
                    st.markdown("<hr>", unsafe_allow_html=True)  # Add a separator line

                    for event in events:
                        event_name = event.get('event_name', 'No Name')
                        url = event.get('url', '#')

                        # Display event name as bold and hyperlink it
                        st.markdown(f"**[ {event_name} ]({url})**")
                        
                        # Display other event details (one row per column)
                        for column_name, value in event.items():
                            if column_name != 'event_name' and column_name != 'url':
                                st.markdown(f"**{column_name}**: {value}")
                        
                        st.markdown("<hr>", unsafe_allow_html=True)  # Add a separator between events

            else:
                st.write("No events found based on your query.")

            # Display the SQL query
            st.markdown(f"**SQL Query**:\n```\n{data['sql_query']}\n```")
            
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
