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

# Get FastAPI API URL from environment variable
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

def error_handling(e, custom_message=None):
    """
    Handle errors by appending a standardized error message to the chat history.
    If custom_message is provided, it will be used as the first line of the error message.
    """
    if custom_message:
        error_message = (
            f"{custom_message}\n\n"
            "I can answer questions such as:\n\n"
            "1. Where can I dance salsa tonight?\n"
            "2. Where can I dance tango this month? Only show me the social dance events.\n"
            "3. When does the West Coast Swing event on Saturdays start?\n"
            "4. etc. etc. ..."
        )
    else:
        error_message = (
            "Sorry, I did not quite catch that.\n\n"
            "I can answer questions such as:\n\n"
            "1. Where can I dance salsa tonight?\n"
            "2. Where can I dance tango this month? Only show me the social dance events.\n"
            "3. When does the West Coast Swing event on Saturdays start?\n"
            "4. etc. etc. ..."
        )
    
    st.session_state["messages"].append({"role": "assistant", "content": error_message})
    logging.error(f"app.py: Error encountered - {e}")

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
                        
                        # Only create a hyperlink if the URL is properly formatted (starts with "http")
                        if isinstance(url, str) and url.startswith("http"):
                            st.markdown(f'<a href="{url}" target="_blank"><strong>{event_name}</strong></a>', unsafe_allow_html=True)
                        else:
                            st.markdown(f"<strong>{event_name}</strong>")

                        # Display other event details (one row per column)
                        for column_name, value in event.items():
                            if column_name not in ('event_name', 'url'):
                                st.markdown(f"**{column_name}**: {value}")
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
            else:
                # If no events are returned and a valid SQL query exists, call error_handling with a custom message BEFORE showing the SQL query.
                if data.get('sql_query'):
                    error_handling("No events returned", custom_message="Sorry, I could not find those events in my database.")
            
            # Display the SQL query (shown after error handling if triggered)
            st.markdown(f"**SQL Query**:\n```\n{data.get('sql_query', 'No SQL query provided')}\n```")
            
        except Exception as e:
            error_handling(e)
    else:
        st.write("Please enter a message")
else:
    st.write("Please enter a message")

# Render the conversation history from newest to oldest without a header
for message in reversed(st.session_state["messages"]):
    if message["role"] == "user":
        st.markdown(f"**You wrote:** {message['content']}")
    else:
        st.markdown(message["content"])
