# app.py 

import streamlit as st
import requests
import os
import yaml
import logging
import uuid
from dotenv import load_dotenv


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

# Initialize session state for conversation management
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize session token for conversation context
if "session_token" not in st.session_state:
    st.session_state["session_token"] = str(uuid.uuid4())
    logging.info(f"app.py: Generated new session token: {st.session_state['session_token']}")

# Initialize conversation context
if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = None

if "last_intent" not in st.session_state:
    st.session_state["last_intent"] = None

# Load chatbot instructions from a file specified in the YAML config
prompt_config = config['prompts']['chatbot_instructions']
if isinstance(prompt_config, dict):
    instructions_path = os.path.join(base_dir, prompt_config['file'])
else:
    # Backward compatibility with old string format
    instructions_path = os.path.join(base_dir, prompt_config)
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
            f"{custom_message} "
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


# Show conversation context info (for debugging)
if st.session_state.get("conversation_id"):
    with st.expander("🔍 Conversation Info", expanded=False):
        st.write(f"**Session Token:** {st.session_state['session_token'][:8]}...")
        st.write(f"**Conversation ID:** {st.session_state['conversation_id'][:8] if st.session_state['conversation_id'] else 'None'}...")
        st.write(f"**Last Intent:** {st.session_state.get('last_intent', 'None')}")

# Show initial input field only if no messages yet
if not st.session_state["messages"]:
    with st.container():
        st.markdown("### 💭 Ask a Question")
        
        # Get user input
        user_input = st.text_area("Ask a question, then click Send:", height=100, key="user_input")
        
        # Create columns for buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            send_button = st.button("Send", type="primary")
        
        with col2:
            if st.button("New Search"):
                # Clear conversation context for new search
                st.session_state["session_token"] = str(uuid.uuid4())
                st.session_state["conversation_id"] = None
                st.session_state["last_intent"] = None
                st.session_state["messages"] = []
                st.rerun()
        
        with col3:
            if st.button("Clear Chat"):
                st.session_state["messages"] = []
                st.rerun()
else:
    # If there are messages, we'll show the input field after the conversation
    send_button = False
    user_input = ""

# Process user input
process_input = False
if send_button and user_input.strip():
    process_input = True
    input_to_process = user_input
elif "suggested_input" in st.session_state:
    process_input = True  
    input_to_process = st.session_state["suggested_input"]
    del st.session_state["suggested_input"]

if process_input:
    # Display the user's message in the chat history
    st.session_state["messages"].append({"role": "user", "content": input_to_process})
    logging.info("app.py: About to send user input to FastAPI backend.")
    
    try:
        # Send the query to the FastAPI backend with session context
        payload = {
            "user_input": input_to_process,
            "session_token": st.session_state["session_token"]
        }
        response = requests.post(FASTAPI_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Update session state with conversation info
        if data.get("conversation_id"):
            st.session_state["conversation_id"] = data["conversation_id"]
        if data.get("intent"):
            st.session_state["last_intent"] = data["intent"]
        
        # Store the complete response data for this query in session state
        query_result = {
            "role": "assistant",
            "content": f"Found {len(data['data'])} events" if data.get('data') else "No events found",
            "events": data.get('data', []),
            "sql_query": data.get('sql_query', ''),
            "intent": data.get('intent', ''),
            "timestamp": input_to_process  # Store what user asked
        }
        st.session_state["messages"].append(query_result)

        # Just refresh the page to show the new conversation history
        st.rerun()
        
    except Exception as e:
        error_handling(e)

# If there are messages, show the full conversation thread like ChatGPT
if st.session_state["messages"]:
    st.markdown("---")
    
    # Display full conversation history in chronological order (oldest first)
    for i, message in enumerate(st.session_state["messages"]):
        if message["role"] == "user":
            st.markdown(f"### 🧑 You:")
            st.markdown(f"*{message['content']}*")
            
        elif message["role"] == "assistant":
            st.markdown(f"### 🤖 Assistant:")
            
            # Display events if they exist in this message
            events = message.get("events", [])
            if events:
                intent_info = f" (Intent: {message.get('intent', 'search')})" if message.get('intent') else ""
                st.success(f"Found {len(events)} events{intent_info}")
                
                # Add follow-up suggestion buttons only for the most recent assistant message
                if i == len(st.session_state["messages"]) - 1 and message.get('intent') == 'search' and len(events) > 0:
                    st.write("💡 **Try these follow-up questions:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Show different styles", key=f"styles_{i}"):
                            st.session_state["suggested_input"] = "What other dance styles are available?"
                            st.rerun()
                    with col2:
                        if st.button(f"Show classes instead", key=f"classes_{i}"):
                            st.session_state["suggested_input"] = "Any classes or workshops?"
                            st.rerun()
                    with col3:
                        if st.button(f"Show tomorrow", key=f"tomorrow_{i}"):
                            st.session_state["suggested_input"] = "What about tomorrow?"
                            st.rerun()
                
                # Display events in expandable cards
                with st.container():
                    for j, event in enumerate(events):
                        event_name = event.get('event_name', 'No Name')
                        url = event.get('url', '#')
                        
                        # Create expandable event cards (expand first 2 for latest query only)
                        expanded = (i == len(st.session_state["messages"]) - 1 and j < 2)
                        with st.expander(f"🎵 {event_name}", expanded=expanded):
                            # Only create a hyperlink if the URL is properly formatted
                            if isinstance(url, str) and url.startswith("http"):
                                st.markdown(f'🔗 [**Event Link**]({url})')
                            
                            # Display event details in organized columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Dance Style:** {event.get('dance_style', 'N/A')}")
                                st.write(f"**Event Type:** {event.get('event_type', 'N/A')}")
                                st.write(f"**Day:** {event.get('day_of_week', 'N/A')}")
                                st.write(f"**Date:** {event.get('start_date', 'N/A')}")
                            
                            with col2:
                                st.write(f"**Time:** {event.get('start_time', 'N/A')} - {event.get('end_time', 'N/A')}")
                                st.write(f"**Price:** {event.get('price', 'N/A')}")
                                st.write(f"**Source:** {event.get('source', 'N/A')}")
                            
                            st.write(f"**Location:** {event.get('location', 'N/A')}")
                            
                            if event.get('description'):
                                st.write(f"**Description:** {event.get('description')}")
            else:
                # Handle case where no events were found
                st.info("No events found for this query.")
                
                # Show refinement suggestions for the most recent query with no results
                if i == len(st.session_state["messages"]) - 1:
                    st.write("💡 **Try refining your search:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Try different dance style", key=f"diff_style_{i}"):
                            st.session_state["suggested_input"] = "Show me any dance events"
                            st.rerun()
                    with col2:
                        if st.button("Expand time range", key=f"expand_time_{i}"):
                            st.session_state["suggested_input"] = "Show me events this week"
                            st.rerun()
            
            # Show SQL query for the most recent query only
            if i == len(st.session_state["messages"]) - 1 and message.get("sql_query"):
                with st.expander("🔍 View Generated SQL Query", expanded=False):
                    st.code(message.get('sql_query', 'No SQL query provided'), language='sql')
        
        # Add separator between conversation turns
        if i < len(st.session_state["messages"]) - 1:
            st.markdown("---")
    
    # Show the input field at the bottom for continued conversation
    with st.container():
        st.markdown("### 💭 Continue the Conversation")
        
        # Get user input for follow-up
        followup_input = st.text_area("Ask another question or refine your search:", height=100, key="followup_input")
        
        # Create columns for buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("Send", type="primary", key="followup_send"):
                if followup_input.strip():
                    # Add to messages and process
                    st.session_state["messages"].append({"role": "user", "content": followup_input})
                    
                    # Process the follow-up input
                    try:
                        payload = {
                            "user_input": followup_input,
                            "session_token": st.session_state["session_token"]
                        }
                        response = requests.post(FASTAPI_API_URL, json=payload)
                        response.raise_for_status()
                        data = response.json()
                        
                        # Update session state
                        if data.get("conversation_id"):
                            st.session_state["conversation_id"] = data["conversation_id"]
                        if data.get("intent"):
                            st.session_state["last_intent"] = data["intent"]
                        
                        st.rerun()  # Refresh to show new results
                        
                    except Exception as e:
                        error_handling(e)
                        st.rerun()
        
        with col2:
            if st.button("New Search", key="followup_new"):
                # Clear conversation context for new search
                st.session_state["session_token"] = str(uuid.uuid4())
                st.session_state["conversation_id"] = None
                st.session_state["last_intent"] = None
                st.session_state["messages"] = []
                st.rerun()
        
        with col3:
            if st.button("Clear Chat", key="followup_clear"):
                st.session_state["messages"] = []
                st.rerun()
else:
    st.info("💡 Start a conversation by asking a question above!")
