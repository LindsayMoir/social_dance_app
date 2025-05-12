# app.py

import streamlit as st
import requests
import os
import yaml
import logging
from dotenv import load_dotenv
import difflib

# ─── Logging & config ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logging.info("app.py: Streamlit app starting...")

load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, "config", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
logging.info("app.py: config loaded.")

FASTAPI_API_URL = os.getenv("FASTAPI_API_URL")
if not FASTAPI_API_URL:
    raise ValueError("FASTAPI_API_URL not set.")

st.set_page_config(layout="wide")

# ─── Load & show your chatbot instructions ────────────────────────────────────
instructions_path = os.path.join(
    base_dir,
    config["prompts"]["chatbot_instructions"]
)
with open(instructions_path, "r") as file:
    chatbot_instructions = file.read()
st.markdown(chatbot_instructions)

# ─── Load FAQ & Example Prompts ────────────────────────────────────────────────
faq_path = os.path.join(base_dir, "config", "faq.yaml")
examples_path = os.path.join(base_dir, "config", "examples.yaml")

with open(faq_path, "r") as f:
    faq = yaml.safe_load(f)

with open(examples_path, "r") as f:
    examples = yaml.safe_load(f)

# ─── Session State Setup ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "example_idx" not in st.session_state:
    st.session_state["example_idx"] = 0

# rotate placeholder
placeholder = examples[st.session_state["example_idx"]]
st.session_state["example_idx"] = (
    st.session_state["example_idx"] + 1
) % len(examples)

# ─── Chat Form ────────────────────────────────────────────────────────────────
with st.form(key="chat_form"):
    # We bind the text area to session_state["chat_input"]
    st.text_area(
        "Ask a question, then click Send:",
        height=100,
        key="chat_input",
        placeholder=placeholder
    )
    submit = st.form_submit_button("Send")

# Now handle the click _outside_ the with‐block
if submit:
    chat_text = st.session_state.get("chat_input", "")
    if not chat_text.strip():
        st.warning("Please enter a question before sending.")
    else:
        # (optional) clear the box immediately
        st.session_state["chat_input"] = ""

        # record the user message
        st.session_state["messages"].append({
            "role": "user",
            "content": chat_text
        })

        with st.spinner("Looking up dance events…"):
            try:
                resp = requests.post(
                    FASTAPI_API_URL,
                    json={"user_input": chat_text}
                )
                resp.raise_for_status()
                data = resp.json()
                events = data.get("data", [])

                if events:
                    # … your event‐display logic …
                    for ev in events:
                        st.markdown(f"**{ev.get('event_name','No Name')}**")
                        # etc.

                else:
                    # no results → FAQ fallback
                    keys    = list(faq.keys())
                    matches = difflib.get_close_matches(chat_text, keys, n=3, cutoff=0.3)
                    if matches:
                        with st.expander("Need help? Try these…"):
                            for k in matches:
                                st.markdown(f"- **{k}**: {faq[k]}")

                    # follow‑up question in chat history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": (
                            "I’m not finding any events. "
                            "Could you tell me which dance style or date range you’d like?"
                        )
                    })

                # debug SQL if allowed
                if config.get("testing", {}).get("sql"):
                    with st.expander("Show debug SQL"):
                        st.code(data.get("sql_query", "<none>"))

            except Exception as e:
                logging.error(f"app.py: Error - {e}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": (
                        "Sorry, something went wrong. "
                        "Try rephrasing your question or try again later."
                    )
                })

# ─── Render chat history (newest first) ───────────────────────────────────────
for msg in reversed(st.session_state["messages"]):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(msg["content"], unsafe_allow_html=True)
