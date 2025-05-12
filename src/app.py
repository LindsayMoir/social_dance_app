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
config_path = os.path.join(base_dir, 'config', 'config.yaml')

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
logging.info("app.py: config loaded.")

FASTAPI_API_URL = os.getenv("FASTAPI_API_URL")
if not FASTAPI_API_URL:
    raise ValueError("FASTAPI_API_URL not set.")

# ─── Load FAQ & Example Prompts ────────────────────────────────────────────────

faq_path = os.path.join(base_dir, 'config', 'faq.yaml')
examples_path = os.path.join(base_dir, 'config', 'examples.yaml')

with open(faq_path) as f:
    faq = yaml.safe_load(f)             # { "how to find salsa": "Try asking …", ... }

with open(examples_path) as f:
    examples = yaml.safe_load(f)        # [ "Where can I dance salsa tonight?", ... ]

# ─── Streamlit Layout ─────────────────────────────────────────────────────────

st.set_page_config(layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# rotate placeholder each run
placeholder = examples[st.session_state.get("example_idx", 0)]
st.session_state["example_idx"] = (st.session_state.get("example_idx", 0) + 1) % len(examples)

user_input = st.text_area(
    "Ask a question, then click Send:",
    height=100,
    placeholder=placeholder
)

send = st.button("Send")
if send:
    if user_input.strip():
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.spinner("Looking up dance events…"):
            try:
                resp = requests.post(FASTAPI_API_URL, json={"user_input": user_input})
                resp.raise_for_status()
                data = resp.json()
                events = data.get("data", [])

                if events:
                    # display events as before…
                    for ev in events:
                        st.markdown(f"**{ev.get('event_name','No Name')}**")
                        # …
                else:
                    # No events → show FAQ suggestions
                    keys = list(faq.keys())
                    matches = difflib.get_close_matches(user_input, keys, n=3, cutoff=0.3)
                    if matches:
                        with st.expander("Need help? Try these questions…"):
                            for k in matches:
                                st.markdown(f"- **{k}**: {faq[k]}")
                    # also prompt follow-up
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": "I’m not finding any events. Could you tell me which dance style or date range you’d like?"
                    })

                # show SQL only if testing.sql is true
                if config.get('testing',{}).get('sql'):
                    with st.expander("Show debug SQL"):
                        st.code(data.get('sql_query','<none>'))

            except Exception as e:
                st.session_state["messages"].append({
                    "role":"assistant",
                    "content":"Sorry, something went wrong. Try rephrasing your question or check back in a moment."
                })
                logging.error(f"app.py: Error - {e}")
    else:
        st.warning("Please enter a question before sending.")

# ─── Render chat history ────────────────────────────────────────────────────────

for msg in reversed(st.session_state["messages"]):
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(msg["content"])
