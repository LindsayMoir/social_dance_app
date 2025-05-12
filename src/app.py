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
base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path= os.path.join(base_dir, "config", "config.yaml")
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
faq_path      = os.path.join(base_dir, "config", "faq.yaml")
examples_path = os.path.join(base_dir, "config", "examples.yaml")

with open(faq_path,      "r") as f: faq      = yaml.safe_load(f)
with open(examples_path, "r") as f: examples = yaml.safe_load(f)

# ─── Session State Setup ──────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state["messages"]    = []
if "example_idx" not in st.session_state: st.session_state["example_idx"] = 0

# rotate placeholder
placeholder = examples[st.session_state["example_idx"]]
st.session_state["example_idx"] = (
    st.session_state["example_idx"] + 1
) % len(examples)

# ─── Error Handler (your original) ────────────────────────────────────────────
def error_handling(e, custom_message=None):
    if custom_message:
        msg = (
            f"{custom_message} "
            "I can answer questions such as:\n\n"
            "1. Where can I dance salsa tonight?\n"
            "2. Where can I dance tango this month? Only show me the social dance events.\n"
            "3. When does the West Coast Swing event on Saturdays start?\n"
            "4. etc. etc. ...")
    else:
        msg = (
            "Sorry, I did not quite catch that.\n\n"
            "I can answer questions such as:\n\n"
            "1. Where can I dance salsa tonight?\n"
            "2. Where can I dance tango this month? Only show me the social dance events.\n"
            "3. When does the West Coast Swing event on Saturdays start?\n"
            "4. etc. etc. ...")
    st.session_state["messages"].append({"role":"assistant","content":msg})
    logging.error(f"app.py: Error encountered - {e}")

# ─── Input & Button ───────────────────────────────────────────────────────────
chat_text = st.text_area(
    "Ask a question, then click Send:",
    height=100,
    key="chat_input",
    placeholder=placeholder
)

if st.button("Send"):
    user_input = st.session_state.get("chat_input", "").strip()
    if not user_input:
        st.warning("Please enter a question before sending.")
    else:
        # record the user message
        st.session_state["messages"].append({"role":"user","content":user_input})

        with st.spinner("Looking up dance events…"):
            try:
                resp = requests.post(
                    FASTAPI_API_URL,
                    json={"user_input": user_input}
                )
                resp.raise_for_status()
                data   = resp.json()
                events = data.get("data", [])

                if events:
                    for ev in events:
                        st.markdown(f"**{ev.get('event_name', 'No Name')}**")
                        for k,v in ev.items():
                            if k not in ("event_name","url"):
                                st.markdown(f"- **{k}**: {v}")
                        st.markdown("---")
                else:
                    # no results → FAQ fallback
                    keys    = list(faq.keys())
                    matches = difflib.get_close_matches(user_input, keys, n=3, cutoff=0.3)
                    if matches:
                        with st.expander("Need help? Try these…"):
                            for phrase in matches:
                                st.markdown(f"- **{phrase}**: {faq[phrase]}")
                    # append your custom “no events” error
                    error_handling("No events", custom_message="Sorry, I could not find those events in my database.")

                # debug SQL if enabled
                if config.get("testing", {}).get("sql"):
                    with st.expander("Show debug SQL"):
                        st.code(data.get("sql_query", "<none>"))

            except Exception as e:
                # This except matches the above try
                error_handling(e)

