# src/app.py

import streamlit as st
import requests
import os
import yaml
import logging
from dotenv import load_dotenv

# ─── Initialize session_state keys up‐front ────────────────────────────────────
initial_state = {
    "stage": 1,
    "used_before": "",
    "event_type": "Social dance",    # canonical
    "dance_styles": [],
    "time_frame": "",
    "final_question": None,
    "messages": []
}
for key, default in initial_state.items():
    if key not in st.session_state:
        st.session_state[key] = default
# ────────────────────────────────────────────────────────────────────────────────

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info("app.py: Streamlit app starting…")

# Load environment variables & configuration
load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_dir, "config", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

FASTAPI_API_URL = os.getenv("FASTAPI_API_URL", "http://localhost:8000/query")
if not FASTAPI_API_URL:
    raise ValueError("FASTAPI_API_URL not set")

st.set_page_config(layout="wide")

# Load chatbot instructions
instructions_path = os.path.join(base_dir, config["prompts"]["chatbot_instructions"])
with open(instructions_path, "r") as file:
    st.markdown(file.read())


def error_handling(e, custom_message=None):
    """Append a standardized error message to chat history."""
    msg = custom_message or "Sorry, I did not catch that."
    msg += "\n\nTry questions like:\n" \
           "• Where can I dance salsa tonight?\n" \
           "• When do the weekend WCS socials start?\n" \
           "• etc."
    st.session_state.messages.append({"role": "assistant", "content": msg})
    logging.error(f"app.py: Error - {e}")


# ─── Multi-step Q&A flow ───────────────────────────────────────────────────────

# Stage 1: Have you used DanceScoop before?
if st.session_state.stage == 1:
    st.write("**Q1. Have you used DanceScoop before?**")
    used_before_input = st.text_area(
        "Reply `Yes` + your full query, or `No` to go step-by-step:",
        value=st.session_state.used_before,
        height=200,
        key="used_before_input"
    )
    if st.button("Submit Q1"):
        ans = used_before_input.strip()
        st.session_state.used_before = ans
        if ans.lower().startswith("yes"):
            st.session_state.final_question = ans[len("yes"):].strip()
            st.session_state.stage = "backend"
        else:
            st.session_state.stage = 2

# Stage 2: Pick event type
elif st.session_state.stage == 2:
    st.write("**Q2. Pick an event type:**")
    st.write(
        "• Social dance\n"
        "• Dance classes\n"
        "• Dance workshops\n"
        "• Live music venues\n"
        "• Other"
    )
    event_type_input = st.selectbox(
        "Your choice:",
        ["Social dance", "Dance classes", "Dance workshops", "Live music venues", "Other"],
        index=["Social dance", "Dance classes", "Dance workshops", "Live music venues", "Other"]
              .index(st.session_state.event_type),
        key="event_type_input"
    )
    if st.button("Next – Event Type"):
        st.session_state.event_type = event_type_input
        st.session_state.stage = 3

# Stage 3: Choose dance styles
elif st.session_state.stage == 3:
    st.write("**Q3. Choose one or more dance styles:**")
    dance_styles_input = st.multiselect(
        "Select dance styles:",
        [
            "cha cha", "foxtrot", "quickstep", "rumba", "waltz",
            "2 step", "double shuffle", "line dancing", "nite club 2",
            "bachata", "cumbia", "kizomba", "merengue", "rueda", "salsa",
            "balboa", "lindy", "swing", "wcs", "west coast swing"
        ],
        default=st.session_state.dance_styles,
        key="dance_styles_input"
    )
    if st.button("Next – Dance Styles"):
        st.session_state.dance_styles = dance_styles_input
        st.session_state.stage = 4

# Stage 4: Time frame
elif st.session_state.stage == 4:
    styles = ", ".join(st.session_state.dance_styles)
    st.write(f"**Q4. When do you want to dance {styles}?**")
    time_frame_input = st.text_area(
        "e.g. This week, Tomorrow evening, June 5 to June 10",
        value=st.session_state.time_frame,
        height=200,
        key="time_frame_input"
    )
    if st.button("Next – Time Frame"):
        st.session_state.time_frame = time_frame_input.strip()
        st.session_state.stage = 5

# Stage 5: Confirmation
elif st.session_state.stage == 5:
    question = (
        f"Where can I dance {', '.join(st.session_state.dance_styles)} "
        f"{st.session_state.time_frame}? "
        f"Please restrict your answer to {st.session_state.event_type.lower()} only."
    )
    st.write(f"**Q5. Please confirm:**\n\n\"{question}\"")
    col1, col2 = st.columns(2)
    if col1.button("Yes, submit"):
        st.session_state.final_question = question
        st.session_state.stage = "backend"
    if col2.button("No, let me correct"):
        st.session_state.stage = 1

# Stage 'backend': call the API
elif st.session_state.stage == "backend":
    st.session_state.messages.append({
        "role": "user", "content": st.session_state.final_question
    })
    st.write("**Sending your query to the backend…**")
    try:
        resp = requests.post(
            FASTAPI_API_URL,
            json={"user_input": st.session_state.final_question}
        )
        resp.raise_for_status()
        data = resp.json()
        events = data.get("data", [])

        if events:
            for ev in events:
                nm, url = ev.get("event_name", "No Name"), ev.get("url", "#")
                if isinstance(url, str) and url.startswith("http"):
                    st.markdown(
                        f'<a href="{url}" target="_blank"><strong>{nm}</strong></a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**{nm}**")
                for k, v in ev.items():
                    if k not in ("event_name", "url"):
                        st.markdown(f"**{k}**: {v}")
                st.markdown("---")
        else:
            if data.get("sql_query"):
                st.warning("No results found for that query.")

        # Display the SQL and offer to ask another
        st.markdown(
            f"""**SQL Query**:
```
{data.get('sql_query', '')}
```""",
                unsafe_allow_html=False
            )
        if st.button("Ask another question"):
            st.session_state.stage = 1
            st.session_state.final_question = None

    except Exception as e:
        st.error(f"Error querying backend: {e}")
        logging.error(f"Backend error: {e}")

# Render conversation history (newest first)
for msg in reversed(st.session_state.messages):
    prefix = "**You wrote:**" if msg["role"] == "user" else ""
    st.markdown(f"{prefix} {msg['content']}")
