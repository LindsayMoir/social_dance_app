import json
import pandas as pd
import re

json_str = """
    [{
        "org_name": "Miguelito Valdes Trio",
        "dance_style": "swing",
        "url": "",
        "event_type": "other",
        "event_name": "Miguelito Valdes Trio",
        "day_of_week": "Tuesday",
        "start_date": "2025-01-07",
        "end_date": "2025-01-07",
        "start_time": "20:30",
        "end_time": "",
        "price": "",
        "location": "",
        "description": ""
    },
    {
        "org_name": "Cuban Salsa Club",
        "dance_style": "salsa",
        "url": "",
        "event_type": "social dance",
        "event_name": "[tentative] Cuban Salsa Club",
        "day_of_week": "Wednesday",
        "start_date": "2025-01-08",
        "end_date": "2025-01-08",
        "start_time": "19:00",
        "end_time": "",
        "price": "",
        "location": "",
        "description": ""
    }]
"""

# Step 1: Remove single-line comments
no_comments = re.sub(r'\s*//.*', '', json_str)

# Step 2: Remove ellipsis patterns (if they occur)
cleaned_str = re.sub(r',\s*\.\.\.\s*', '', no_comments, flags=re.DOTALL)

# Step 3: Ensure the string is a valid JSON array
cleaned_str = cleaned_str.strip()

# If the string doesn't start with '[', prepend it.
if not cleaned_str.startswith('['):
    cleaned_str = '[' + cleaned_str

# If the string doesn't end with ']', append it.
if not cleaned_str.endswith(']'):
    cleaned_str = cleaned_str + ']'

# Step 4: Remove any trailing commas before the closing bracket
cleaned_str = re.sub(r',\s*\]', ']', cleaned_str)

# For debugging: print the cleaned JSON string
print(cleaned_str)

# Parse the cleaned JSON string
try:
    data = json.loads(cleaned_str)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {e}")
    data = []

# Convert to DataFrame if parsing was successful
if data:
    df = pd.DataFrame(data)
    print(df)
else:
    print("No data parsed.")
