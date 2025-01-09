import json
import pandas as pd
import re

json_str = """
{
        "org_name": "Salsa Del Barrio",
        "dance_style": "salsa",
        "url": "",
        "event_type": "other",
        "event_name": "Salsa Del Barrio",
        "day_of_week": "Saturday",
        "start_date": "2023-11-11",
        "end_date": "2023-11-11",
        "start_time": "11:00",
        "end_time": "12:00",
        "price": "",
        "location": "",
        "description": "Radio show featuring interviews with local and international Salsa dancers and musicians, the history of Salsa music and dance and lots of great Salsa music!"
    },
    {
        "org_name": "Salsa Del Barrio",
        "dance_style": "salsa",
        "url": "",
        "event_type": "[other, fred]",
        "event_name": "Salsa Del Barrio",
        "day_of_week": "Saturday",
        "start_date": "2023-11-18",
        "end_date": "2023-11-18",
        "start_time": "11:00",
        "end_time": "12:00",
        "price": "",
        "location": "",
        "description": "Radio show featuring interviews with local and international Salsa dancers and musicians, the history of Salsa music and dance and lots of great Salsa music!"
    },
    {
        "org_name": "Salsa Del Barrio",
        "dance_style": "salsa",
        "url": "",
        "event_type": "other",
        "event_name": "[Salsa Del Barrio]",
        "day_of_week": "Saturday",
        "start_date": "2023-11-25",
        "end_date": "2023-11-25",
        "start_time": "11:00",
        "end_time": "12:00",
        "price": "",
        "location": "",
        "description": "Radio show featuring interviews with local and international Salsa dancers and musicians, the history of Salsa music and dance and lots of great Salsa music!"
    },
    // ... Repeat for each Saturday up to 2024-10-05
]
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
