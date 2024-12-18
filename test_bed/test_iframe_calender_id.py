import requests
from datetime import datetime, timedelta, timezone

# Replace with your API key and calendar ID
# Removed secrets
API_KEY = ''
CALENDAR_ID = ""

# API endpoint
url = f"https://www.googleapis.com/calendar/v3/calendars/{CALENDAR_ID}/events"

# Parameters for the API request
params = {
    "key": API_KEY,
    "singleEvents": "true",  # Expand recurring events into individual ones
    "timeMin": datetime.now(timezone.utc).isoformat(),  # Start time in UTC
    "timeMax": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),  # End time in UTC
    "fields": "items,nextPageToken",  # Get all items and pagination token
    "maxResults": 2500  # Max results allowed per page
}

all_events = []

# Fetch all pages of events
while True:
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        all_events.extend(data.get("items", []))  # Add items to the list
        next_page_token = data.get("nextPageToken")
        if next_page_token:
            params["pageToken"] = next_page_token  # Fetch the next page
        else:
            break  # No more pages
    else:
        print(f"Error: {response.status_code} - {response.text}")
        break

# Print every key-value pair for each event
for idx, event in enumerate(all_events, start=1):
    print(f"Event #{idx}:")
    for key, value in event.items():
        print(f"{key}: {value}")
    print("-" * 80)
