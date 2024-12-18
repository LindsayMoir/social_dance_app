from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
import facebook
import os
import pandas as pd

print(os.getcwd())

# Read the keys from the security file
keys_df = pd.read_csv('/mnt/d/OneDrive/Security/keys.csv')
appid = keys_df.loc[keys_df['Organization'] == 'Meta', 'App_ID'].values[0]
appsecret = keys_df.loc[keys_df['Organization'] == 'Meta', 'Key'].values[0]
accesstoken = keys_df.loc[keys_df['Organization'] == 'Meta', 'Access_Token'].values[0]

# Initialize the Graph API client
graph = facebook.GraphAPI(access_token=accesstoken)

# Replace with your group ID
GROUP_ID = "1634269246863069"

# Keywords to look for in posts and events
KEYWORDS = ["dance", "event", "class", "workshop", "Salsa", "Bachata", "Swing", "Kizomba", "Urban Kiz"]

# Function to check if a post or event contains keywords
def contains_keywords(text, keywords):
    return any(keyword.lower() in text.lower() for keyword in keywords)

# Function to extract and print event-related information
def extract_event_info(source, event):
    name = event.get("name", "No name provided")
    description = event.get("description", "No description provided")
    start_time = event.get("start_time", "No start time provided")
    end_time = event.get("end_time", "No end time provided")
    place = event.get("place", {}).get("name", "No location provided")
    
    print(f"Source: {source}")
    print(f"Event Name: {name}")
    print(f"Description: {description}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Location: {place}")
    print("-" * 40)

try:
    # Fetch the group's feed
    print("Fetching group feed...")
    fields = "feed{message,created_time,from,link,attachments}"
    group_data = graph.get_object(id=GROUP_ID, fields=fields)

    for post in group_data.get("feed", {}).get("data", []):
        message = post.get("message", "")
        created_time = post.get("created_time", "No timestamp")
        user = post.get("from", {}).get("name", "Unknown")
        link = post.get("link", "No link provided")

        if contains_keywords(message, KEYWORDS):
            print("Event Found in Feed!")
            print(f"Posted by: {user}")
            print(f"Posted on: {created_time}")
            print(f"Message: {message}")
            print(f"Link: {link}")
            print("-" * 40)

    # Fetch the group's events
    print("Fetching group events...")
    event_fields = "events{name,description,start_time,end_time,place}"
    group_events = graph.get_object(id=GROUP_ID, fields=event_fields)

    for event in group_events.get("events", {}).get("data", []):
        if contains_keywords(event.get("name", "") + " " + event.get("description", ""), KEYWORDS):
            extract_event_info("Events Endpoint", event)

except facebook.GraphAPIError as e:
    print("Error:", e)
