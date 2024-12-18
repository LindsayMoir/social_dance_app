import requests
import pandas as pd

# Read the API key from the security file
keys_df = pd.read_csv('/mnt/d/OneDrive/Security/keys.csv')
api_key = keys_df.loc[keys_df['Organization'] == 'Google', 'Key'].values[0]

url = 'https://www.googleapis.com/calendar/v3/calendars/nufist0fq81lhh238ptkgjdhl8@group.calendar.google.com/events'

response = requests.get(url, params={'key': api_key})

print(response.text)