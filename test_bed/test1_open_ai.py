#!/usr/bin/env python

import os
import pandas as pd
from openai import OpenAI

# Read the API key from the Excel file
keys_df = pd.read_csv('/mnt/d/OneDrive/Security/keys.csv')
api_key = keys_df.loc[keys_df['Organization'] == 'OpenAI', 'Key'].values[0]

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

client = OpenAI()

response = client.chat.completions.create(
    model="o1-preview",
    messages=[
        {
            "role": "user", 
            "content": "Please provide me in JSON format, information on organizations in Victoria, BC, Canada, \
                that host social dancing events for salsa, bachata, kizomba, and west coast swing. It is likley that these \
                    venues and organizations have websites, social media pages, and event pages. In the JSON output \
                        please include a description."
        }
    ]
)

print(response.choices[0].message.content)