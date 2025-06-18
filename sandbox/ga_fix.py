#!/usr/bin/env python3
import os
import re
import pandas as pd
import yaml
from googleapiclient.discovery import build
from credentials import get_credentials
from db import DatabaseHandler

# regex to match “123 Any Street [optional comma] Victoria, BC[, Canada]”
ADDRESS_PATTERN = re.compile(
    r'\b(\d+\s+[A-Za-z0-9\s]+?,?\s*Victoria,\s*BC(?:,\s*Canada)?)'
)

def search_google(query: str, api_key: str, cse_id: str, num: int = 10):
    service = build("customsearch", "v1", developerKey=api_key)
    resp = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return [
        {"title": it.get("title", ""), "snippet": it.get("snippet", "")}
        for it in resp.get("items", [])
    ]

def extract_address(snippet: str) -> str:
    if not isinstance(snippet, str):
        return ""
    m = ADDRESS_PATTERN.search(snippet)
    return m.group(1) if m else ""

def main():
    # 1) Load config and init DB handler
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    db_handler = DatabaseHandler(config)

    # 2) Get Google API creds
    _, api_key, cse_id = get_credentials("Google")

    # 3) Query 5 distinct locations with NULL address_id
    sql = """
    SELECT DISTINCT ON (location)
      event_id,
      location
    FROM events
    WHERE address_id IS NULL
    ORDER BY location, event_id
    LIMIT 5;
    """
    loc_df = pd.read_sql(sql, db_handler.conn)

    # 4) Prepare output file
    output_dir  = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gs_address.csv")
    if os.path.exists(output_path):
        os.remove(output_path)

    # 5) Loop through each location, search & append
    for row in loc_df.itertuples(index=False):
        event_id = row.event_id
        location = row.location.strip()
        query    = f"Please give me the the full and complete address for {location} in Victoria, BC, Canada including the postal code."

        results = search_google(query, api_key, cse_id, num=10)
        if not results:
            continue

        df = pd.DataFrame(results, columns=["title", "snippet"])
        df.insert(0, "query",    query)
        df.insert(1, "event_id", event_id)
        df.insert(2, "location", location)
        df["regex"] = df["snippet"].apply(extract_address)

        write_header = not os.path.exists(output_path)
        df.to_csv(output_path, mode='a', index=False, header=write_header)

        print(f"Appended {len(df)} rows for event_id={event_id}, location='{location}'")

    print(f"\nAll done! Results saved to {output_path}")

if __name__ == "__main__":
    main()
