from fuzzywuzzy import fuzz
from googleapiclient.discovery import build
import pandas as pd
import os
import yaml

print(os.getcwd())
config_path = "config/config.yaml"

 # Load configuration from a YAML file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)


keys_df = pd.read_csv(config['input']['keys'])
keys_df = keys_df[keys_df['organization'] == 'Google']
api_key, cse_id = keys_df.iloc[0][['key_pw', 'cse_id']]


def google_search(query, api_key, cse_id, num_results=10):
    """
    Perform a Google search and extract titles and URLs of the results.

    Args:
        query (str): The search query.
        api_key (str): Your Google API key.
        cse_id (str): Your Custom Search Engine ID.
        num_results (int): The number of search results to fetch (max 10 per request).

    Returns:
        list: A list of dictionaries containing 'title' and 'url' for each search result.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    results = []
    
    #try:
    response = service.cse().list(
        q=query,
        cx=cse_id,
        num=num_results  # Fetch up to `num_results` results (max 10 per request).
    ).execute()

    # Extract titles and links from the response
    if 'items' in response:
        for item in response['items']:
            title = item.get('title')  # Get the title of the search result
            url = item.get('link')  # Get the URL of the search result
            results.append({'title': title, 'url': url})

    # except Exception as e:
    #     print(f"An error occurred: {e}")

    return results


# Example Usage
if __name__ == "__main__":


    query = "Sundown Social: Dance with Cupid"
    results = google_search(query, api_key, cse_id)

    for result in results:
        similarity = fuzz.token_set_ratio(result['title'], query)
        if similarity > 80: 
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}\n")
            print(f"Similarity: {similarity}\n")