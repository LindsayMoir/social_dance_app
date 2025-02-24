import pandas as pd
import re

# Load keywords
keywords_df = pd.read_csv('data/other/keywords.csv')

# Convert to a list, strip spaces, split on commas, and remove duplicates
keywords_list = sorted(set(
    keyword.strip()
    for keywords in keywords_df["keywords"]
    for keyword in str(keywords).split(',')
))

# Print the number of keywords and first 10
print(f"Total keywords: {len(keywords_list)}")
print(keywords_list)

extracted_text = 'What is going on. THIS WEEK we are seeing lots of fun!!! Who knows'

# Perform regex search (convert string to actual regex)
match = re.search('(?is)This Week(.*)$', extracted_text, re.DOTALL)

if match:
    extracted_text = match.group(0)  # Extract the matched portion
    print(extracted_text)



