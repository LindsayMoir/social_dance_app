import pandas as pd

keywords_df = pd.read_csv("data/other/keywords.csv")

# Convert to a list, strip spaces, split on commas, and remove duplicates
keywords_list = sorted(set(
    keyword.strip()
    for keywords in keywords_df["keywords"]
    for keyword in str(keywords).split(',')
))

print(keywords_list)