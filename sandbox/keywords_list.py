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

muni_list = "Victoria, Oak Bay, View Royal, Colwood, Sidney, Sooke, Vancouver, British Columbia, Canada, Saanich, Langford, Esquimalt, \
Port Alberni, Courtenay, Cowichan, Port Hardy, Campbell River, Port Renfrew, Nanaimo, White Rock, West Vancouver, North Vancouver, \
Richmond, Delta, Coquitlam, Langley, Maple Ridge, Pitt Meadows. Abbotsford, Bowen Island, Burnaby"

# Convert muni_list to a sorted set
muni_set = sorted(set(muni.strip() for muni in muni_list.split(',')))

# Print the sorted set
print(muni_set)

# Write to a text file with each muni on a new line
with open('data/other/municipalities.txt', 'w') as file:
    for muni in muni_set:
        file.write(f"{muni}\n")
