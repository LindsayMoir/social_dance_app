

# keywords = ["2-step", "cha cha", "double shuffle", "country waltz", "line dance", "night club", "nite club", "nite club 2", 
#             "nite club two", "two step", "west coast swing", "wcs", "kizomba", "urban kiz", "semba", "tarraxo", "tarraxa", 
#             "tarraxinha", "douceur", "salsa", "bachata", "kizomba", "merengue", "cha cha cha", "swing", "balboa", "lindy hop", 
#             "east coast swing", "swing", "balboa", "lindy hop", "east coast swing", "milonga", "tango", "west coast swing", "wcs"]

# keywords = set(keywords)
# keywords = list(keywords)
# keywords.sort()

# # Take the list and turn it into a single string with the elements separated by commas
# keywords_str = ', '.join(keywords)
# print(keywords_str)

import pandas as pd

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
print("First 10 keywords:", keywords_list[:10])
