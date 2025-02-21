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
print(keywords_list)

python_file_name = __file__.split('/')[-1]
print(python_file_name)
