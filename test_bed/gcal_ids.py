import pandas as pd
import re

def extract_gcal_ids(input_csv='keyword_context.csv', output_csv='calendar_ids.csv'):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Initialize a list to store extracted Google Calendar IDs
    gcal_ids = []
    
    # Define a regex pattern to capture the 'gcal' ID
    pattern = re.compile(r'gcal"\s*:\s*"([^"]+)"')
    
    # Iterate through each row in the 'Full_Context' column
    for context in df['Full_Context']:
        # Find all matches of the pattern in the context
        matches = pattern.findall(context)
        for match in matches:
            gcal_ids.append(match)
    
    # Remove duplicate IDs by converting the list to a set
    unique_gcal_ids = list(set(gcal_ids))
    
    # Create a DataFrame from the unique Google Calendar IDs
    gcal_df = pd.DataFrame(unique_gcal_ids, columns=['Google_Calendar_ID'])
    
    # Save the extracted IDs to a new CSV file
    gcal_df.to_csv(output_csv, index=False)
    
    print(f"Extracted {len(unique_gcal_ids)} unique Google Calendar IDs and saved to '{output_csv}'.")

if __name__ == "__main__":
    extract_gcal_ids()
