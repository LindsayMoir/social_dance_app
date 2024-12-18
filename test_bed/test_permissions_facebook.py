import requests
import pandas as pd

# Read the keys from the security file
keys_df = pd.read_csv('/mnt/d/OneDrive/Security/keys.csv')
ACCESS_TOKEN = keys_df.loc[keys_df['Organization'] == 'Meta', 'Access_Token'].values[0]

# Debug permissions
def check_permissions():
    url = f"https://graph.facebook.com/v16.0/me/permissions?access_token={ACCESS_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        permissions = response.json()['data']
        for perm in permissions:
            print(f"Permission: {perm['permission']}, Status: {perm['status']}")
    else:
        print(f"Error checking permissions: {response.status_code} - {response.text}")

check_permissions()
