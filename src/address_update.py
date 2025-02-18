from dotenv import load_dotenv
load_dotenv()
import os
import requests

api_key = os.getenv('GOOGLE_KEY_PW')
postal_code = 'V8N1S3'  # Example postal code

# Construct the API URL
url = f"https://maps.googleapis.com/maps/api/geocode/json?address={postal_code}&components=country:CA&key={api_key}"

# Make the request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    if data['status'] == 'OK' and data['results']:
        location = data['results'][0]
        address = location['formatted_address']
        street_number = None
        street_name = None

        # Extract street number and street name
        for component in location['address_components']:
            if 'street_number' in component['types']:
                street_number = component['long_name']
            if 'route' in component['types']:
                street_name = component['long_name']

        print(f"Full Address: {address}")
        print(f"Street Number: {street_number}")
        print(f"Street Name: {street_name}")
    else:
        print("No results found or error in response.")
else:
    print(f"Error: {response.status_code}")
