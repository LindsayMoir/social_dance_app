from dotenv import load_dotenv
load_dotenv()
import os
import requests


def get_municipality(address, api_key):
    """
    Given an address string, query the Google Geocoding API and extract the municipality.

    Args:
        address (str): The address string to geocode.
        api_key (str): Your Google Maps Geocoding API key.

    Returns:
        str or None: The municipality (locality) if found, otherwise None.
    """
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        raise Exception(f"Geocoding API request failed with status code {response.status_code}")
    
    data = response.json()
    if data.get("status") != "OK":
        raise Exception(f"Geocoding API error: {data.get('status')}")
    
    # Iterate through the results to find the "locality" component (which is typically the municipality)
    for result in data.get("results", []):
        for component in result.get("address_components", []):
            if "locality" in component.get("types", []):
                return component.get("long_name")
    return None

def get_postal_code(address, api_key):
    """
    Given an address string, query the Google Geocoding API and extract the postal code.

    Args:
        address (str): The address string to geocode.
        api_key (str): Your Google Maps Geocoding API key.

    Returns:
        str or None: The postal code if found, otherwise None.
    """
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }
    
    response = requests.get(endpoint, params=params)
    if response.status_code != 200:
        raise Exception(f"Geocoding API request failed with status code {response.status_code}")
    
    data = response.json()
    if data.get("status") != "OK":
        raise Exception(f"Geocoding API error: {data.get('status')}")
    
    # Look for the postal_code component in the results.
    for result in data.get("results", []):
        for component in result.get("address_components", []):
            if "postal_code" in component.get("types", []):
                return component.get("long_name")
    return None

# Example usage:
if __name__ == "__main__":
    address = "411 Gorge Rd E, BC, V8T 2W1, CA"
    google_api_key = os.getenv("GOOGLE_KEY_PW")
    municipality = get_municipality(address, google_api_key)
    if municipality:
        print(f"The likely municipality is: {municipality}")
    else:
        print("Municipality could not be determined.")

    postal_code = get_postal_code(address, google_api_key)
    if postal_code:
        print(f"The postal code is: {postal_code}")
    else:
        print("Postal code could not be determined.")
