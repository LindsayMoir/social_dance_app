#!/usr/bin/env python3
import requests

def geocode_cameron_bandshell():
    """
    Search for “Cameron Bandshell” but restrict to Beacon Hill Park’s bounding box.
    Returns address, postal code, latitude & longitude.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q":              "Cameron Bandshell",
        "format":         "json",
        "addressdetails": 1,
        "limit":          1,
        # Beacon Hill Park bbox:  (lon_min, lat_max, lon_max, lat_min)
        "viewbox":        "-123.3660,48.4335,-123.3590,48.4255",
        "bounded":        1
    }
    headers = {"User-Agent": "social_dance_app (you@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        return {}

    top = data[0]
    addr_parts = top.get("address", {})
    formatted = top.get("display_name", "")
    postcode  = addr_parts.get("postcode", "")
    return {
        "formatted_address": formatted,
        "postal_code":       postcode,
        "latitude":          float(top["lat"]),
        "longitude":         float(top["lon"])
    }

if __name__ == "__main__":
    result = geocode_cameron_bandshell()
    if not result:
        print("No result inside Beacon Hill Park")
    else:
        print("Address:     ", result["formatted_address"])
        print("Postal Code: ", result["postal_code"])
        print("Coordinates: ", result["latitude"], result["longitude"])
