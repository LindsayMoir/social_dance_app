#!/usr/bin/env python3
"""
Test script to identify why resolve_or_insert_address is returning None.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
import yaml

def test_resolve_address_failures():
    """Test problematic address resolution scenarios."""
    print("\n=== Testing resolve_or_insert_address Failures ===")
    
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize handler
    db_handler = DatabaseHandler(config)
    
    # Test cases that likely cause failures
    test_cases = [
        {
            "name": "Empty parsed_address",
            "data": {}
        },
        {
            "name": "Minimal parsed_address with only building_name",
            "data": {
                "building_name": "Test Venue",
                "city": "Vancouver"
            }
        },
        {
            "name": "Address with null values",
            "data": {
                "building_name": None,
                "street_number": None,
                "street_name": None,
                "city": "Vancouver"
            }
        },
        {
            "name": "Address with empty strings",
            "data": {
                "building_name": "",
                "street_number": "",
                "street_name": "",
                "city": ""
            }
        },
        {
            "name": "Missing required fields",
            "data": {
                "building_name": "Community Center"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        try:
            result = db_handler.resolve_or_insert_address(test_case['data'])
            if result:
                print(f"✓ Success: address_id = {result}")
            else:
                print(f"✗ Failed: returned None")
                # Let's see what full_address would have been generated
                if test_case['data']:
                    full_addr = db_handler.build_full_address(
                        building_name=test_case['data'].get('building_name'),
                        street_number=test_case['data'].get('street_number'),
                        street_name=test_case['data'].get('street_name'),
                        street_type=test_case['data'].get('street_type'),
                        city=test_case['data'].get('city'),
                        province_or_state=test_case['data'].get('province_or_state'),
                        postal_code=test_case['data'].get('postal_code'),
                        country_id=test_case['data'].get('country_id')
                    )
                    print(f"  Generated full_address would be: '{full_addr}'")
                
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    # Test case: Try to insert a duplicate full_address
    print(f"\n--- Test: Duplicate full_address ---")
    try:
        # First insertion
        test_data = {
            "building_name": "Duplicate Test Venue",
            "city": "Vancouver",
            "province_or_state": "BC",
            "country_id": "Canada"
        }
        result1 = db_handler.resolve_or_insert_address(test_data)
        print(f"First insertion: address_id = {result1}")
        
        # Second insertion (should find existing)
        result2 = db_handler.resolve_or_insert_address(test_data)
        print(f"Second insertion: address_id = {result2}")
        
        if result1 == result2:
            print("✓ Duplicate handling works correctly")
        else:
            print("✗ Duplicate handling failed")
            
    except Exception as e:
        print(f"✗ Exception during duplicate test: {e}")

if __name__ == "__main__":
    print("Testing resolve_or_insert_address failure scenarios...")
    
    # Set up detailed logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    
    test_resolve_address_failures()