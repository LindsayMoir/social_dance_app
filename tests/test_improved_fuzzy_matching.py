#!/usr/bin/env python3
"""
Test the improved fuzzy matching logic.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
import yaml

def test_edelweiss_club_matching():
    """Test that the improved matching would find address_id=1 for Edelweiss Club."""
    print("\n=== Testing Edelweiss Club Improved Matching ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    # This is the address data that would have created address_id=712
    # (similar to what was processed from "108 Niagara Street Victoria, BC V8V 1E9")
    test_address = {
        "building_name": "Victoria Edelweiss Club",
        "street_number": "108",
        "street_name": "Niagara",
        "city": "Victoria",
        "province_or_state": "BC",
        "postal_code": "V8V 1E9",
        "country_id": "Canada"
    }
    
    print("Testing address that should match existing Edelweiss Club:")
    print(f"  Building: {test_address['building_name']}")
    print(f"  Address: {test_address['street_number']} {test_address['street_name']}")
    print(f"  Postal: {test_address['postal_code']}")
    
    try:
        result = db_handler.resolve_or_insert_address(test_address)
        
        if result == 1:
            print(f"✅ SUCCESS: Found address_id=1 (correct existing address)")
        elif result == 712:
            print(f"⚠️  Found address_id=712 (duplicate that should be avoided)")
        else:
            print(f"❌ Found address_id={result} (unexpected)")
            
        return result
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_various_edelweiss_variations():
    """Test various ways the Edelweiss Club might be referenced."""
    print("\n=== Testing Edelweiss Club Variations ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    test_cases = [
        {
            "name": "Original Edelweiss Club format",
            "data": {
                "building_name": "Edelweiss Club",
                "street_number": "108", 
                "street_name": "Niagra",  # Original typo
                "city": "Victoria",
                "postal_code": "V8V 1E9"
            }
        },
        {
            "name": "Victoria Edelweiss Club format",
            "data": {
                "building_name": "Victoria Edelweiss Club",
                "street_number": "108",
                "street_name": "Niagara",  # Corrected spelling
                "city": "Victoria", 
                "postal_code": "V8V 1E9"
            }
        },
        {
            "name": "Just 'Edelweiss' with postal code",
            "data": {
                "building_name": "Edelweiss",
                "street_number": "108",
                "street_name": "Niagara",
                "city": "Victoria",
                "postal_code": "V8V 1E9"
            }
        }
    ]
    
    results = {}
    for test_case in test_cases:
        try:
            # Don't actually call resolve_or_insert_address to avoid creating more entries
            # Just test the matching logic
            print(f"\nTesting: {test_case['name']}")
            
            # Simulate what the postal code + street number matching would find
            postal_code = test_case['data'].get('postal_code', '').strip()
            street_number = test_case['data'].get('street_number', '').strip()
            
            if postal_code and street_number:
                postal_match_query = """
                    SELECT address_id, building_name, street_number, street_name, postal_code
                    FROM address
                    WHERE LOWER(postal_code) = LOWER(:postal_code)
                    AND LOWER(street_number) = LOWER(:street_number)
                """
                matches = db_handler.execute_query(postal_match_query, {
                    "postal_code": postal_code,
                    "street_number": street_number
                })
                
                print(f"  Found {len(matches) if matches else 0} postal+street matches:")
                if matches:
                    for addr_id, b_name, s_num, s_name, p_code in matches:
                        print(f"    address_id={addr_id}: {b_name}")
                
                results[test_case['name']] = len(matches) if matches else 0
            
        except Exception as e:
            print(f"  Exception: {e}")
            results[test_case['name']] = "Error"
    
    return results

if __name__ == "__main__":
    print("🔍 TESTING IMPROVED FUZZY MATCHING")
    print("=" * 40)
    
    # Enable detailed logging to see the matching process
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Test the specific case
    result = test_edelweiss_club_matching()
    
    # Test variations
    variations_results = test_various_edelweiss_variations()
    
    print(f"\n{'=' * 40}")
    print("📊 RESULTS SUMMARY")
    print(f"{'=' * 40}")
    print(f"Main test result: address_id = {result}")
    print(f"Variations found matches: {variations_results}")
    
    if result == 1:
        print("✅ Improved matching is working correctly!")
    else:
        print("⚠️  May need further tuning of matching thresholds")