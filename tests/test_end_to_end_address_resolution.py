#!/usr/bin/env python3
"""
End-to-end test of the improved address resolution system.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
import yaml

def test_duplicate_prevention():
    """Test that the system now prevents creating duplicates."""
    print("\n=== Testing Duplicate Prevention ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    # Test cases that should all resolve to the same address_id
    test_variations = [
        {
            "name": "Exact match to address_id=1",
            "data": {
                "building_name": "Edelweiss Club",
                "street_number": "108",
                "street_name": "Niagra",  # Original spelling
                "city": "Victoria",
                "postal_code": "V8V 1E9",
                "country_id": "CA"
            }
        },
        {
            "name": "Variation with corrected street name",
            "data": {
                "building_name": "Edelweiss Club",
                "street_number": "108", 
                "street_name": "Niagara",  # Corrected spelling
                "city": "Victoria",
                "postal_code": "V8V 1E9",
                "country_id": "CA"
            }
        },
        {
            "name": "Variation with 'Victoria' prefix",
            "data": {
                "building_name": "Victoria Edelweiss Club",
                "street_number": "108",
                "street_name": "Niagara",
                "city": "Victoria", 
                "postal_code": "V8V 1E9",
                "country_id": "CA"
            }
        },
        {
            "name": "Minimal info with postal code",
            "data": {
                "building_name": "Edelweiss",
                "street_number": "108",
                "postal_code": "V8V 1E9",
                "city": "Victoria"
            }
        }
    ]
    
    results = []
    expected_address_id = 1
    
    for test_case in test_variations:
        try:
            result = db_handler.resolve_or_insert_address(test_case["data"])
            results.append((test_case["name"], result))
            
            if result == expected_address_id:
                print(f"✅ {test_case['name']}: Found address_id={result} (correct)")
            else:
                print(f"❌ {test_case['name']}: Found address_id={result} (expected {expected_address_id})")
                
        except Exception as e:
            print(f"❌ {test_case['name']}: Exception - {e}")
            results.append((test_case["name"], "Error"))
    
    # Check if all results are the same (no duplicates created)
    unique_results = set(r[1] for r in results if r[1] != "Error")
    if len(unique_results) == 1 and expected_address_id in unique_results:
        print(f"\n✅ SUCCESS: All variations resolved to address_id={expected_address_id}")
        return True
    else:
        print(f"\n❌ FAILURE: Got different address_ids: {unique_results}")
        return False

def test_different_locations():
    """Test that genuinely different locations get different address_ids."""
    print("\n=== Testing Different Locations Get Different IDs ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    different_locations = [
        {
            "name": "Different postal code",
            "data": {
                "building_name": "Some Other Club",
                "street_number": "108", 
                "street_name": "Niagara",
                "city": "Victoria",
                "postal_code": "V8V 2A2",  # Different postal code
                "country_id": "CA"
            }
        },
        {
            "name": "Different street number",
            "data": {
                "building_name": "Another Club", 
                "street_number": "110",  # Different street number
                "street_name": "Niagara",
                "city": "Victoria",
                "postal_code": "V8V 1E9",
                "country_id": "CA"
            }
        }
    ]
    
    edelweiss_id = 1
    results = []
    
    for test_case in different_locations:
        try:
            result = db_handler.resolve_or_insert_address(test_case["data"])
            results.append((test_case["name"], result))
            
            if result != edelweiss_id:
                print(f"✅ {test_case['name']}: Got address_id={result} (correctly different from {edelweiss_id})")
            else:
                print(f"❌ {test_case['name']}: Got address_id={result} (incorrectly same as Edelweiss)")
                
        except Exception as e:
            print(f"❌ {test_case['name']}: Exception - {e}")
            results.append((test_case["name"], "Error"))
    
    # Check that different locations got different IDs
    all_different = all(r[1] != edelweiss_id and r[1] != "Error" for r in results)
    if all_different:
        print(f"\n✅ SUCCESS: Different locations got different address_ids")
        return True
    else:
        print(f"\n❌ FAILURE: Some different locations were incorrectly matched")
        return False

def run_end_to_end_test():
    """Run comprehensive end-to-end test."""
    print("🎯 END-TO-END ADDRESS RESOLUTION TEST")
    print("=" * 45)
    
    # Suppress most logging for cleaner output
    logging.getLogger().setLevel(logging.WARNING)
    
    tests = [
        ("Duplicate prevention", test_duplicate_prevention()),
        ("Different locations", test_different_locations())
    ]
    
    print(f"\n{'=' * 45}")
    print("📊 END-TO-END TEST RESULTS")
    print(f"{'=' * 45}")
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 OVERALL: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Address resolution system is working correctly!")
        print("   - Prevents duplicate addresses")
        print("   - Uses improved fuzzy matching")
        print("   - Prioritizes existing addresses over creating new ones")
        return True
    else:
        print("⚠️  Address resolution system needs further adjustment")
        return False

if __name__ == "__main__":
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)