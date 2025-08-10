#!/usr/bin/env python3
"""
Comprehensive test to verify the entire address resolution system works correctly
after all the changes, and hasn't introduced bugs.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
from llm import LLMHandler
import yaml

def test_resolve_or_insert_address():
    """Test the core resolve_or_insert_address method extensively."""
    print("\n=== Testing resolve_or_insert_address Core Method ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    # Test cases that previously failed
    test_cases = [
        {
            "name": "Valid complete address",
            "data": {
                "building_name": "Test Building",
                "street_number": "123",
                "street_name": "Main Street",
                "city": "Vancouver",
                "province_or_state": "BC",
                "postal_code": "V6B 1A1",
                "country_id": "Canada"
            },
            "should_succeed": True
        },
        {
            "name": "Minimal address (only building and city)",
            "data": {
                "building_name": "Minimal Building",
                "city": "Vancouver"
            },
            "should_succeed": True
        },
        {
            "name": "Empty address data",
            "data": {},
            "should_succeed": False
        },
        {
            "name": "Address with some missing fields",
            "data": {
                "building_name": "Partial Building",
                "city": "Vancouver",
                "province_or_state": "BC"
            },
            "should_succeed": True
        }
    ]
    
    success_count = 0
    for test_case in test_cases:
        try:
            result = db_handler.resolve_or_insert_address(test_case["data"])
            
            if test_case["should_succeed"]:
                if result and result > 0:
                    print(f"✓ {test_case['name']}: address_id = {result}")
                    success_count += 1
                else:
                    print(f"✗ {test_case['name']}: Expected success, got {result}")
            else:
                if result is None:
                    print(f"✓ {test_case['name']}: Correctly returned None")
                    success_count += 1
                else:
                    print(f"✗ {test_case['name']}: Expected None, got {result}")
                    
        except Exception as e:
            print(f"✗ {test_case['name']}: Exception - {e}")
    
    print(f"Core method tests: {success_count}/{len(test_cases)} passed")
    return success_count == len(test_cases)

def test_process_event_address():
    """Test the process_event_address method with various edge cases."""
    print("\n=== Testing process_event_address Method ===")
    
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    db_handler = DatabaseHandler(config)
    
    test_events = [
        {
            "name": "Normal event with good location",
            "event": {
                "event_name": "Normal Dance Event",
                "location": "Community Center, 123 Main St, Vancouver, BC",
                "url": "https://test.com/normal"
            }
        },
        {
            "name": "Event with very short location",
            "event": {
                "event_name": "Short Location Event", 
                "location": "x",
                "url": "https://test.com/short"
            }
        },
        {
            "name": "Event with None location",
            "event": {
                "event_name": "No Location Event",
                "location": None,
                "url": "https://test.com/none"
            }
        },
        {
            "name": "Event with empty location",
            "event": {
                "event_name": "Empty Location Event",
                "location": "",
                "url": "https://test.com/empty"
            }
        }
    ]
    
    success_count = 0
    for test_case in test_events:
        try:
            result = db_handler.process_event_address(test_case["event"])
            
            if result and result.get("address_id") and result["address_id"] > 0:
                print(f"✓ {test_case['name']}: address_id = {result['address_id']}")
                success_count += 1
            else:
                print(f"✗ {test_case['name']}: Failed - {result}")
                
        except Exception as e:
            print(f"✗ {test_case['name']}: Exception - {e}")
    
    print(f"Event processing tests: {success_count}/{len(test_events)} passed")
    return success_count == len(test_events)

def test_llm_methods():
    """Test LLM-related address methods."""
    print("\n=== Testing LLM Address Methods ===")
    
    llm_handler = LLMHandler()
    
    # Disable actual LLM calls for testing
    llm_handler.config['llm']['spend_money'] = False
    
    test_cases = [
        {
            "name": "Very short location string",
            "location": "abc"
        },
        {
            "name": "Empty location string", 
            "location": ""
        },
        {
            "name": "None location string",
            "location": None
        }
    ]
    
    success_count = 0
    for test_case in test_cases:
        try:
            result = llm_handler.parse_location_with_llm(test_case["location"])
            
            # Should either return valid address_id or None (not 0)
            if result is None:
                print(f"✓ {test_case['name']}: Returned None (acceptable)")
                success_count += 1
            elif result and result.get("address_id") and result["address_id"] > 0:
                print(f"✓ {test_case['name']}: address_id = {result['address_id']}")
                success_count += 1
            else:
                print(f"✗ {test_case['name']}: Bad result - {result}")
                
        except Exception as e:
            print(f"✗ {test_case['name']}: Exception - {e}")
    
    print(f"LLM method tests: {success_count}/{len(test_cases)} passed") 
    return success_count == len(test_cases)

def check_database_integrity():
    """Check that database is in a valid state."""
    print("\n=== Checking Database Integrity ===")
    
    import subprocess
    
    # Check for invalid address_id values
    result = subprocess.run([
        'bash', '-c',
        'PGPASSWORD=5539 psql -h localhost -U postgres -d social_dance_db -c "'
        'SELECT '
        '    \'Zero address_id\' as status,'
        '    COUNT(*) as count  '
        'FROM events '
        'WHERE address_id = 0 '
        'UNION ALL '
        'SELECT  '
        '    \'NULL address_id\' as status,'
        '    COUNT(*) as count  '
        'FROM events '
        'WHERE address_id IS NULL '
        'UNION ALL '
        'SELECT '
        '    \'Invalid address_id refs\' as status,'
        '    COUNT(*) as count '
        'FROM events e '
        'LEFT JOIN address a ON e.address_id = a.address_id '
        'WHERE e.address_id IS NOT NULL AND a.address_id IS NULL;"'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Database integrity check:")
        print(result.stdout)
        
        # Parse output to check if all counts are 0
        lines = result.stdout.strip().split('\n')
        integrity_ok = True
        for line in lines:
            if '|' in line and not line.startswith(' '):
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        count = int(parts[1].strip())
                        if count > 0:
                            integrity_ok = False
                    except:
                        pass
        
        if integrity_ok:
            print("✓ Database integrity: All checks passed")
            return True
        else:
            print("✗ Database integrity: Issues found")
            return False
    else:
        print(f"✗ Database check failed: {result.stderr}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("🧪 COMPREHENSIVE ADDRESS SYSTEM TEST")
    print("=" * 50)
    
    # Suppress most logging for cleaner output
    logging.getLogger().setLevel(logging.ERROR)
    
    results = []
    
    # Run all test suites
    results.append(("Core resolve_or_insert_address", test_resolve_or_insert_address()))
    results.append(("Event address processing", test_process_event_address()))
    results.append(("LLM address methods", test_llm_methods())) 
    results.append(("Database integrity", check_database_integrity()))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'=' * 50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 OVERALL: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System appears to be working correctly!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - There may be bugs introduced by the changes!")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)