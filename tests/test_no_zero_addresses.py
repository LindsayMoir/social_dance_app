#!/usr/bin/env python3
"""
Test script to verify that the fixed methods no longer return address_id = 0.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
from llm import LLMHandler
import yaml

def test_no_zero_addresses():
    """Test that methods create valid address_id values instead of 0."""
    print("\n=== Testing No Zero Addresses ===")
    
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize handlers
    db_handler = DatabaseHandler(config)
    llm_handler = LLMHandler()
    
    # Test 1: process_event_address with short location
    print("\n--- Test 1: process_event_address with short location ---")
    test_event = {
        "event_name": "Test Event",
        "location": "x",  # Very short location
        "url": "https://test.com"
    }
    
    result_event = db_handler.process_event_address(test_event)
    
    if result_event and result_event.get("address_id") and result_event["address_id"] > 0:
        print(f"✓ Success: address_id = {result_event['address_id']} (not 0)")
    else:
        print(f"✗ Failed: address_id = {result_event.get('address_id') if result_event else 'None'}")
    
    # Test 2: process_event_address with None location
    print("\n--- Test 2: process_event_address with None location ---")
    test_event2 = {
        "event_name": "Test Event 2",
        "location": None,
        "url": "https://test2.com"
    }
    
    result_event2 = db_handler.process_event_address(test_event2)
    
    if result_event2 and result_event2.get("address_id") and result_event2["address_id"] > 0:
        print(f"✓ Success: address_id = {result_event2['address_id']} (not 0)")
    else:
        print(f"✗ Failed: address_id = {result_event2.get('address_id') if result_event2 else 'None'}")
    
    # Test 3: parse_location_with_llm with very short location
    print("\n--- Test 3: parse_location_with_llm with short location ---")
    result = llm_handler.parse_location_with_llm("xyz")  # Very short
    
    if result and result.get("address_id") and result["address_id"] > 0:
        print(f"✓ Success: address_id = {result['address_id']} (not 0)")
    elif result is None:
        print("✓ Success: Returned None (acceptable, not 0)")
    else:
        print(f"✗ Failed: address_id = {result.get('address_id') if result else 'None'}")
    
    # Test 4: resolve_or_insert_address edge cases
    print("\n--- Test 4: resolve_or_insert_address edge cases ---")
    edge_cases = [
        {"city": "Test City"},
        {"building_name": "Test Building", "city": "Vancouver"},
        {"building_name": "", "city": "", "province_or_state": "BC"}
    ]
    
    for i, test_case in enumerate(edge_cases, 1):
        result = db_handler.resolve_or_insert_address(test_case)
        if result and result > 0:
            print(f"✓ Edge case {i}: address_id = {result} (not 0)")
        else:
            print(f"✗ Edge case {i}: address_id = {result}")

def check_database_state():
    """Check that we have no 0 address_id values in events table."""
    print("\n=== Checking Database State ===")
    
    import subprocess
    
    result = subprocess.run([
        'psql', '-h', 'localhost', '-U', 'postgres', '-d', 'social_dance_db',
        '-c', '''
        SELECT 
            'Zero address_id' as status,
            COUNT(*) as count  
        FROM events
        WHERE address_id = 0
        UNION ALL
        SELECT 
            'NULL address_id' as status,
            COUNT(*) as count  
        FROM events
        WHERE address_id IS NULL;
        '''
    ], env={'PGPASSWORD': '5539'}, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Database state:")
        print(result.stdout)
    else:
        print(f"Database check failed: {result.stderr}")

if __name__ == "__main__":
    print("Testing that methods no longer return address_id = 0...")
    
    # Set up minimal logging
    logging.basicConfig(level=logging.WARNING)
    
    test_no_zero_addresses()
    check_database_state()
    
    print("\n=== Summary ===")
    print("✓ If all tests pass, methods now create valid addresses instead of setting 0")
    print("✓ The existing address resolution pipeline should work properly")