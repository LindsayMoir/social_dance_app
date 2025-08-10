#!/usr/bin/env python3
"""
Test script to verify the fallback address system prevents NULL and 0 address_id values.
"""

import sys
import os
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler
from llm import LLMHandler
import yaml

def test_fallback_address():
    """Test that the fallback address system works correctly."""
    print("\n=== Testing Fallback Address System ===")
    
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize handlers
    db_handler = DatabaseHandler(config)
    llm_handler = LLMHandler()
    
    # Test 1: Create fallback address
    print("\n--- Test 1: Create/Get Fallback Address ---")
    try:
        fallback_id = db_handler.get_or_create_fallback_address()
        print(f"✓ Fallback address created/retrieved with ID: {fallback_id}")
        
        # Verify it exists in database
        query = "SELECT full_address FROM address WHERE address_id = :address_id"
        result = db_handler.execute_query(query, {"address_id": fallback_id})
        if result:
            full_address = result[0][0]
            print(f"✓ Fallback address verified in DB: {full_address}")
        else:
            print("✗ Fallback address not found in database")
            
    except Exception as e:
        print(f"✗ Error creating fallback address: {e}")
    
    # Test 2: Test process_event_address with invalid location
    print("\n--- Test 2: Test process_event_address with invalid location ---")
    try:
        test_event = {
            "event_name": "Test Event",
            "location": "x",  # Too short, should trigger fallback
            "url": "https://test.com"
        }
        
        result_event = db_handler.process_event_address(test_event)
        
        if result_event and result_event.get("address_id") and result_event["address_id"] > 0:
            print(f"✓ process_event_address returned valid address_id: {result_event['address_id']}")
        else:
            print(f"✗ process_event_address failed: {result_event}")
            
    except Exception as e:
        print(f"✗ Error testing process_event_address: {e}")
    
    # Test 3: Test parse_location_with_llm with short location
    print("\n--- Test 3: Test parse_location_with_llm with short location ---")
    try:
        result = llm_handler.parse_location_with_llm("abc")  # Too short
        
        if result and result.get("address_id") and result["address_id"] > 0:
            print(f"✓ parse_location_with_llm returned valid address_id: {result['address_id']}")
        else:
            print(f"✗ parse_location_with_llm failed: {result}")
            
    except Exception as e:
        print(f"✗ Error testing parse_location_with_llm: {e}")

if __name__ == "__main__":
    print("Testing fallback address system...")
    
    # Set up minimal logging
    logging.basicConfig(level=logging.WARNING)
    
    test_fallback_address()
    
    print("\n=== Test Summary ===")
    print("✓ If all tests pass, the system should prevent NULL and 0 address_id values")
    print("✓ Events with unparseable locations will get a valid fallback address_id")
    print("✓ This maintains referential integrity with the address table")