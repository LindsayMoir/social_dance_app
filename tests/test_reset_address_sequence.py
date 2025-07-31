#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def test_reset_address_sequence():
    """Test the reset_address_id_sequence method"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Testing address ID sequence reset...")
    
    # Show current address IDs before reset
    print("\nBefore reset:")
    addresses_before = db_handler.execute_query("SELECT address_id, full_address FROM address ORDER BY address_id LIMIT 10;")
    for addr in addresses_before:
        print(f"  ID: {addr[0]}, Address: {addr[1][:50]}...")
    
    # Get total count
    count_result = db_handler.execute_query("SELECT COUNT(*) FROM address;")
    total_addresses = count_result[0][0] if count_result else 0
    print(f"\nTotal addresses: {total_addresses}")
    
    # Get max address_id before reset
    max_before = db_handler.execute_query("SELECT MAX(address_id) FROM address;")
    max_id_before = max_before[0][0] if max_before and max_before[0][0] else 0
    print(f"Max address_id before: {max_id_before}")
    
    # Reset the sequence
    result = db_handler.reset_address_id_sequence()
    print(f"\nReset completed. Renumbered {result} addresses.")
    
    # Show address IDs after reset
    print("\nAfter reset:")
    addresses_after = db_handler.execute_query("SELECT address_id, full_address FROM address ORDER BY address_id LIMIT 10;")
    for addr in addresses_after:
        print(f"  ID: {addr[0]}, Address: {addr[1][:50]}...")
    
    # Get max address_id after reset
    max_after = db_handler.execute_query("SELECT MAX(address_id) FROM address;")
    max_id_after = max_after[0][0] if max_after and max_after[0][0] else 0
    print(f"\nMax address_id after: {max_id_after}")
    
    # Verify sequence is working
    print("\nTesting sequence (simulating new address insertion):")
    next_val = db_handler.execute_query("SELECT nextval('address_address_id_seq');")
    if next_val:
        print(f"Next sequence value: {next_val[0][0]}")
    
    # Verify events table still has valid address_ids
    events_check = db_handler.execute_query("""
        SELECT COUNT(*) as total_events,
               COUNT(CASE WHEN address_id IS NOT NULL THEN 1 END) as events_with_address,
               MIN(address_id) as min_addr_id,
               MAX(address_id) as max_addr_id
        FROM events 
        WHERE address_id IS NOT NULL AND address_id > 0;
    """)
    
    if events_check:
        row = events_check[0]
        print(f"\nEvents verification:")
        print(f"  Total events: {row[0]}")
        print(f"  Events with address_id: {row[1]}")
        print(f"  Min address_id in events: {row[2]}")
        print(f"  Max address_id in events: {row[3]}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_reset_address_sequence()