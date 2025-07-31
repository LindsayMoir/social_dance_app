#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def verify_address_reset():
    """Verify the address reset worked correctly"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Verifying address ID reset results...")
    
    # Check address table
    addr_stats = db_handler.execute_query("""
        SELECT 
            COUNT(*) as total_addresses,
            MIN(address_id) as min_id,
            MAX(address_id) as max_id
        FROM address;
    """)
    
    if addr_stats:
        row = addr_stats[0]
        print(f"\nAddress table:")
        print(f"  Total addresses: {row[0]}")
        print(f"  Min address_id: {row[1]}")
        print(f"  Max address_id: {row[2]}")
        print(f"  Sequential? {row[2] == row[0] if row[1] == 1 else 'No (min != 1)'}")
    
    # Check events table
    events_stats = db_handler.execute_query("""
        SELECT 
            COUNT(*) as total_events,
            COUNT(CASE WHEN address_id IS NOT NULL AND address_id > 0 THEN 1 END) as events_with_address,
            MIN(address_id) as min_addr_id,
            MAX(address_id) as max_addr_id
        FROM events;
    """)
    
    if events_stats:
        row = events_stats[0]
        print(f"\nEvents table:")
        print(f"  Total events: {row[0]}")
        print(f"  Events with address_id: {row[1]}")
        print(f"  Min address_id: {row[2]}")
        print(f"  Max address_id: {row[3]}")
    
    # Check for any orphaned references
    orphaned = db_handler.execute_query("""
        SELECT COUNT(*) 
        FROM events e
        LEFT JOIN address a ON e.address_id = a.address_id
        WHERE e.address_id IS NOT NULL 
        AND e.address_id > 0 
        AND a.address_id IS NULL;
    """)
    
    if orphaned:
        print(f"\nOrphaned references: {orphaned[0][0]} events reference non-existent addresses")
    
    # Show some sample mappings
    sample_events = db_handler.execute_query("""
        SELECT e.event_id, e.event_name, e.address_id, a.full_address
        FROM events e
        JOIN address a ON e.address_id = a.address_id
        WHERE e.address_id IS NOT NULL
        ORDER BY e.event_id
        LIMIT 5;
    """)
    
    print(f"\nSample event-address mappings:")
    for event in sample_events:
        print(f"  Event {event[0]}: '{event[1][:30]}...' -> Address {event[2]}: '{event[3][:50]}...'")
    
    print(f"\nVerification complete!")

if __name__ == "__main__":
    verify_address_reset()