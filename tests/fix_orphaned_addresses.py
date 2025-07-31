#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def fix_orphaned_addresses():
    """Fix events that reference non-existent address IDs"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Fixing orphaned address references...")
    
    # Find orphaned events
    orphaned_events = db_handler.execute_query("""
        SELECT e.event_id, e.event_name, e.address_id, e.location
        FROM events e
        LEFT JOIN address a ON e.address_id = a.address_id
        WHERE e.address_id IS NOT NULL 
        AND e.address_id > 0 
        AND a.address_id IS NULL
        LIMIT 10;  -- Show first 10 for inspection
    """)
    
    print(f"\nFound orphaned events (showing first 10):")
    for event in orphaned_events:
        print(f"  Event {event[0]}: address_id={event[2]}, location='{event[3]}'")
    
    # Get count of all orphaned events
    orphaned_count = db_handler.execute_query("""
        SELECT COUNT(*) 
        FROM events e
        LEFT JOIN address a ON e.address_id = a.address_id
        WHERE e.address_id IS NOT NULL 
        AND e.address_id > 0 
        AND a.address_id IS NULL;
    """)
    
    total_orphaned = orphaned_count[0][0] if orphaned_count else 0
    print(f"\nTotal orphaned events: {total_orphaned}")
    
    if total_orphaned > 0:
        # Set orphaned address_ids to 0 (which means "no address")
        print("Setting orphaned address_ids to 0...")
        
        update_result = db_handler.execute_query("""
            UPDATE events 
            SET address_id = 0
            WHERE address_id IS NOT NULL 
            AND address_id > 0 
            AND address_id NOT IN (SELECT address_id FROM address);
        """)
        
        print(f"Updated {total_orphaned} events to have address_id = 0")
        
        # Verify fix
        remaining_orphaned = db_handler.execute_query("""
            SELECT COUNT(*) 
            FROM events e
            LEFT JOIN address a ON e.address_id = a.address_id
            WHERE e.address_id IS NOT NULL 
            AND e.address_id > 0 
            AND a.address_id IS NULL;
        """)
        
        remaining_count = remaining_orphaned[0][0] if remaining_orphaned else 0
        print(f"Remaining orphaned events: {remaining_count}")
    
    print("\nOrphaned address fix completed!")

if __name__ == "__main__":
    fix_orphaned_addresses()