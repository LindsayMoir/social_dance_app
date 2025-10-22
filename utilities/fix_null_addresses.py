#!/usr/bin/env python3
"""
Script to fix NULL address_id in events table by processing location strings
through the address resolution system.
"""

import sys
import os
import logging
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import LLMHandler
from db import DatabaseHandler
import yaml

def fix_null_addresses():
    """Fix events with NULL address_id by processing their location data."""
    
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Initialize handlers
    llm_handler = LLMHandler()
    db_handler = DatabaseHandler(config)
    
    # Get events with NULL address_id that have location data using direct query
    query = """
    SELECT 
        event_id,
        event_name,
        CASE 
            WHEN location IS NOT NULL AND TRIM(location) <> '' THEN location
            WHEN description LIKE '%,%' THEN description
            ELSE NULL
        END as processable_location
    FROM events 
    WHERE address_id IS NULL
        AND (
            (location IS NOT NULL AND TRIM(location) <> '') 
            OR description LIKE '%,%'
        )
    ORDER BY event_id;
    """
    
    # Use execute_query to get results
    results = db_handler.execute_query(query)
    logging.info(f"Found {len(results)} events with NULL address_id but processable location data")
    
    updated_count = 0
    failed_count = 0
    
    for row in results:
        event_id = row[0]
        event_name = row[1] 
        location_str = row[2]
        
        if not location_str or len(location_str.strip()) < 10:
            logging.warning(f"Skipping event {event_id}: Location string too short")
            failed_count += 1
            continue
            
        try:
            logging.info(f"Processing event {event_id}: {event_name}")
            logging.info(f"Location string: {location_str[:100]}...")
            
            # Process the event through the updated address system
            test_event = {
                "event_name": event_name,
                "location": location_str,
                "url": f"https://example.com/event/{event_id}"
            }
            
            updated_event = db_handler.process_event_address(test_event)
            
            if updated_event and updated_event.get("address_id") and updated_event["address_id"] > 0:
                address_id = updated_event["address_id"]
                
                # Update the event with the resolved address_id
                update_query = """
                UPDATE events 
                SET address_id = :address_id
                WHERE event_id = :event_id
                """
                
                db_handler.execute_query(update_query, {
                    "address_id": address_id,
                    "event_id": event_id
                })
                
                logging.info(f"✓ Updated event {event_id} with address_id {address_id}")
                updated_count += 1
                
            else:
                logging.warning(f"✗ Failed to resolve address for event {event_id}")
                failed_count += 1
                
        except Exception as e:
            logging.error(f"✗ Error processing event {event_id}: {e}")
            failed_count += 1
    
    logging.info(f"Summary: {updated_count} events updated, {failed_count} events failed")
    
    # Check remaining NULL address_id events  
    remaining_query = "SELECT COUNT(*) FROM events WHERE address_id IS NULL"
    remaining_results = db_handler.execute_query(remaining_query)
    remaining_count = remaining_results[0][0]
    
    logging.info(f"Remaining events with NULL address_id: {remaining_count}")
    
    return updated_count, failed_count, remaining_count

if __name__ == "__main__":
    updated, failed, remaining = fix_null_addresses()
    print(f"\nResults:")
    print(f"  Updated: {updated}")
    print(f"  Failed: {failed}")  
    print(f"  Remaining NULL: {remaining}")