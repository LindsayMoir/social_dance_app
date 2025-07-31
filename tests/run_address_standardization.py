#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def run_address_standardization():
    """Run the address standardization process"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Running address standardization...")
    
    # Show current state
    before_sample = db_handler.execute_query("""
        SELECT address_id, full_address, building_name 
        FROM address 
        WHERE building_name IS NOT NULL AND building_name != ''
        ORDER BY address_id 
        LIMIT 5;
    """)
    
    print("\nBefore standardization (sample):")
    print("-" * 120)
    for addr in before_sample:
        print(f"ID {addr[0]:3d}: Building='{addr[2]}', Full='{addr[1]}'")
    
    # Run standardization
    updated_count = db_handler.update_full_address_with_building_names()
    print(f"\n✅ Standardization complete! Updated {updated_count} addresses.")
    
    # Show results
    after_sample = db_handler.execute_query("""
        SELECT address_id, full_address, building_name 
        FROM address 
        WHERE building_name IS NOT NULL AND building_name != ''
        ORDER BY address_id 
        LIMIT 5;
    """)
    
    print("\nAfter standardization (sample):")
    print("-" * 120)
    for addr in after_sample:
        print(f"ID {addr[0]:3d}: Building='{addr[2]}', Full='{addr[1]}'")
    
    # Verify building names are at the start of full_address
    verification = db_handler.execute_query("""
        SELECT 
            COUNT(*) as total_with_building,
            COUNT(*) FILTER (WHERE LOWER(full_address) LIKE LOWER(building_name || '%')) as building_at_start
        FROM address 
        WHERE building_name IS NOT NULL AND building_name != '' AND full_address IS NOT NULL;
    """)
    
    if verification:
        total, at_start = verification[0]
        print(f"\n📊 Verification:")
        print(f"  Addresses with building_name: {total}")
        print(f"  Full_address starts with building_name: {at_start}")
        print(f"  Success rate: {(at_start/total*100):.1f}%" if total > 0 else "  No addresses to verify")

if __name__ == "__main__":
    run_address_standardization()