#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def check_final_results():
    """Check final results of address standardization"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Checking final results of address standardization...")
    
    # Show examples of addresses with building names
    examples = db_handler.execute_query("""
        SELECT address_id, building_name, full_address 
        FROM address 
        WHERE building_name LIKE '%Kitchen%' OR building_name LIKE '%Dance%' OR building_name LIKE '%Hall%'
        ORDER BY address_id 
        LIMIT 8;
    """)
    
    print(f"\nSample addresses now have building_name at the start of full_address:")
    print("-" * 100)
    for addr in examples:
        address_id, building_name, full_address = addr
        print(f"ID {address_id:3d}: {full_address}")
        print(f"      Building: \"{building_name}\"")
        print()
    
    # Check the specific original issue (addresses not starting with building_name)
    problem_check = db_handler.execute_query("""
        SELECT 
            COUNT(*) as total_addresses,
            COUNT(*) FILTER (WHERE building_name IS NOT NULL AND building_name != '') as with_building,
            COUNT(*) FILTER (WHERE building_name IS NOT NULL AND building_name != '' 
                           AND NOT (LOWER(full_address) LIKE LOWER(building_name || '%'))) as building_not_at_start
        FROM address;
    """)
    
    if problem_check:
        total, with_building, not_at_start = problem_check[0]
        print(f"📊 Final Statistics:")
        print(f"  Total addresses: {total}")
        print(f"  Addresses with building_name: {with_building}")
        print(f"  Building_name NOT at start of full_address: {not_at_start}")
        
        if not_at_start == 0:
            print(f"  ✅ SUCCESS: All addresses now have building_name at the start of full_address!")
        else:
            print(f"  ⚠️ {not_at_start} addresses still have issues")

if __name__ == "__main__":
    check_final_results()