#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def check_address_table():
    """Check the address table structure and sequence"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    # Check table structure
    table_info = db_handler.execute_query("""
        SELECT column_name, data_type, column_default, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'address' AND table_schema = 'public'
        ORDER BY ordinal_position;
    """)
    
    print("Address table structure:")
    for col in table_info:
        print(f"  {col[0]}: {col[1]} (default: {col[2]}, nullable: {col[3]})")
    
    # Check if there's a sequence associated with address_id
    sequence_info = db_handler.execute_query("""
        SELECT pg_get_serial_sequence('address', 'address_id') as sequence_name;
    """)
    
    if sequence_info and sequence_info[0][0]:
        print(f"\nSequence for address_id: {sequence_info[0][0]}")
    else:
        print("\nNo sequence found for address_id column")

        # Create the sequence
        print("Creating sequence for address table...")

        # Get current max ID
        max_id = db_handler.execute_query("SELECT MAX(address_id) FROM address;")
        current_max = max_id[0][0] if max_id and max_id[0][0] else 661
        db_handler.ensure_core_event_tables()
        db_handler.ensure_address_sequence(current_max + 1)
        print(f"Created sequence starting from {current_max + 1}")

if __name__ == "__main__":
    check_address_table()
