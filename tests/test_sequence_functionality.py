#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def test_sequence_functionality():
    """Test that the sequence works after reset"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Testing sequence functionality...")
    
    # Get current max address_id
    max_before = db_handler.execute_query("SELECT MAX(address_id) FROM address;")
    max_id_before = max_before[0][0] if max_before and max_before[0][0] else 0
    print(f"Current max address_id: {max_id_before}")
    
    # Test sequence reset (this should fix the sequence issue)
    print("\nRunning sequence reset to fix sequence...")
    result = db_handler.reset_address_id_sequence()
    print(f"Reset completed for {result} addresses")
    
    # Now test the sequence
    print("\nTesting sequence after reset:")
    
    # Get the sequence name
    sequence_query = "SELECT pg_get_serial_sequence('address', 'address_id');"
    sequence_result = db_handler.execute_query(sequence_query)
    
    if sequence_result and sequence_result[0][0]:
        sequence_name = sequence_result[0][0].split('.')[-1]
        print(f"Sequence name: {sequence_name}")
        
        # Test nextval
        next_val = db_handler.execute_query(f"SELECT nextval('{sequence_name}');")
        if next_val:
            print(f"Next sequence value: {next_val[0][0]}")
        
        # Test currval (current value after nextval)
        curr_val = db_handler.execute_query(f"SELECT currval('{sequence_name}');")
        if curr_val:
            print(f"Current sequence value: {curr_val[0][0]}")
    else:
        print("No sequence found!")
    
    print("\nSequence test completed!")

if __name__ == "__main__":
    test_sequence_functionality()