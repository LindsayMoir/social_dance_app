#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def check_sequence_name():
    """Check the actual sequence name for address table"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    # Find all sequences
    sequences = db_handler.execute_query("""
        SELECT schemaname, sequencename 
        FROM pg_sequences 
        WHERE schemaname = 'public';
    """)
    
    print("Available sequences:")
    for seq in sequences:
        print(f"  {seq[0]}.{seq[1]}")
    
    # Check what sequences are related to address table
    address_sequences = db_handler.execute_query("""
        SELECT schemaname, sequencename 
        FROM pg_sequences 
        WHERE schemaname = 'public' AND sequencename LIKE '%address%';
    """)
    
    print("\nAddress-related sequences:")
    for seq in address_sequences:
        print(f"  {seq[0]}.{seq[1]}")

if __name__ == "__main__":
    check_sequence_name()