#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import DatabaseHandler

def test_build_full_address():
    """Test the build_full_address method"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Testing build_full_address method...")
    
    # Test cases
    test_cases = [
        {
            'name': 'Full address with building',
            'params': {
                'building_name': 'Gold Coast Ghal Kitchen',
                'street_number': '1009',
                'street_name': 'Boren',
                'street_type': 'Avenue',
                'city': 'Seattle',
                'province_or_state': 'WA',
                'postal_code': '98104',
                'country_id': 'US'
            },
            'expected': 'Gold Coast Ghal Kitchen, 1009 Boren Avenue, Seattle, WA 98104, US'
        },
        {
            'name': 'Address without building name',
            'params': {
                'street_number': '1009',
                'street_name': 'Boren',
                'street_type': 'Avenue',
                'city': 'Seattle',
                'province_or_state': 'WA',
                'postal_code': '98104',
                'country_id': 'US'
            },
            'expected': '1009 Boren Avenue, Seattle, WA 98104, US'
        },
        {
            'name': 'Minimal address',
            'params': {
                'street_number': '123',
                'street_name': 'Main',
                'street_type': 'St',
                'city': 'Victoria'
            },
            'expected': '123 Main St, Victoria'
        },
        {
            'name': 'Building name only',
            'params': {
                'building_name': 'Community Center',
                'city': 'Victoria',
                'province_or_state': 'BC',
                'country_id': 'CA'
            },
            'expected': 'Community Center, Victoria, BC, CA'
        }
    ]
    
    print("\nTest Results:")
    print("-" * 80)
    
    for test_case in test_cases:
        result = db_handler.build_full_address(**test_case['params'])
        expected = test_case['expected']
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        print(f"{status} {test_case['name']}")
        print(f"  Expected: '{expected}'")
        print(f"  Got:      '{result}'")
        if result != expected:
            print(f"  ❌ Mismatch!")
        print()

def preview_address_updates():
    """Preview what addresses would be updated"""
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create DatabaseHandler instance
    db_handler = DatabaseHandler(config)
    
    print("Previewing address updates...")
    
    # Get sample addresses
    sample_addresses = db_handler.execute_query("""
        SELECT address_id, full_address, building_name, street_number, street_name, 
               street_type, city, province_or_state, postal_code, country_id
        FROM address 
        ORDER BY address_id
        LIMIT 10;
    """)
    
    print(f"\nSample addresses and their standardized versions:")
    print("-" * 120)
    
    for addr in sample_addresses:
        address_id, current_full, building_name, street_number, street_name, street_type, city, province_or_state, postal_code, country_id = addr
        
        # Build standardized version
        new_full = db_handler.build_full_address(
            building_name=building_name,
            street_number=street_number,
            street_name=street_name,
            street_type=street_type,
            city=city,
            province_or_state=province_or_state,
            postal_code=postal_code,
            country_id=country_id
        )
        
        will_change = new_full != (current_full or "")
        status = "🔄 WILL UPDATE" if will_change else "✓ NO CHANGE"
        
        print(f"ID {address_id:3d}: {status}")
        print(f"  Current:  '{current_full}'")
        print(f"  New:      '{new_full}'")
        print()

if __name__ == "__main__":
    test_build_full_address()
    print("\n" + "="*80 + "\n")
    preview_address_updates()