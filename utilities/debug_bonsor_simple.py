#!/usr/bin/env python3

import os
import sys
import yaml

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from clean_up import CleanUp

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Create CleanUp instance
cleanup = CleanUp(config)

# Check the specific Bonsor addresses
print("Checking Bonsor addresses...")
sql = '''
SELECT address_id, full_address, building_name, street_number, street_name, city, postal_code
FROM address 
WHERE address_id IN (1804, 15204)
ORDER BY address_id;
'''

import pandas as pd
df = pd.read_sql(sql, cleanup.conn)
print('Bonsor addresses:')
print(df.to_string())

# Check if they would match in the duplicate query
sql2 = '''
SELECT a1.address_id as id1, a2.address_id as id2,
       a1.street_number as sn1, a2.street_number as sn2,
       a1.street_name as st1, a2.street_name as st2,
       LOWER(a1.street_number) = LOWER(a2.street_number) as num_match,
       LOWER(a1.street_name) = LOWER(a2.street_name) as name_match
FROM address a1
JOIN address a2 ON a1.address_id < a2.address_id
WHERE a1.address_id = 1804 AND a2.address_id = 15204;
'''

df2 = pd.read_sql(sql2, cleanup.conn)
print('\nMatch check:')
print(df2.to_string())

# Test the full duplicate detection query
print('\nTesting full duplicate detection query...')
df_dupes = cleanup.fetch_possible_duplicate_addresses()
print(f"Found {len(df_dupes)} potential duplicates")

# Check if Bonsor addresses still exist in database (should be only 1 remaining)
sql_check = '''
SELECT address_id, full_address, building_name, street_number, street_name, city, postal_code
FROM address 
WHERE address_id IN (1804, 15204)
ORDER BY address_id;
'''

df_remaining = pd.read_sql(sql_check, cleanup.conn)
print(f"\nBonsor addresses remaining in database: {len(df_remaining)}")
if not df_remaining.empty:
    print("Remaining Bonsor address(es):")
    print(df_remaining.to_string())
else:
    print("No Bonsor addresses found - both were deleted!")