#!/usr/bin/env python3

import psycopg2
import pandas as pd
from config.config import load_config

config = load_config()
conn = psycopg2.connect(
    host=config['database']['host'],
    port=config['database']['port'],
    database=config['database']['name'],
    user=config['database']['user'],
    password=config['database']['password']
)

# Check the specific Bonsor addresses
sql = '''
SELECT address_id, full_address, building_name, street_number, street_name, city, postal_code
FROM address 
WHERE address_id IN (1804, 15204)
ORDER BY address_id;
'''

df = pd.read_sql(sql, conn)
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

df2 = pd.read_sql(sql2, conn)
print('\nMatch check:')
print(df2.to_string())

conn.close()