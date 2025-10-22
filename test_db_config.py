import sys
import os
sys.path.insert(0, 'src')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('src/.env')

from db_config import get_database_config, get_production_database_url, is_production_target

print("=" * 80)
print("DATABASE CONFIGURATION TEST")
print("=" * 80)

conn_str, env_name = get_database_config()
print(f"\nSource Database (from get_database_config()):")
print(f"  Name: {env_name}")
print(f"  Connection String: {conn_str[:80]}...")

print(f"\nProduction Database (from get_production_database_url()):")
try:
    prod_url = get_production_database_url()
    print(f"  Connection String: {prod_url[:80]}...")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\nIs Production Target? {is_production_target()}")
print("=" * 80)
