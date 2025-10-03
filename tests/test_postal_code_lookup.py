"""Test postal code lookup after database consolidation."""
import sys
import os
import yaml
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")
sys.path.insert(0, 'src')

from db import DatabaseHandler

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Testing Postal Code Lookup")
print("=" * 60)

db_handler = DatabaseHandler(config)
print("✓ DatabaseHandler initialized\n")

# Test Victoria postal codes
tests = [
    ("1147 Quadra Street, Victoria", "V8W2K5", "1147", "Quadra"),
    ("833 Hillside Ave, Victoria", "V8T1Z7", "833", "Hillside"),
    ("Invalid address", "Z9Z9Z9", None, None),  # Should return None
]

passed = failed = 0

for location, postal, expect_no, expect_st in tests:
    print(f"Test: {location} ({postal})")
    result, addr_id = db_handler.populate_from_db_or_fallback(location, postal)
    
    if expect_no is None:
        if result is None:
            print(f"  ✓ Correctly returned None\n")
            passed += 1
        else:
            print(f"  ✗ Should be None, got: {result}\n")
            failed += 1
    else:
        if result and expect_no in result and expect_st in result:
            print(f"  ✓ {result}\n")
            passed += 1
        else:
            print(f"  ✗ Expected '{expect_no} {expect_st}', got: {result}\n")
            failed += 1

print("=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("✓ Postal code lookup working!" if failed == 0 else "✗ Tests failed")
