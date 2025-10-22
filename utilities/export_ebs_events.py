"""
Script to export all Eventbrite events from the database to CSV.
This is a standalone script to manually trigger the export_events_to_csv function.
"""
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from db import DatabaseHandler
from ebs import export_events_to_csv

def main():
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Initialize database handler
    db_handler = DatabaseHandler(config)

    # Export events to CSV
    print("Exporting Eventbrite events to CSV...")
    export_events_to_csv(db_handler, config)
    print("Export complete!")

if __name__ == "__main__":
    main()
