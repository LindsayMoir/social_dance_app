"""
Utility script to upload auth JSON files to the auth_storage table.

Usage:
    python src/upload_auth_to_db.py

This will scan for *_auth.json files and upload them to the database.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from config_manager import ConfigManager # Consolidated database config get_database_config
from sqlalchemy import create_engine, text

# Setup centralized logging
from logging_config import setup_logging
setup_logging('upload_auth_to_db')
logger = logging.getLogger(__name__)


def upload_auth_file(engine, service_name: str, file_path: str):
    """Upload a single auth file to the database."""
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            auth_data = json.load(f)

        file_size = os.path.getsize(file_path)

        # Insert or update in database
        with engine.connect() as conn:
            # Convert dict to JSON string for JSONB column
            json_str = json.dumps(auth_data)

            conn.execute(text("""
                INSERT INTO auth_storage (service_name, auth_data, file_size_bytes, notes)
                VALUES (:service, :data, :size, :notes)
                ON CONFLICT (service_name)
                DO UPDATE SET
                    auth_data = EXCLUDED.auth_data,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    notes = EXCLUDED.notes
            """), {
                'service': service_name,
                'data': json_str,
                'size': file_size,
                'notes': f'Auto-uploaded from {file_path}'
            })
            conn.commit()

        logger.info(f"✓ Uploaded {service_name}: {file_size:,} bytes")

    except Exception as e:
        logger.error(f"✗ Failed to upload {service_name}: {e}")


def main():
    """Find and upload all *_auth.json files to database."""
    # Get database connection
    connection_string, env_name = ConfigManager.get_database_config()
    logger.info(f"Connecting to: {env_name}")
    engine = create_engine(connection_string)

    # Find all auth files in project root
    project_root = Path(__file__).parent.parent
    auth_files = list(project_root.glob('*_auth.json'))

    if not auth_files:
        logger.warning("No *_auth.json files found in project root")
        return

    logger.info(f"Found {len(auth_files)} auth file(s)")

    for file_path in auth_files:
        # Extract service name from filename (e.g., 'eventbrite_auth.json' -> 'eventbrite')
        service_name = file_path.stem.replace('_auth', '')
        upload_auth_file(engine, service_name, str(file_path))

    logger.info("\n✓ Upload complete!")
    logger.info("Auth files are now stored in the database and will sync across environments.")


if __name__ == '__main__':
    main()
