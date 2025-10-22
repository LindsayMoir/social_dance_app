#!/usr/bin/env python3
"""
cleanup_stale_instagram_urls.py

Utility script to delete stale Instagram and fbcdn URLs from the database.

Instagram CDN URLs (fbcdn.net) contain time-limited access tokens that expire after 24-48 hours.
This script removes old URLs that are no longer accessible to prevent wasted processing time.

Usage:
    python utilities/cleanup_stale_instagram_urls.py [--days N] [--dry-run]

Options:
    --days N    Delete URLs older than N days (default: 2)
    --dry-run   Show what would be deleted without actually deleting
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml
from sqlalchemy import text
from db import DatabaseHandler


def cleanup_stale_instagram_urls(days_threshold: int = 2, dry_run: bool = False):
    """
    Delete Instagram and fbcdn URLs older than the specified number of days.

    Args:
        days_threshold: Delete URLs older than this many days (default: 2)
        dry_run: If True, only show what would be deleted without actually deleting
    """
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize database handler
    db_handler = DatabaseHandler(config)

    print(f"\n{'DRY RUN: ' if dry_run else ''}Cleaning up Instagram/fbcdn URLs older than {days_threshold} days...")

    # Query to count URLs to be deleted
    count_query = text("""
        SELECT
            COUNT(*) as total_count,
            COUNT(*) FILTER (WHERE link LIKE '%instagram%') as instagram_count,
            COUNT(*) FILTER (WHERE link LIKE '%fbcdn.net%') as fbcdn_count
        FROM urls
        WHERE (link LIKE '%instagram%' OR link LIKE '%fbcdn.net%')
          AND time_stamp < (CURRENT_TIMESTAMP - INTERVAL ':days days')
    """)

    # Use proper parameter binding
    count_result = db_handler.conn.execute(
        text(f"""
            SELECT
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE link LIKE '%instagram%') as instagram_count,
                COUNT(*) FILTER (WHERE link LIKE '%fbcdn.net%') as fbcdn_count
            FROM urls
            WHERE (link LIKE '%instagram%' OR link LIKE '%fbcdn.net%')
              AND time_stamp < (CURRENT_TIMESTAMP - INTERVAL '{days_threshold} days')
        """)
    ).fetchone()

    total_count, instagram_count, fbcdn_count = count_result

    print(f"\nFound {total_count} stale URLs to delete:")
    print(f"  - Instagram URLs: {instagram_count}")
    print(f"  - fbcdn URLs: {fbcdn_count}")

    if total_count == 0:
        print("\nNo stale URLs to delete. Database is clean!")
        return

    if dry_run:
        print("\nDRY RUN: No URLs were deleted.")
        print("\nTo actually delete these URLs, run without --dry-run flag:")
        print(f"  python utilities/cleanup_stale_instagram_urls.py --days {days_threshold}")
        return

    # Perform the deletion
    delete_query = text(f"""
        DELETE FROM urls
        WHERE (link LIKE '%instagram%' OR link LIKE '%fbcdn.net%')
          AND time_stamp < (CURRENT_TIMESTAMP - INTERVAL '{days_threshold} days')
    """)

    result = db_handler.conn.execute(delete_query)
    db_handler.conn.commit()

    print(f"\n✅ Successfully deleted {result.rowcount} stale Instagram/fbcdn URLs from the database.")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean up stale Instagram and fbcdn URLs from the database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--days',
        type=int,
        default=2,
        help='Delete URLs older than this many days (default: 2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    if args.days < 1:
        print("Error: --days must be at least 1")
        sys.exit(1)

    cleanup_stale_instagram_urls(days_threshold=args.days, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
