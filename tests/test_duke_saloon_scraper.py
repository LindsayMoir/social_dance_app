#!/usr/bin/env python3
"""
Test script for The Duke Saloon scraper.

This script:
1. Initializes Playwright and rd_ext
2. Calls extract_calendar_events() to scrape individual event links
3. Processes each event through the LLM
4. Writes events to the database

Run with: python tests/test_duke_saloon_scraper.py
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import pytest

sys.path.insert(0, 'src')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('src/.env')

from logging_config import setup_logging
from rd_ext_v2 import ReadExtractV2
from llm import LLMHandler
from db import DatabaseHandler
import yaml

# Setup logging
setup_logging('test_duke_saloon_scraper')
logger = logging.getLogger(__name__)

# Load config
config_path = Path("config/config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)

DUKE_URL = "https://thedukesaloon.com/"
DUKE_SOURCE = "The Duke Saloon"
DUKE_KEYWORDS = "country swing, swing, live music"


@pytest.mark.asyncio
async def test_duke_saloon_scraper():
    """Main test function"""
    print("\n" + "=" * 80)
    print("THE DUKE SALOON SCRAPER TEST")
    print("=" * 80)
    print(f"\nTarget URL: {DUKE_URL}")
    print(f"Source: {DUKE_SOURCE}")
    print(f"Keywords: {DUKE_KEYWORDS}")
    print("\n" + "=" * 80)

    # Initialize handlers
    read_extract = ReadExtractV2(config_path=str(config_path))
    llm_handler = LLMHandler(config)
    db_handler = llm_handler.db_handler

    try:
        # Initialize browser
        logger.info("Initializing browser...")
        await read_extract.init_browser()
        print("✓ Browser initialized")

        # Extract Duke Saloon events
        logger.info("Extracting Duke Saloon events...")
        print("\nExtracting individual event links from The Duke Saloon events calendar...")
        event_data = await read_extract.extract_calendar_events(DUKE_URL, venue_name=DUKE_SOURCE)

        if not event_data:
            print("✗ No events extracted!")
            return

        print(f"✓ Extracted {len(event_data)} events")
        print("\n" + "-" * 80)
        print("PROCESSING EVENTS THROUGH LLM")
        print("-" * 80)

        # Process each event
        events_processed = 0
        events_failed = 0
        event_details = []

        for idx, (event_url, event_text) in enumerate(event_data, 1):
            print(f"\n[{idx}/{len(event_data)}] Processing: {event_url}")
            print(f"   Text length: {len(event_text)} characters")

            try:
                # Process through LLM with Duke Saloon-specific prompt
                logger.info(f"Processing event {idx}: {event_url}")
                llm_status = llm_handler.process_llm_response(
                    event_url,           # url
                    DUKE_URL,            # parent_url
                    event_text,          # extracted_text
                    DUKE_SOURCE,         # source
                    DUKE_KEYWORDS,       # keywords
                    prompt_type=DUKE_URL # Uses default.txt
                )

                if llm_status:
                    print(f"   ✓ LLM processing successful")
                    events_processed += 1
                    event_details.append({"url": event_url, "status": "success"})
                else:
                    print(f"   ✗ LLM processing returned False")
                    events_failed += 1
                    event_details.append({"url": event_url, "status": "failed"})

            except Exception as e:
                print(f"   ✗ Error processing event: {e}")
                logger.error(f"Error processing event {event_url}: {e}")
                events_failed += 1
                event_details.append({"url": event_url, "status": "error", "error": str(e)})

        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Total events extracted:    {len(event_data)}")
        print(f"Events processed:          {events_processed}")
        print(f"Events failed:             {events_failed}")
        if len(event_data) > 0:
            print(f"Success rate:              {(events_processed/len(event_data)*100):.1f}%")
        print("=" * 80)

        # Verify data in database
        print("\nVerifying data in database...")
        try:
            import pandas as pd
            query = f"SELECT COUNT(*) as count FROM events WHERE source = '{DUKE_SOURCE}'"
            df = pd.read_sql(query, db_handler.conn)
            count = df['count'].values[0] if len(df) > 0 else 0
            print(f"✓ Total Duke Saloon events in database: {count}")
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            print(f"✗ Could not verify database: {e}")

        print("\n✓ Test completed successfully!")

    except Exception as e:
        logger.exception(f"Test failed: {e}")
        print(f"\n✗ Test failed with error: {e}")

    finally:
        # Close browser
        logger.info("Closing browser...")
        await read_extract.close()
        print("\n✓ Browser closed")


if __name__ == "__main__":
    print("\nStarting The Duke Saloon Scraper Test")
    print(f"Time: {datetime.now()}")

    # Run async test
    asyncio.run(test_duke_saloon_scraper())

    print(f"\nCompleted at: {datetime.now()}\n")
