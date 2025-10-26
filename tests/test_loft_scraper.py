#!/usr/bin/env python3
"""
Test script for The Loft scraper.

This script:
1. Initializes Playwright and rd_ext
2. Calls extract_calendar_events() to scrape individual event links
3. Processes each event through the LLM
4. Writes events to the database

Run with: python tests/test_loft_scraper.py
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
setup_logging('test_loft_scraper')
logger = logging.getLogger(__name__)

# Load config
config_path = Path("config/config.yaml")
with config_path.open() as f:
    config = yaml.safe_load(f)

LOFT_URL = "https://loftpubvictoria.com/events/month/"
LOFT_SOURCE = "The Loft"
LOFT_KEYWORDS = "swing, balboa, lindy hop, east coast swing, west coast swing, wcs"


@pytest.mark.asyncio
async def test_loft_scraper():
    """Main test function"""
    print("\n" + "=" * 80)
    print("THE LOFT SCRAPER TEST")
    print("=" * 80)
    print(f"\nTarget URL: {LOFT_URL}")
    print(f"Source: {LOFT_SOURCE}")
    print(f"Keywords: {LOFT_KEYWORDS}")
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

        # Extract Loft events
        logger.info("Extracting Loft events...")
        print("\nExtracting individual event links from The Loft events calendar...")
        event_data = await read_extract.extract_calendar_events(LOFT_URL, venue_name=LOFT_SOURCE)

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
                # Process through LLM with Loft-specific prompt
                logger.info(f"Processing event {idx}: {event_url}")
                llm_status = llm_handler.process_llm_response(
                    event_url,           # url
                    LOFT_URL,            # parent_url
                    event_text,          # extracted_text
                    LOFT_SOURCE,         # source
                    LOFT_KEYWORDS,       # keywords
                    prompt_type=LOFT_URL # Uses the_loft_prompt.txt
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
        print(f"Success rate:              {(events_processed/len(event_data)*100):.1f}%")
        print("=" * 80)

        # Verify data in database
        print("\nVerifying data in database...")
        try:
            import pandas as pd
            query = f"SELECT COUNT(*) as count FROM events WHERE source = '{LOFT_SOURCE}'"
            df = pd.read_sql(query, db_handler.conn)
            count = df['count'].values[0] if len(df) > 0 else 0
            print(f"✓ Total Loft events in database: {count}")
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
    print("\nStarting The Loft Scraper Test")
    print(f"Time: {datetime.now()}")

    # Run async test
    asyncio.run(test_loft_scraper())

    print(f"\nCompleted at: {datetime.now()}\n")
