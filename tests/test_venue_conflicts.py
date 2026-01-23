"""
Test script for venue/time conflict resolution

This script tests the new venue/time conflict detection and resolution methods
added to dedup_llm.py.

Usage:
    python tests/test_venue_conflicts.py
"""

import sys
sys.path.insert(0, 'src')

from dedup_llm import DeduplicationHandler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_find_conflicts():
    """Test finding venue/time conflicts"""
    print("=" * 70)
    print("TEST 1: Find Venue/Time Conflicts")
    print("=" * 70)

    deduper = DeduplicationHandler()
    conflicts_df = deduper.find_venue_time_conflicts()

    if conflicts_df.empty:
        print("✓ No conflicts found (good!)")
    else:
        print(f"✓ Found {len(conflicts_df)} conflicts:")
        for idx, row in conflicts_df.head(5).iterrows():
            print(f"\n  Conflict {idx + 1}:")
            print(f"    Event 1: {row['event_name_1']} (ID: {row['event_id_1']}, Source: {row['source_1']})")
            print(f"    Event 2: {row['event_name_2']} (ID: {row['event_id_2']}, Source: {row['source_2']})")
            print(f"    Location: {row['location']}")
            print(f"    Date/Time: {row['start_date']} {row['start_time']}")

    return conflicts_df


def test_scrape_url():
    """Test URL scraping"""
    print("\n" + "=" * 70)
    print("TEST 2: Scrape URL Content")
    print("=" * 70)

    deduper = DeduplicationHandler()

    # Test with a known URL
    test_url = "https://www.google.com/calendar/event?eid=bTZkbThhbjlsaTdtdWRrNzduNWRyY25obWVfMjAyNjA5MjZUMDMwMDAwWiA3OWNmcDNhc2Y3Ym9kOXE0aWNnYzRydHRub0Bn"

    content = deduper.scrape_url_content(test_url)

    if content:
        print(f"✓ Successfully scraped URL")
        print(f"  Content length: {len(content)} characters")
        print(f"  Preview: {content[:200]}...")
    else:
        print("✗ Failed to scrape URL (may be expected for Google Calendar links)")


def test_analyze_conflict():
    """Test LLM analysis of a conflict"""
    print("\n" + "=" * 70)
    print("TEST 3: Analyze Conflict with LLM")
    print("=" * 70)

    deduper = DeduplicationHandler()
    conflicts_df = deduper.find_venue_time_conflicts()

    if conflicts_df.empty:
        print("✓ No conflicts to analyze")
        return

    # Analyze first conflict
    first_conflict = conflicts_df.iloc[0]

    print(f"\nAnalyzing conflict:")
    print(f"  Event 1: {first_conflict['event_name_1']}")
    print(f"  Event 2: {first_conflict['event_name_2']}")

    decision = deduper.analyze_conflict_with_llm(first_conflict)

    if decision:
        print(f"\n✓ LLM Analysis Complete:")
        print(f"  Correct Event ID: {decision['correct_event_id']}")
        print(f"  Incorrect Event ID: {decision['incorrect_event_id']}")
        print(f"  Confidence: {decision['confidence']}")
        print(f"  Reasoning: {decision['reasoning']}")
    else:
        print("✗ LLM analysis failed")


def test_resolve_conflicts_dry_run():
    """Test full conflict resolution in dry-run mode"""
    print("\n" + "=" * 70)
    print("TEST 4: Resolve Conflicts (Dry Run)")
    print("=" * 70)

    deduper = DeduplicationHandler()
    deleted_count = deduper.resolve_venue_time_conflicts(dry_run=True)

    print(f"\n✓ Dry run complete: Would have deleted {deleted_count} events")


if __name__ == "__main__":
    print("\nVENUE/TIME CONFLICT RESOLUTION TEST SUITE")
    print("=" * 70)

    try:
        # Test 1: Find conflicts
        conflicts_df = test_find_conflicts()

        # Test 2: Scrape URL (optional - may fail for some URLs)
        test_scrape_url()

        # Test 3: Analyze conflict with LLM (only if conflicts exist)
        if not conflicts_df.empty:
            test_analyze_conflict()

        # Test 4: Full dry run
        test_resolve_conflicts_dry_run()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
