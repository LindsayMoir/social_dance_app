# Run Results Tracking System

**Date:** October 24, 2025
**Status:** ✅ Production Ready

## Overview

The `run_results` table tracks execution statistics for every scraper run. This centralized system replaces the deprecated `runs` table and provides a unified way to record:

- Which script executed (file_name)
- When it started and finished
- How many events/URLs existed at start and end
- Net changes (new events/URLs added)
- Total execution time

## Database Schema

```sql
CREATE TABLE run_results (
    run_result_id SERIAL PRIMARY KEY,
    file_name TEXT,                    -- 'fb_v2.py', 'ebs.py', 'fb.py', etc.
    start_time_df TEXT,                -- ISO format start timestamp
    events_count_start INTEGER,        -- Events in DB at start
    urls_count_start INTEGER,          -- URLs in DB at start
    events_count_end INTEGER,          -- Events in DB at end
    urls_count_end INTEGER,            -- URLs in DB at end
    new_events_in_db INTEGER,          -- events_count_end - events_count_start
    new_urls_in_db INTEGER,            -- urls_count_end - urls_count_start
    time_stamp TIMESTAMP,              -- When row was inserted
    elapsed_time TEXT                  -- Duration (e.g., "0 days 00:30:45.123456")
);
```

## Using RunResultsTracker

### In Your Scraper

```python
from run_results_tracker import RunResultsTracker, get_database_counts

class MyScraperClass:
    def __init__(self, db_handler, ...):
        super().__init__(...)

        # Initialize run tracking
        file_name = 'my_scraper.py'
        self.run_results_tracker = RunResultsTracker(file_name, db_handler)

        # Get initial database counts
        events_count, urls_count = get_database_counts(db_handler)
        self.run_results_tracker.initialize(events_count, urls_count)

    async def main(self):
        start_time = datetime.now()

        try:
            # Your scraping logic here
            await self.scrape()
        finally:
            # Finalize tracking at the end
            events_count, urls_count = get_database_counts(self.db_handler)
            self.run_results_tracker.finalize(events_count, urls_count)

            # Calculate and write results
            elapsed_time = str(datetime.now() - start_time)
            self.run_results_tracker.write_results(elapsed_time)
```

## RunResultsTracker API

### Constructor
```python
tracker = RunResultsTracker(file_name: str, db_handler)
```

### Methods

#### `initialize(events_count: int, urls_count: int) -> None`
Called at the start of execution to capture baseline counts.

#### `finalize(events_count: int, urls_count: int) -> None`
Called at the end of execution with final database counts.

#### `write_results(elapsed_time: str) -> bool`
Inserts a row into the `run_results` table.

**Parameters:**
- `elapsed_time`: Duration as string (e.g., from `str(timedelta)`)

**Returns:**
- `True` if successfully written
- `False` if database error occurred

#### `get_summary() -> Dict[str, Any]`
Returns a dictionary with current tracking state:
```python
{
    'file_name': 'fb_v2.py',
    'start_time': '2025-10-24T14:30:00.123456',
    'events_count_start': 1000,
    'urls_count_start': 500,
    'events_count_end': 1050,
    'urls_count_end': 525,
    'new_events_in_db': 50,
    'new_urls_in_db': 25,
}
```

### Helper Functions

#### `get_database_counts(db_handler) -> Tuple[int, int]`
Queries the database for current event and URL counts.

**Returns:**
- Tuple of `(events_count, urls_count)`
- Returns `(0, 0)` if query fails

```python
events_count, urls_count = get_database_counts(db_handler)
```

## Migration from Old System

### Old System (DEPRECATED)
```python
self.run_name = f"Test Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
self.run_description = "Test Run Description"
self.start_time = datetime.now()

# ... later ...

def write_run_statistics(self):
    # Complex custom logic in each scraper
    run_data.to_sql("runs", db_handler.get_db_connection(), ...)
```

### New System (CURRENT)
```python
file_name = 'my_scraper.py'
self.run_results_tracker = RunResultsTracker(file_name, db_handler)
events_count, urls_count = get_database_counts(db_handler)
self.run_results_tracker.initialize(events_count, urls_count)

# ... later ...

events_count, urls_count = get_database_counts(db_handler)
self.run_results_tracker.finalize(events_count, urls_count)
elapsed_time = str(datetime.now() - start_time)
self.run_results_tracker.write_results(elapsed_time)
```

## Files Modified

**Scrapers Updated:**
- ✅ `src/fb.py` - Facebook scraper
- ✅ `src/fb_v2.py` - Facebook V2 scraper
- ✅ `src/ebs.py` - Eventbrite scraper
- ✅ `src/ebs_v2.py` - Eventbrite V2 scraper

**Database:**
- ✅ `src/db.py` - Replaced `runs` table with `run_results`
- ✅ `src/run_results_tracker.py` - New utility module

**Tests:**
- ✅ `tests/test_fb_v2_scraper.py` - Removed outdated write_run_statistics checks

## Querying Results

```python
import pandas as pd

# Read all run results
df = pd.read_sql("SELECT * FROM run_results", engine)

# Get results for specific scraper
fb_runs = pd.read_sql(
    "SELECT * FROM run_results WHERE file_name = 'fb_v2.py'",
    engine
)

# Get runs from last 24 hours
recent = pd.read_sql(
    "SELECT * FROM run_results WHERE time_stamp > NOW() - INTERVAL '1 day'",
    engine
)

# Get summary statistics
summary = pd.read_sql("""
    SELECT
        file_name,
        COUNT(*) as num_runs,
        SUM(new_events_in_db) as total_events_added,
        SUM(new_urls_in_db) as total_urls_added,
        AVG(EXTRACT(EPOCH FROM elapsed_time::interval)) as avg_duration_seconds
    FROM run_results
    GROUP BY file_name
    ORDER BY num_runs DESC
""", engine)
```

## Benefits of New System

✅ **Unified Schema** - All scrapers use identical table structure
✅ **No Code Duplication** - Single RunResultsTracker class replaces 5 separate methods
✅ **Consistent Tracking** - Standard initialization/finalization across all scrapers
✅ **Easier Maintenance** - Changes to tracking logic only need to be made once
✅ **Better Error Handling** - Centralized error handling in RunResultsTracker
✅ **CSV Compatible** - Data format matches existing `events_urls_diff.csv`

## Troubleshooting

### Issue: "Database handler not configured"
**Cause:** RunResultsTracker initialized with `db_handler=None`
**Solution:** Pass valid db_handler instance to constructor

### Issue: Write returns False
**Cause:** Database connection failed or table doesn't exist
**Solution:**
1. Check database connection
2. Run `create_tables()` in db.py to create table
3. Check logs for detailed error messages

### Issue: Counts don't match
**Cause:** Timing issue between queries
**Solution:**
- Ensure database is committed before reading
- Check for concurrent writes during query
- Add debug logging to track counts at initialization and finalization

## Future Enhancements

Potential improvements for future versions:

1. **Async Support** - Make write_results() async-compatible
2. **Metrics Export** - Built-in Prometheus metrics export
3. **Alert Thresholds** - Automatic alerts for anomalies
4. **Performance Tracking** - Per-URL/per-domain timing statistics
5. **Rollup Reports** - Daily/weekly aggregation views

---

**Last Updated:** October 24, 2025
**Version:** 1.0
**Status:** Production Ready
