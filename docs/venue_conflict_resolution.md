# Venue/Time Conflict Resolution

## Overview

The venue/time conflict resolution system automatically detects and resolves cases where two different events claim to occur at the same venue, date, and time. This is a data quality issue that can only be resolved by finding ground truth on the internet.

## Problem Description

**Conflict Pattern:**
- Same location (venue)
- Same date
- Same start time
- **Different event names**

**Example:**
```
Event 1: "Party- Salsa Night @ Dance City" (Salsa Caliente)
Event 2: "Swing City" (Victoria Latin Dance Association)
Location: Victoria Edelweiss Club, 108 Niagara St
Date: 2026-01-23
Time: 19:30:00
```

Only ONE of these events can be correct - they cannot both occur simultaneously at the same venue.

## Solution

### Implementation Location
- **File:** `src/dedup_llm.py`
- **Class:** `DeduplicationHandler`

### New Methods

#### 1. `find_venue_time_conflicts()`
Finds all events that have venue/time conflicts.

**SQL Logic:**
- Self-join events table on same date, location, and time
- Filter for different event names
- Only check future events (>= CURRENT_DATE)

**Returns:** DataFrame with paired conflicts

#### 2. `scrape_url_content(url)`
Scrapes content from event source URLs to verify information.

**Features:**
- Simple HTTP GET with requests library
- User-Agent header to avoid blocking
- 10-second timeout
- Returns text content (up to 3000 chars used for analysis)

#### 3. `analyze_conflict_with_llm(conflict_row)`
Uses LLM to analyze conflict and determine which event is correct.

**Process:**
1. Scrapes both event URLs
2. Constructs detailed prompt with:
   - Event details (name, source, dance style, description)
   - Scraped web content from both sources
3. LLM analyzes and returns JSON decision:
   ```json
   {
       "correct_event_id": 983,
       "incorrect_event_id": 1211,
       "confidence": "high",
       "reasoning": "Event 1 matches the official Edelweiss calendar..."
   }
   ```

#### 4. `resolve_venue_time_conflicts(dry_run=True)`
Main orchestration method that resolves all conflicts.

**Parameters:**
- `dry_run` (bool): If True, only logs decisions. If False, deletes incorrect events.

**Process:**
1. Find all conflicts
2. For each conflict:
   - Analyze with LLM
   - Record decision
   - Delete incorrect event (if not dry run)
3. Save decisions to CSV: `output/venue_time_conflict_resolutions.csv`

**Returns:** Number of events deleted

## Integration

The conflict resolution runs as part of the deduplication pipeline in `dedup_llm.py`:

```python
def driver(self):
    # ... existing deduplication steps ...

    # Resolve venue/time conflicts
    logging.info("Starting venue/time conflict resolution...")
    self.resolve_venue_time_conflicts(dry_run=False)

    # ... rest of pipeline ...
```

## Usage

### Manual Testing

```python
from dedup_llm import DeduplicationHandler

deduper = DeduplicationHandler()

# Find conflicts (read-only)
conflicts_df = deduper.find_venue_time_conflicts()
print(f"Found {len(conflicts_df)} conflicts")

# Resolve conflicts (dry run - no deletions)
deleted_count = deduper.resolve_venue_time_conflicts(dry_run=True)
print(f"Would delete {deleted_count} events")

# Resolve conflicts (live - deletes incorrect events)
deleted_count = deduper.resolve_venue_time_conflicts(dry_run=False)
print(f"Deleted {deleted_count} incorrect events")
```

### Automated (via pipeline)

```bash
python src/dedup_llm.py
```

This runs the full deduplication pipeline including venue/time conflict resolution.

## Output Files

### `output/venue_time_conflict_resolutions.csv`

Logs all conflict resolution decisions with columns:
- `event_id_correct`: ID of the event kept
- `event_id_incorrect`: ID of the event deleted
- `confidence`: LLM confidence level (high/medium/low)
- `reasoning`: Explanation of the decision
- `location`: Venue where conflict occurred
- `date`: Event date
- `time`: Event time

## Example Results

Based on current database (2026-01-23):

```
Found 23 venue/time conflicts
Most common conflict:
  - Victoria Edelweiss Club (4th Friday monthly)
  - "Salsa Night @ Dance City" vs "Swing City"
  - 8 occurrences from Jan-Sept 2026
```

## Error Handling

- If URL scraping fails: LLM analyzes based on available metadata only
- If LLM analysis fails: Logs error and skips that conflict
- If JSON parsing fails: Retries with cleaned response
- All errors logged, pipeline continues

## Best Practices

1. **Always test with dry_run=True first** to review decisions
2. **Check output CSV** before running live deletions
3. **Verify LLM confidence levels** - review "low" confidence decisions
4. **Monitor logs** for scraping failures or analysis errors

## Limitations

1. **Google Calendar URLs**: Some Google Calendar event URLs redirect/require auth - scraping may fail
2. **LLM Accuracy**: Depends on quality of scraped content and event metadata
3. **Single Conflict Resolution**: Currently resolves each conflict independently - doesn't detect patterns across multiple conflicts

## Future Enhancements

1. **Pattern Detection**: Identify recurring conflicts (e.g., monthly events) and resolve all at once
2. **Confidence Thresholds**: Require manual review for low-confidence decisions
3. **Alternative Scraping**: Use Playwright for dynamic content when simple requests fail
4. **Venue Validation**: Cross-check with venue websites/calendars directly
5. **User Feedback Loop**: Allow manual override of LLM decisions
