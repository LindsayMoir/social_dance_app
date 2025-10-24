# API Documentation: Scrapers & Utilities

**Date:** October 24, 2025
**Scrapers Documented:** 5 (ReadExtractV2, EventbriteScraperV2, FacebookScraperV2, ReadPDFsV2, GeneralScraper)
**Base Class:** BaseScraper

---

## Table of Contents

1. [BaseScraper](#basescraper-base-class)
2. [ReadExtractV2](#readextractv2)
3. [EventbriteScraperV2](#eventbritescraperv2)
4. [FacebookScraperV2](#facebookscraperv2)
5. [ReadPDFsV2](#readpdfv2)
6. [GeneralScraper](#generalscraper)
7. [Utility Managers](#utility-managers)

---

## BaseScraper (Base Class)

**File:** `src/base_scraper.py`
**Purpose:** Abstract base class for all scrapers
**Inheritance:** All scrapers extend BaseScraper

### Key Attributes

```python
class BaseScraper:
    # Browser management
    browser_manager: PlaywrightManager

    # Resource management
    db_writer: DBWriter
    circuit_breaker: CircuitBreaker

    # Utilities
    logger: Logger
    config: dict
```

### Key Methods

```python
async def scrape(self) -> pd.DataFrame:
    """Main scraping method - implemented by each scraper"""
    pass

def set_db_writer(self, db_handler):
    """Set database handler for writing events"""
    pass
```

---

## ReadExtractV2

**File:** `src/rd_ext_v2.py`
**Purpose:** Extract events from calendar websites
**Inheritance:** Extends BaseScraper
**Size:** 436 lines

### Initialization

```python
from rd_ext_v2 import ReadExtractV2

scraper = ReadExtractV2(config_path='config/config.yaml')
```

### Main Methods

#### `extract_from_urls(urls: list) -> pd.DataFrame`
Extract events from a list of calendar URLs.

```python
urls = ['https://gotothecoda.com/calendar', 'https://example.com/events']
events_df = await scraper.extract_from_urls(urls)
```

**Returns:** DataFrame with columns:
- `event_name`, `start_date`, `end_date`, `location`, `source`, etc.

#### `extract_from_calendar(url: str) -> list`
Extract events from a single calendar website.

```python
events = await scraper.extract_from_calendar('https://gotothecoda.com/calendar')
```

**Returns:** List of event dictionaries

#### `async scrape() -> pd.DataFrame`
Main scraping method (required by BaseScraper).

```python
result = await scraper.scrape()
print(f"Extracted {len(result)} events")
```

### Statistics

```python
stats = scraper.get_statistics() if hasattr(scraper, 'get_statistics') else scraper.stats
print(f"Events extracted: {stats.get('events_extracted', 0)}")
```

---

## EventbriteScraperV2

**File:** `src/ebs_v2.py`
**Purpose:** Scrape events from Eventbrite
**Inheritance:** Extends BaseScraper
**Size:** 448 lines

### Initialization

```python
from ebs_v2 import EventbriteScraperV2

scraper = EventbriteScraperV2(config_path='config/config.yaml')
```

### Main Methods

#### `eventbrite_search(keywords: list) -> pd.DataFrame`
Search Eventbrite for events by keywords.

```python
keywords = ['salsa', 'bachata', 'swing']
results = await scraper.eventbrite_search(keywords)
```

**Returns:** DataFrame with Eventbrite event results

#### `extract_event_urls(search_url: str) -> set`
Extract event URLs from Eventbrite search results.

```python
urls = scraper.extract_event_urls('https://www.eventbrite.com/...')
print(f"Found {len(urls)} event URLs")
```

**Returns:** Set of Eventbrite event URLs

#### `process_event(event_url: str) -> dict`
Process and extract data from a single Eventbrite event page.

```python
event_data = scraper.process_event('https://www.eventbrite.com/e/...')
print(f"Event: {event_data['name']}")
```

**Returns:** Dictionary with event details

### Statistics Tracking

```python
stats = scraper.stats
print(f"URLs processed: {stats['urls_processed']}")
print(f"Events extracted: {stats['events_extracted']}")
```

---

## FacebookScraperV2

**File:** `src/fb_v2.py`
**Purpose:** Scrape events from Facebook (Local execution only)
**Inheritance:** Extends BaseScraper
**Size:** 850 lines
**Status:** Production-ready, Standalone

### Initialization

```python
from fb_v2 import FacebookScraperV2

scraper = FacebookScraperV2(config_path='config/config.yaml')
# Automatically logs in to Facebook and initializes utilities
```

**Configuration Required:**
- Facebook credentials in environment or config
- `facebook_auth.json` for session persistence
- Browser configuration

### Authentication

#### `login_to_facebook() -> bool`
Log in to Facebook (automatic on init).

```python
# Called automatically during initialization
if scraper.login_to_facebook():
    print("Login successful")
```

**Note:** Complex process including:
- Manual login (visible browser mode)
- Automated login (headless mode)
- 2FA handling
- CAPTCHA detection

### Event Extraction

#### `extract_event_links(search_url: str) -> set`
Extract Facebook event links from a search URL.

```python
search_url = 'https://www.facebook.com/search/events/?q=salsa...'
event_links = scraper.extract_event_links(search_url)
print(f"Found {len(event_links)} event links")
```

**Returns:** Set of Facebook event URLs

#### `extract_event_text(link: str) -> str`
Extract full text content from a Facebook event page.

```python
text = scraper.extract_event_text('https://www.facebook.com/events/...')
print(f"Text length: {len(text)} characters")
```

**Returns:** Event page text or None

#### `extract_relevant_text(content: str, link: str) -> str`
Extract relevant portion of event content based on patterns.

```python
relevant = scraper.extract_relevant_text(full_text, event_url)
```

**Returns:** Filtered relevant text or None

### Drivers (Orchestration)

#### `driver_fb_search()`
Search Facebook for events using keywords and process them.

```python
scraper.driver_fb_search()
# Processes events from keyword searches
```

#### `driver_fb_urls()`
Process Facebook URLs from database and extract events.

```python
scraper.driver_fb_urls()
# Fetches URLs from database and processes them
```

### Statistics

#### `get_statistics() -> dict`
Get execution statistics.

```python
stats = scraper.get_statistics()
print(f"Unique URLs: {stats['unique_urls']}")
print(f"Events written to DB: {stats['events_written_to_db']}")
```

**Statistics Keys:**
- `unique_urls` - Number of unique URLs processed
- `total_url_attempts` - Total number of URL navigation attempts
- `urls_with_extracted_text` - URLs where text was successfully extracted
- `urls_with_found_keywords` - URLs matching search keywords
- `events_written_to_db` - Number of events saved to database

#### `log_statistics()`
Print formatted statistics to logs.

```python
scraper.log_statistics()
# Outputs: "=== Facebook Scraper Statistics ==="
```

### Utilities Available

```python
# All BaseScraper utilities are accessible
scraper.browser_manager       # Browser management
scraper.circuit_breaker       # Fault tolerance
scraper.text_extractor        # HTML text extraction
scraper.retry_manager         # Error handling with retries
scraper.url_navigator         # URL validation
```

---

## ReadPDFV2

**File:** `src/read_pdfs_v2.py`
**Purpose:** Extract events from PDF documents
**Inheritance:** Extends BaseScraper
**Size:** 461 lines

### Initialization

```python
from read_pdfs_v2 import ReadPDFsV2

scraper = ReadPDFsV2(config_path='config/config.yaml')
```

### Main Methods

#### `read_write_pdf() -> pd.DataFrame`
Read, parse, and write PDF events to database.

```python
result = scraper.read_write_pdf()
print(f"Extracted {len(result)} events from PDFs")
```

**Returns:** DataFrame with parsed PDF events

**Configuration:**
- `config['input']['pdfs']` - Path to CSV with PDF URLs
- PDF parsers registered for each source

#### Registered Parsers

```python
# Available parsers (registered via decorator)
@register_parser("Victoria Summer Music")
def parse_victoria_summer_music(pdf_file) -> pd.DataFrame:
    # Custom parsing logic
    pass

@register_parser("The Butchart Gardens Outdoor Summer Concerts")
def parse_butchart_gardens_concerts(pdf_file) -> pd.DataFrame:
    # Custom parsing logic (uses LLM)
    pass
```

### Statistics

```python
stats = scraper.stats
print(f"Events written: {stats.get('events_written', 0)}")
```

---

## GeneralScraper

**File:** `src/gen_scraper.py`
**Purpose:** Unified extraction pipeline (Calendar + PDF sources)
**Inheritance:** Extends BaseScraper
**Size:** 490 lines
**Note:** FacebookScraperV2 kept independent due to IP blocking

### Initialization

```python
from gen_scraper import GeneralScraper

scraper = GeneralScraper(config_path='config/config.yaml')
# Initializes ReadExtractV2, ReadPDFsV2, and optionally EventSpiderV2
```

### Extraction Methods

#### `async extract_from_calendars_async() -> pd.DataFrame`
Extract from calendar websites (ReadExtractV2).

```python
calendar_events = await scraper.extract_from_calendars_async()
```

#### `async extract_from_pdfs_async() -> pd.DataFrame`
Extract from PDF documents (ReadPDFsV2).

```python
pdf_events = await scraper.extract_from_pdfs_async()
```

#### `async extract_from_websites_async() -> pd.DataFrame`
Extract from websites via crawling (EventSpiderV2 if available).

```python
web_events = await scraper.extract_from_websites_async()
```

### Pipeline Execution

#### `async run_pipeline_parallel() -> pd.DataFrame`
Run all extractions in parallel.

```python
result = await scraper.run_pipeline_parallel()
print(f"Total unique events: {len(result)}")
```

**Benefits:** 2-3x faster than sequential

#### `async run_pipeline_sequential() -> pd.DataFrame`
Run extractions one at a time.

```python
result = await scraper.run_pipeline_sequential()
```

**Benefits:** Lower resource usage

### Deduplication

#### `deduplicate_events(events: list, source: str) -> list`
Remove duplicate events across sources.

```python
unique = scraper.deduplicate_events(events, source='calendar')
print(f"Removed {len(events) - len(unique)} duplicates")
```

**Deduplication Logic:**
- MD5 hash of (URL + name + date)
- Tracks source attribution
- Removes exact duplicates

### Statistics

```python
stats = scraper.get_statistics()
print(f"Calendar events: {stats['calendar_events']}")
print(f"PDF events: {stats['pdf_events']}")
print(f"Total unique: {stats['total_unique']}")
print(f"Duplicates removed: {stats['duplicates_removed']}")
```

---

## Utility Managers

### PlaywrightManager

**Purpose:** Centralized browser management
**Used by:** All scrapers

```python
manager = scraper.browser_manager
browser = manager.browser
page = manager.browser.new_page()
```

### TextExtractor

**Purpose:** HTML → text conversion
**Used by:** ReadExtractV2, ReadPDFsV2, FacebookScraperV2

```python
extractor = scraper.text_extractor
text = extractor.extract_from_html(html_content)
```

### RetryManager

**Purpose:** Resilient operation with retries
**Used by:** All scrapers

```python
retry_mgr = scraper.retry_manager
result = await retry_mgr.execute_with_retry(
    async_function,
    max_retries=3
)
```

### CircuitBreaker

**Purpose:** Fault tolerance and failure tracking
**Used by:** All scrapers

```python
cb = scraper.circuit_breaker
cb.record_failure()
if cb.is_open():
    print("Circuit breaker open, stopping operations")
```

### URLNavigator

**Purpose:** URL validation and normalization
**Used by:** FacebookScraperV2

```python
navigator = scraper.url_navigator
if navigator.is_valid_url(url):
    normalized = navigator.normalize_url(url)
```

---

## Common Usage Patterns

### Pattern 1: Simple Extraction

```python
from fb_v2 import FacebookScraperV2

async def scrape_facebook():
    scraper = FacebookScraperV2()
    scraper.driver_fb_search()
    scraper.driver_fb_urls()
    stats = scraper.get_statistics()
    print(f"Extracted {stats['events_written_to_db']} events")
    scraper.browser.close()
    scraper.playwright.stop()

# Run
import asyncio
asyncio.run(scrape_facebook())
```

### Pattern 2: Unified Pipeline

```python
from gen_scraper import GeneralScraper
import asyncio

async def unified_extraction():
    scraper = GeneralScraper()
    result = await scraper.run_pipeline_parallel()
    scraper.log_statistics()
    return result

# Run
events_df = asyncio.run(unified_extraction())
```

### Pattern 3: Facebook-Only (Recommended)

```python
from fb_v2 import FacebookScraperV2

def facebook_extraction():
    scraper = FacebookScraperV2()
    try:
        scraper.driver_fb_search()
        scraper.driver_fb_urls()
    finally:
        scraper.browser.close()
        scraper.playwright.stop()

# Run
facebook_extraction()
```

---

## Error Handling

### Common Exceptions

```python
try:
    scraper = FacebookScraperV2()
except ImportError:
    print("FacebookScraperV2 import failed - check dependencies")
except ValueError:
    print("Configuration error - check config/config.yaml")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Circuit Breaker Handling

```python
from fb_v2 import FacebookScraperV2

scraper = FacebookScraperV2()
try:
    scraper.driver_fb_search()
except Exception as e:
    scraper.circuit_breaker.record_failure()
    if scraper.circuit_breaker.is_open():
        print("Too many failures, stopping execution")
```

---

## Configuration Reference

**File:** `config/config.yaml`

```yaml
crawling:
  headless: true  # Run browser in headless mode
  scroll_depth: 5  # How many times to scroll
  urls_run_limit: 100  # Max URLs to process
  prompt_max_length: 8000  # Max prompt size

testing:
  status: false  # Enable test mode

input:
  data_keywords: 'data/keywords.csv'  # Keywords for search
  pdfs: 'data/pdfs.csv'  # PDF sources

constants:
  fb_base_url: 'https://www.facebook.com/search/'
  fb_location_id: '&t=events'  # Location for search
```

---

## Testing

### Running Tests

```bash
# FacebookScraperV2 tests
pytest tests/test_fb_v2_scraper.py -v

# All tests
pytest tests/ --ignore=tests/test_llm_schema_parsing.py -q
```

### Test Coverage

- **25 tests** for FacebookScraperV2
- **100% method coverage**
- **Utility manager integration verified**
- **Backward compatibility confirmed**

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 24, 2025 | FacebookScraperV2 refactored, Test suite created |
| - | - | ReadExtractV2, EventbriteScraperV2 refactored |
| - | - | GeneralScraper unified pipeline created |

---

## Support & Questions

**API Documentation Version:** 1.0
**Date:** October 24, 2025
**Status:** Production-Ready
**Scrapers Covered:** 5 (ReadExtract, Eventbrite, Facebook, PDF, General)

For detailed implementation examples, see respective scraper docstrings in source code.

