# GeneralScraper (gen_scraper.py) Design Document
**Unified Event Extraction Pipeline Integrating rd_ext.py, read_pdfs.py, and scraper.py**

**Status:** Planning Phase
**Estimated Implementation:** 6-8 hours
**Branch:** `refactor/code-cleanup-phase2`

---

## Executive Summary

`gen_scraper.py` will be a unified event extraction system that integrates three separate data sources:

1. **ReadExtract** (rd_ext.py) - Calendar website event extraction
2. **ReadPDFs** (read_pdfs.py) - PDF document event extraction
3. **EventSpider** (scraper.py) - Web crawling with Scrapy

Instead of running three separate processes, `gen_scraper.py` provides a unified orchestration layer that:
- Manages all three extraction methods
- Coordinates shared resources (browser, database, LLM)
- Provides unified error handling and logging
- Deduplicates results across sources
- Tracks extraction statistics across all methods

---

## Current Architecture (Before gen_scraper)

```
Individual Processes:
├── rd_ext.py
│   ├── ReadExtract class (764 lines)
│   ├── Browser management
│   ├── Website calendar extraction
│   └── Database writes
│
├── read_pdfs.py
│   ├── ReadPDFs class (271 lines)
│   ├── HTTP PDF fetching
│   ├── PDF parsing
│   └── Database writes
│
└── scraper.py
    ├── EventSpider class (481 lines)
    ├── Scrapy framework
    ├── Web crawling
    └── Database writes

Problems:
- No coordination between extractors
- Duplicate browser/database connections
- No shared error handling
- Unclear extraction priority
- Resource inefficiency
```

---

## Proposed Architecture (With gen_scraper.py)

```
Unified Extraction Pipeline:
┌─────────────────────────────────────────────┐
│         GeneralScraper (gen_scraper.py)    │
│                                             │
│  - Unified orchestration layer             │
│  - Shared resource management              │
│  - Deduplication logic                     │
│  - Unified statistics & logging            │
└──────────┬──────────────┬──────────────┬───┘
           │              │              │
      ┌────▼─┐      ┌────▼──┐      ┌────▼────┐
      │rd_ext│      │read   │      │scraper  │
      │(V2)  │      │pdfs   │      │(V2)     │
      │      │      │       │      │         │
      │Uses: │      │Uses:  │      │Uses:    │
      │·Text │      │·PDF   │      │·Text    │
      │·URL  │      │·Text  │      │·URL     │
      │·Auth │      │·LLM   │      │·Retry   │
      │·DB   │      │·DB    │      │·Circuit │
      └──────┘      └───────┘      └─────────┘
           │              │              │
           └──────────────┴──────────────┘
                    │
              ┌─────▼──────┐
              │  Shared    │
              │ Resources: │
              │ · Browser  │
              │ · LLM      │
              │ · Database │
              │ · Logger   │
              └────────────┘
```

---

## Component Integration Points

### 1. ReadExtract Integration
**Current:** `rd_ext.py` → ReadExtract class (764 lines)
**Status:** Refactored as ReadExtractV2 (436 lines)

**Integration Points:**
- Browser management → PlaywrightManager
- Text extraction → TextExtractor
- Authentication → AuthenticationManager
- Error handling → RetryManager
- URL validation → URLNavigator
- Database operations → DBWriter

**Usage in gen_scraper:**
```python
self.read_extract = ReadExtractV2(config_path)
# Set shared database handler
self.read_extract.set_db_writer(self.shared_db_writer)

# Run extraction
results = await self.read_extract.extract_from_urls(calendar_urls)
self.stats['calendar_events'] += len(results)
```

### 2. ReadPDFs Integration
**Current:** `read_pdfs.py` → ReadPDFs class (271 lines)

**Integration Points:**
- PDF extraction → PDFExtractor (from BaseScraper)
- HTTP requests → Shared session management
- Text extraction → TextExtractor
- LLM processing → Shared LLMHandler
- Database operations → Shared DBWriter

**Usage in gen_scraper:**
```python
self.read_pdfs = ReadPDFsV2(config_path)  # Create V2 version
self.read_pdfs.set_db_writer(self.shared_db_writer)

# Run extraction
results = await self.read_pdfs.extract_from_urls(pdf_urls)
self.stats['pdf_events'] += len(results)
```

### 3. EventSpider Integration
**Current:** `scraper.py` → EventSpider class (481 lines)

**Integration Points:**
- Text extraction → TextExtractor
- URL validation → URLNavigator
- Error handling → RetryManager
- Circuit breaker → CircuitBreaker
- Database operations → Shared DBWriter

**Usage in gen_scraper:**
```python
self.spider = EventSpiderV2(config)
self.spider.set_db_writer(self.shared_db_writer)

# Run crawling
results = await self.spider.crawl(start_urls)
self.stats['web_events'] += len(results)
```

---

## GeneralScraper Class Design

### Core Structure

```python
class GeneralScraper(BaseScraper):
    """
    Unified event extraction pipeline combining multiple data sources.

    Integrates:
    - ReadExtract (calendar websites)
    - ReadPDFs (PDF documents)
    - EventSpider (web crawling)

    Provides:
    - Resource coordination
    - Deduplication
    - Unified statistics
    - Error handling
    - Progress tracking
    """

    def __init__(self, config_path="config/config.yaml"):
        super().__init__(config_path)

        # Initialize component extractors
        self.read_extract = ReadExtractV2(config_path)
        self.read_pdfs = ReadPDFsV2(config_path)
        self.spider = EventSpiderV2(config)

        # Set shared database writer
        for extractor in [self.read_extract, self.read_pdfs, self.spider]:
            extractor.set_db_writer(self.db_writer)

        # Deduplication tracking
        self.seen_events = set()  # Track event hashes
        self.extraction_source_map = {}  # Event ID → source

        # Statistics
        self.stats = {
            'calendar_events': 0,
            'pdf_events': 0,
            'web_events': 0,
            'duplicates_removed': 0,
            'total_unique': 0
        }
```

### Key Methods

```python
async def extract_from_all_sources(self):
    """
    Run extraction from all three data sources in parallel.
    """

async def extract_from_calendar_urls(self, urls):
    """Extract from calendar websites using ReadExtract."""

async def extract_from_pdfs(self, urls):
    """Extract from PDF documents using ReadPDFs."""

async def crawl_websites(self, start_urls):
    """Crawl websites using EventSpider."""

def deduplicate_events(self, events):
    """Remove duplicate events across sources."""

async def run_pipeline(self):
    """Execute full extraction pipeline."""

async def get_statistics(self):
    """Return extraction statistics."""
```

---

## Execution Flow

### Sequential Approach (Default)
```
1. Initialize GeneralScraper
2. Load configuration & handlers
3. Extract from calendar URLs (ReadExtract)
4. Extract from PDFs (ReadPDFs)
5. Crawl websites (EventSpider)
6. Deduplicate results
7. Write unique events to database
8. Report statistics
9. Cleanup resources
```

### Parallel Approach (Async)
```
1. Initialize GeneralScraper
2. Load configuration & handlers
3. Create tasks for all three sources
4. Run in parallel:
   - ReadExtract.extract_from_calendar_urls()
   - ReadPDFs.extract_from_urls()
   - EventSpider.crawl()
5. Wait for all to complete
6. Deduplicate results
7. Write unique events
8. Report statistics
```

---

## Deduplication Strategy

### Event Matching
Events are considered duplicates if they match on:
1. **Primary match:** Same URL + name + date
2. **Secondary match:** Similar name + date + location (fuzzy matching)
3. **Tertiary match:** Same name within 1-hour time window

### Deduplication Process
```python
def _create_event_hash(event):
    """Create unique hash for event."""
    return hashlib.md5(
        f"{event['URL']}{event['name']}{event['date']}".encode()
    ).hexdigest()

def deduplicate_events(self, events):
    """Remove duplicates and track sources."""
    unique_events = []
    for event in events:
        event_hash = self._create_event_hash(event)
        if event_hash not in self.seen_events:
            self.seen_events.add(event_hash)
            self.extraction_source_map[event_hash] = event['source']
            unique_events.append(event)
        else:
            self.stats['duplicates_removed'] += 1
    return unique_events
```

---

## Resource Management

### Shared Resources
```python
# Browser (managed by PlaywrightManager)
self.browser_manager.playwright
self.browser_manager.browser

# Database (managed by DBWriter)
self.db_writer.connection
self.db_writer.handlers

# LLM (managed by LLMHandler)
self.llm_handler

# Logger (unified logging)
self.logger
```

### Resource Lifecycle
```
1. Initialization (GeneralScraper.__init__)
   - Create browser instance
   - Connect to database
   - Initialize LLM handler

2. Extraction Phase
   - All three extractors share same browser/db/llm
   - Reduces overhead by ~60%

3. Cleanup Phase
   - Close browser connection
   - Commit database transactions
   - Release all resources
```

---

## Error Handling Strategy

### Circuit Breaker Pattern
```python
# Individual circuit breaker per extractor
self.read_extract.circuit_breaker
self.read_pdfs.circuit_breaker
self.spider.circuit_breaker

# Master circuit breaker for pipeline
self.circuit_breaker
```

### Retry Logic
```python
# Per-extractor retries via RetryManager
read_extract_task = self.retry_manager.execute_with_retry(
    self.read_extract.extract_from_urls(),
    max_retries=3
)

# Graceful degradation - continue if one source fails
try:
    calendar_results = await read_extract_task
except Exception as e:
    self.logger.error(f"Calendar extraction failed: {e}")
    calendar_results = []

# Proceed with other sources
```

---

## Performance Improvements

### Resource Efficiency
- **Before:** 3 separate browser instances + 3 database connections
- **After:** 1 shared browser instance + 1 database connection
- **Savings:** 66% reduction in resource overhead

### Execution Time
- **Sequential:** Sum of all three extraction times
- **Parallel:** Max(calendar_time, pdf_time, web_time)
- **Speedup:** Typically 2-3x faster with parallel execution

### Statistics
```
Example Run (before gen_scraper):
- ReadExtract: 450 seconds
- ReadPDFs: 200 seconds
- EventSpider: 600 seconds
Total: 1250 seconds (sequential)

Example Run (after gen_scraper with parallel):
- Expected time: ~620 seconds (max of three)
- Speedup: 2x
```

---

## Implementation Phases

### Phase 1: Create ReadPDFsV2
- Refactor ReadPDFs to use BaseScraper utilities
- Integrate PDFExtractor, TextExtractor, DBWriter
- Estimated: 2-3 hours

### Phase 2: Create GeneralScraper
- Design unified orchestration layer
- Implement deduplication logic
- Add statistics tracking
- Estimated: 3-4 hours

### Phase 3: Integration & Testing
- Test all three extractors through gen_scraper
- Verify deduplication accuracy
- Performance benchmarking
- Estimated: 1-2 hours

---

## Files to Create/Modify

### New Files
1. **src/gen_scraper.py** (500-600 lines)
   - GeneralScraper class
   - Orchestration logic
   - Deduplication

2. **src/read_pdfs_v2.py** (250-300 lines)
   - ReadPDFsV2 class using BaseScraper utilities

3. **GEN_SCRAPER_COMPLETION.md**
   - Implementation report
   - Statistics
   - Testing results

### Modified Files
- **No breaking changes** to existing files
- Original rd_ext.py, read_pdfs.py, scraper.py preserved

---

## Success Criteria

- ✅ GeneralScraper class fully functional
- ✅ All three extractors integrated and working
- ✅ Deduplication logic tested and verified
- ✅ Resource sharing working correctly
- ✅ Error handling robust across all sources
- ✅ Statistics accurately tracked
- ✅ Full documentation with examples
- ✅ Backward compatibility maintained

---

## Next Steps

1. Review and approve design
2. Implement ReadPDFsV2 (refactor read_pdfs.py)
3. Implement GeneralScraper orchestration
4. Integration testing across all sources
5. Performance benchmarking
6. Production rollout

---

**Document Version:** 1.0
**Date:** October 24, 2025
**Status:** Design Phase Complete, Ready for Implementation
