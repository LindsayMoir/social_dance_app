# Phase 11: Scraper Consolidation & Playwright Migration Plan

**Objective:** Eliminate code duplication across scrapers, migrate away from Scrapy, consolidate to Playwright-based architecture.

**Status:** Planning phase
**Estimated Effort:** 8-12 hours
**Expected Code Reduction:** 500-700 lines eliminated, cleaner architecture

---

## Current State Analysis

### Scrapers to Consolidate

1. **fb.py** - Facebook scraper (keep separate, specific logic)
2. **images.py** - Instagram scraper (keep separate, specific logic)
3. **ebs.py** - Eventbrite scraper (keep separate, specific logic)
4. **scraper.py** - General website scraper (USES SCRAPY - needs migration to Playwright)
5. **rd_ext.py** - Rueda/event extension scraper (consolidate into gen_scraper)
6. **read_pdfs.py** - PDF reader (consolidate into gen_scraper)

### Target Architecture

```
Specific Scrapers (keep separate):
├── fb.py              (Facebook-specific logic)
├── images.py          (Instagram-specific logic)
├── ebs.py             (Eventbrite-specific logic)
└── gen_scraper.py     (General website + PDFs + Rueda)
    ├── Inherits from BaseScraper
    ├── Uses PlaywrightManager
    ├── Uses PDFExtractor
    └── Uses GenericWebScraper

Shared Utilities (new):
├── base_scraper.py    (Abstract base class)
├── browser_utils.py   (Playwright setup & management)
├── text_utils.py      (HTML/text extraction)
├── auth_manager.py    (Login/session handling)
├── resilience.py      (Retry logic & error handling)
├── db_utils.py        (Database operations)
├── pdf_utils.py       (PDF parsing - extracted from read_pdfs.py)
└── url_nav.py         (URL tracking & navigation)
```

---

## Phase 11a: Create Shared Utility Modules

### 1. `src/browser_utils.py` (NEW)

**Purpose:** Centralized Playwright management

```python
class PlaywrightManager:
    """Unified Playwright browser management"""

    # Extracted from: fb.py, rd_ext.py, images.py, scraper.py

    @staticmethod
    def get_headers():
        """Returns standardized headers with user-agent"""
        # Lines from: rd_ext.py 38-42, images.py 78-81
        # Eliminates: 15 lines of duplication

    @staticmethod
    def get_viewport_size():
        """Returns standard viewport"""

    @staticmethod
    async def create_browser_context(page, cookies=None):
        """Create browser context with headers"""
        # Lines from: rd_ext.py 104-109, images.py 171-182
        # Eliminates: 20 lines of duplication

    @staticmethod
    def get_timeout():
        """Get standard timeout value"""

    @staticmethod
    def get_random_delay():
        """Anti-bot random delay"""
        # Lines from: fb.py 265-329, rd_ext.py throughout
        # Eliminates: 25 lines of duplication across files

    @staticmethod
    async def navigate_safe(page, url, timeout=None):
        """Navigate with error handling"""
        # Consolidates: fb.py 265-329, rd_ext.py 345-414, images.py similar
        # Eliminates: 40 lines of duplication
```

**Estimated Savings: ~100 lines**
**Files affected:** fb.py, rd_ext.py, images.py, scraper.py

---

### 2. `src/text_utils.py` (NEW)

**Purpose:** Unified text/HTML extraction

```python
class TextExtractor:
    """Text extraction from various sources"""

    @staticmethod
    def extract_from_html(html, beautifulsoup=True):
        """Extract clean text from HTML"""
        # Lines from: rd_ext.py 522-546, fb.py 427-450, images.py 547-565
        # Consolidates BeautifulSoup usage across all files
        # Eliminates: 60 lines of duplication

    @staticmethod
    def extract_links_from_html(html, base_url, same_domain_only=False):
        """Extract and normalize links"""
        # Lines from: scraper.py Scrapy parsing, rd_ext.py 511-546
        # Eliminates: 50 lines of duplication

    @staticmethod
    def extract_images_from_html(html, base_url):
        """Extract image URLs with validation"""
        # Lines from: images.py 452-516

    @staticmethod
    def clean_html(html):
        """Remove scripts, styles, etc."""
        # Lines from: images.py 548-549, rd_ext.py 625-626
        # Eliminates: 10 lines of duplication

    @staticmethod
    def find_keywords_in_text(text, keywords_list):
        """Find keywords in text"""
        # Lines from: scraper.py 162-191, fb.py 628-634
        # Eliminates: 20 lines of duplication
```

**Estimated Savings: ~140 lines**
**Files affected:** scraper.py, rd_ext.py, fb.py, images.py

---

### 3. `src/auth_manager.py` (NEW)

**Purpose:** Unified authentication across platforms

```python
class AuthenticationManager:
    """Platform-specific and generic login handling"""

    def __init__(self, db_handler=None):
        self.db = db_handler
        self.captcha_handler = CaptchaHandler()

    async def login_to_facebook(self, page, config=None):
        """Facebook login with CAPTCHA handling"""
        # Lines from: fb.py 151-247 (97 lines)
        # Eliminates: 97 lines, consolidates with other platforms

    async def login_to_website(self, page, username, password, selectors):
        """Generic website login with configurable selectors"""
        # Lines from: rd_ext.py 160-297 (137 lines)
        # Eliminates: 137 lines

    async def login_to_instagram(self, page, cookies=None):
        """Instagram login with session handling"""
        # Lines from: images.py 168-246 (79 lines)
        # Eliminates: 79 lines

    def _try_saved_session(self, auth_file):
        """Load saved session cookies"""
        # Consolidates cookie loading from all 3 files

    def _save_session(self, context, auth_file):
        """Save session cookies"""
        # Consolidates session saving across platforms

    async def _handle_captcha_if_needed(self, page):
        """Detect and handle CAPTCHA"""
        # Lines from: fb.py 208-247, rd_ext.py 236-277
        # Eliminates: 40 lines of duplication
```

**Estimated Savings: ~250 lines**
**Files affected:** fb.py, rd_ext.py, images.py

---

### 4. `src/resilience.py` (NEW)

**Purpose:** Retry logic with exponential backoff

```python
class RetryManager:
    """Centralized retry and error handling"""

    @staticmethod
    async def retry_async(func, max_attempts=3, base_delay=1, backoff=2):
        """Async function with exponential backoff"""
        # Lines from: rd_ext.py 354-414, images.py 281-319
        # Eliminates: 70 lines of duplication

    @staticmethod
    def retry_sync(func, max_attempts=3, base_delay=1, backoff=2):
        """Sync function with exponential backoff"""

    @staticmethod
    def is_page_crashed(error_msg):
        """Detect Playwright page crash"""
        # Lines from: rd_ext.py 396-410

    @staticmethod
    def is_rate_limited(error):
        """Detect 429/403 rate limiting"""

    @staticmethod
    def is_timeout_error(error):
        """Detect timeout errors"""
```

**Estimated Savings: ~70 lines**
**Files affected:** rd_ext.py, images.py

---

### 5. `src/url_nav.py` (NEW)

**Purpose:** URL tracking and navigation safety

```python
class URLNavigator:
    """URL navigation, tracking, and filtering"""

    def __init__(self, db_handler=None):
        self.visited = set()
        self.db = db_handler

    async def navigate_safe(self, page, url, timeout=None):
        """Navigate with safety checks"""
        # Consolidates: fb.py 265-329, rd_ext.py navigation

    def should_process(self, url, db_handler):
        """Check if URL should be processed"""
        # Uses: db_handler.should_process_url()

    def extract_same_domain_links(self, html, base_url):
        """Get links from same domain"""
        # Lines from: rd_ext.py 511-546
        # Eliminates: 35 lines

    def add_visited(self, url):
        """Track visited URL"""

    def is_visited(self, url):
        """Check if already visited"""

    def normalize_url(self, url):
        """Normalize URL for comparison"""
```

**Estimated Savings: ~50 lines**
**Files affected:** rd_ext.py, scraper.py, fb.py

---

### 6. `src/pdf_utils.py` (NEW)

**Purpose:** PDF extraction (extracted from read_pdfs.py)

```python
class PDFExtractor:
    """Extract events from PDF files"""

    def __init__(self, config):
        self.config = config
        self.llm_handler = LLMHandler()

    def extract_events_from_pdf(self, pdf_path):
        """Main PDF processing"""
        # Lines from: read_pdfs.py 76-165

    def register_custom_parser(self, source_name, parser_func):
        """Allow custom parsing for specific PDFs"""
        # Lines from: read_pdfs.py 31-46

    async def query_llm_for_pdf_content(self, pdf_text):
        """Process PDF text with LLM"""
        # Lines from: read_pdfs.py 234-255
```

**Estimated Savings: ~120 lines**
**Files affected:** read_pdfs.py (consolidation)

---

### 7. `src/db_utils.py` (NEW)

**Purpose:** Database operations (simplified wrapper)

```python
class DBWriter:
    """Simplified database writing"""

    def __init__(self, db_handler):
        self.db = db_handler

    def write_url(self, url, parent_url, source, keywords, relevant=False):
        """Write URL to database"""
        # Consolidates: 30+ duplicate calls across scrapers
        # Eliminates: 60 lines of tuple construction/error handling

    def write_event(self, df, url, parent_url, source, keywords):
        """Write event to database"""
        # Consolidates: calls in scraper.py, fb.py, rd_ext.py

    def track_statistics(self, scraper_name, start_time, stats_dict):
        """Record scraper statistics"""
        # Consolidates: fb.py 129-142 (comprehensive tracking)
```

**Estimated Savings: ~60 lines**
**Files affected:** All scrapers

---

## Phase 11b: Create Base Scraper Class

### `src/base_scraper.py` (NEW)

```python
from abc import ABC, abstractmethod
import logging

class BaseScraper(ABC):
    """Abstract base class for all scrapers"""

    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize handlers
        self.db_handler = self._get_db_handler()
        self.llm_handler = self._get_llm_handler()

        # Initialize utilities
        self.browser_mgr = PlaywrightManager()
        self.text_extractor = TextExtractor()
        self.url_navigator = URLNavigator(self.db_handler)
        self.auth_manager = AuthenticationManager()
        self.db_writer = DBWriter(self.db_handler)
        self.retry_manager = RetryManager()

        # Statistics
        self.stats = {
            'urls_processed': 0,
            'events_found': 0,
            'errors': 0,
            'start_time': datetime.now()
        }

    def _load_config(self, path):
        """Load YAML config"""
        # Lines from: all scrapers (duplicated 5+ times)

    def _get_db_handler(self):
        """Initialize database handler"""

    def _get_llm_handler(self):
        """Initialize LLM handler"""

    @abstractmethod
    async def scrape(self):
        """Main scraping method - implemented by subclasses"""

    async def start(self):
        """Start the scraper"""
        try:
            await self.scrape()
            self.logger.info(f"✓ {self.__class__.__name__} completed successfully")
        except Exception as e:
            self.logger.error(f"✗ {self.__class__.__name__} failed: {e}")
            raise
        finally:
            self._report_statistics()

    def _report_statistics(self):
        """Log statistics"""
        # Consolidates: fb.py 945-962, logs from other scrapers
```

**Estimated Savings: ~100 lines**

---

## Phase 11c: Migrate scraper.py to Playwright

### Current State
- Uses **Scrapy** (EventSpider)
- Async-based
- ~400 lines

### Migration Plan

**Remove Scrapy, rewrite as:**

```python
# src/scrapers/gen_scraper.py (placeholder name - you may rename)

class GenericWebScraper(BaseScraper):
    """General-purpose website scraper + PDF + Rueda events"""

    async def scrape(self):
        """Main scraping logic combining:
        - scraper.py (Scrapy-based crawling)
        - rd_ext.py (event extraction)
        - read_pdfs.py (PDF reading)
        """

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.config['crawling']['headless']
            )

            # Process websites (from scraper.py + rd_ext.py)
            await self._scrape_websites(browser)

            # Process PDFs (from read_pdfs.py)
            await self._process_pdfs(browser)

            await browser.close()

    async def _scrape_websites(self, browser):
        """Generic website crawling (from scraper.py + rd_ext.py)"""
        # Consolidates Scrapy logic into Playwright
        # URLs from config or database

        for url in self.get_urls_to_scrape():
            try:
                page = await browser.new_page()
                await self.url_navigator.navigate_safe(page, url)

                # Extract text and links
                content = await page.content()
                text = self.text_extractor.extract_from_html(content)
                keywords = self.text_extractor.find_keywords_in_text(text, ...)

                if keywords:
                    # Process with LLM
                    events = await self.llm_handler.process_text_for_events(text)
                    self.db_writer.write_event(events, url, ...)

                # Extract and process links (from scraper.py crawling logic)
                links = self.text_extractor.extract_links_from_html(
                    content, url, same_domain_only=True
                )

                for link in links:
                    if not self.url_navigator.is_visited(link):
                        await self._scrape_websites_recursive(browser, link)

                await page.close()
                self.stats['urls_processed'] += 1

            except Exception as e:
                self.logger.error(f"Error processing {url}: {e}")
                self.stats['errors'] += 1

    async def _process_pdfs(self, browser):
        """Extract events from PDFs (from read_pdfs.py)"""
        pdf_extractor = PDFExtractor(self.config)
        # Process PDF files

    def get_urls_to_scrape(self):
        """Get URLs from config or database"""
        # Logic from scraper.py + rd_ext.py
```

**Estimated Savings:**
- Remove Scrapy dependency (~50 lines)
- Consolidate rd_ext.py (~30% overlap eliminated)
- Consolidate read_pdfs.py (~40% overlap eliminated)
- **Total: ~200 lines eliminated**

---

## Phase 11d: Update Remaining Scrapers

### fb.py (Existing, minimal changes)

**Change from:**
```python
class FacebookEventScraper:
    def __init__(self):
        # 50+ lines of setup
        self.driver = ...
        self.config = ...
        # etc.

    def login_to_facebook(self):
        # 97 lines of login logic
```

**Change to:**
```python
class FacebookEventScraper(BaseScraper):
    async def scrape(self):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(...)
            page = await browser.new_page()

            # Use shared authentication
            await self.auth_manager.login_to_facebook(page, self.config)

            # Use shared utilities
            content = await page.content()
            text = self.text_extractor.extract_from_html(content)

            # Rest of Facebook-specific logic
```

**Savings: ~100 lines** (eliminate duplicate setup + login code)

---

### images.py (Existing, minimal changes)

**Change from:**
```python
class ImageScraper:
    def __init__(self):
        # 20+ lines Playwright setup
        # 60+ lines login logic
        # 50+ lines retry logic
```

**Change to:**
```python
class ImageScraper(BaseScraper):
    async def scrape(self):
        # Use shared PlaywrightManager
        # Use shared AuthenticationManager
        # Use shared RetryManager
        # Keep Instagram-specific logic
```

**Savings: ~130 lines** (eliminate duplication)

---

### ebs.py (Existing, minimal changes)

**Current structure:**
```python
class EventbriteScraper:
    async def eventbrite_search(self, query, ...):
        # Eventbrite-specific logic
        # Uses shared Playwright
        # Uses shared LLM calls
```

**Keep mostly as-is, add:**
```python
class EventbriteScraper(BaseScraper):
    async def scrape(self):
        # Use inherited PlaywrightManager
        # Use inherited RetryManager
        # Keep Eventbrite-specific search logic
```

**Savings: ~50 lines** (eliminate duplicate setup + retry)

---

## Summary of All Phases

| Phase | Component | Lines | Savings | Breaking Changes |
|-------|-----------|-------|---------|-----------------|
| 11a | browser_utils.py | 150 | 100 | None |
| 11a | text_utils.py | 180 | 140 | None |
| 11a | auth_manager.py | 280 | 250 | None |
| 11a | resilience.py | 100 | 70 | None |
| 11a | url_nav.py | 120 | 50 | None |
| 11a | pdf_utils.py | 150 | 120 | None |
| 11a | db_utils.py | 80 | 60 | None |
| 11b | base_scraper.py | 120 | 100 | Minor (inheritance) |
| 11c | gen_scraper.py | 300 | 200 | Yes (remove scraper.py) |
| 11d | Update fb.py | ~350 → ~250 | 100 | Minor |
| 11d | Update images.py | ~370 → ~240 | 130 | Minor |
| 11d | Update ebs.py | ~450 → ~400 | 50 | Minor |
| 11d | Remove read_pdfs.py | - | - | Consolidate |
| 11d | Remove rd_ext.py | - | - | Consolidate |
| **TOTAL** | - | **~3,440** | **~1,170** | **Moderate** |

---

## Implementation Order (Recommended)

### Step 1: Create Utility Modules (No breaking changes)
1. `browser_utils.py` - Start with PlaywrightManager
2. `text_utils.py` - TextExtractor
3. `auth_manager.py` - AuthenticationManager
4. `resilience.py` - RetryManager
5. `url_nav.py` - URLNavigator
6. `db_utils.py` - DBWriter
7. `pdf_utils.py` - PDFExtractor (extracted from read_pdfs.py)

**Testing:** All utilities have unit tests

### Step 2: Create Base Class
1. `base_scraper.py` - BaseScraper abstract class

### Step 3: Migrate Existing Scrapers (Safe)
1. Update `fb.py` - Inherit from BaseScraper
2. Update `images.py` - Inherit from BaseScraper
3. Update `ebs.py` - Inherit from BaseScraper
4. Test each individually

**Testing:** Run each scraper in isolation

### Step 4: Consolidate (Breaking changes)
1. Create `gen_scraper.py` - Consolidates scraper.py + rd_ext.py + read_pdfs.py
2. Update `pipeline.py` to call gen_scraper instead of individual files
3. Delete old files: scraper.py, rd_ext.py, read_pdfs.py

**Testing:** Full pipeline test

---

## Benefits After Phase 11

### Code Quality
- ✅ **Eliminate Scrapy** - Single Playwright-based architecture
- ✅ **Reduce duplication** - 1,170 lines consolidated
- ✅ **Consistent patterns** - All scrapers inherit from BaseScraper
- ✅ **Shared utilities** - Login, retry, text extraction in 1 place

### Maintainability
- ✅ **Fix bugs once** - Auth issues fixed in AuthenticationManager, used by all
- ✅ **Add features easily** - Add retry to all scrapers by updating RetryManager
- ✅ **Consistent error handling** - Same retry logic everywhere
- ✅ **Easier testing** - Utilities testable in isolation

### Performance
- ✅ **Same performance** - No algorithmic changes
- ✅ **Better resource usage** - Playwright pooling possible in future
- ✅ **No Scrapy overhead** - Lighter weight

### Developer Experience
- ✅ **New scrapers faster** - Just inherit BaseScraper + add logic
- ✅ **Clearer code** - Each class has single responsibility
- ✅ **Better documentation** - Utilities well-documented

---

## Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Breaking existing scrapers | Medium | High | Update each scraper incrementally, test separately |
| PDFs not extracting | Low | Medium | Comprehensive testing of pdf_utils.py |
| Auth failing | Medium | High | Thorough testing of AuthenticationManager against real sites |
| Performance regression | Low | Low | Benchmark before/after key operations |
| Playwright compatibility | Low | Low | Test on all supported platforms |

---

## Success Criteria

✅ Phase 11 Complete When:
- [ ] All 7 utility modules created and tested
- [ ] BaseScraper class working and documented
- [ ] fb.py, images.py, ebs.py updated and tested
- [ ] gen_scraper.py created and tested
- [ ] pipeline.py updated to use gen_scraper
- [ ] Old scraper files (scraper.py, rd_ext.py, read_pdfs.py) deleted
- [ ] Full pipeline tested end-to-end
- [ ] No Scrapy imports remaining in codebase
- [ ] 265+ unit tests still passing
- [ ] Estimated 1,100+ lines of duplicate code eliminated

---

**Document Version:** 1.0
**Created:** 2025-10-23
**Estimated Start:** After Phase 10 completion
**Estimated Duration:** 8-12 hours
**Priority:** MEDIUM (improves maintainability significantly)

