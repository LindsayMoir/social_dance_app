# Phase 11B: Scraper Integration with Utility Modules

**Status:** Planning
**Target:** Integrate Phase 11A utility modules (browser_utils, text_utils, auth_manager, resilience, url_nav, pdf_utils, db_utils) into main scrapers

## Executive Summary

Phase 11A created 7 utility modules and a BaseScraper abstract class. Phase 11B integrates these into existing scrapers (rd_ext.py, scraper.py, fb.py, ebs.py) to eliminate duplicate code and improve maintainability.

## Current State Analysis

### Scrapers and Their Architectures

| Scraper | File | Architecture | Key Classes | Status |
|---------|------|--------------|------------|--------|
| **General Web Scraper** | `rd_ext.py` | Playwright-based async | `ReadExtract` | 764 lines, complex |
| **Scrapy Framework** | `scraper.py` | Scrapy-based spider | `EventSpider` | 628 lines, framework-dependent |
| **Facebook Scraper** | `fb.py` | Custom Playwright/Requests | `FacebookScraper` | 1,264 lines, custom auth |
| **Eventbrite Scraper** | `ebs.py` | Requests-based | `EventbriteScraperRaw` | 347 lines, simpler |

### Utility Modules Created (Phase 11A)

| Module | Size | Purpose |
|--------|------|---------|
| `browser_utils.py` | 9.2K | PlaywrightManager - unified browser management |
| `text_utils.py` | 9.1K | TextExtractor - HTML text extraction |
| `auth_manager.py` | 21K | AuthenticationManager - platform-specific login |
| `resilience.py` | 13K | RetryManager + CircuitBreaker - error handling |
| `url_nav.py` | 12K | URLNavigator - URL validation and filtering |
| `pdf_utils.py` | 12K | PDFExtractor - PDF handling |
| `db_utils.py` | 13K | DBWriter - database operations |
| `base_scraper.py` | 12K | BaseScraper - abstract base class |

## Integration Strategy

### Phase 11B-1: ReadExtract (rd_ext.py) - PRIMARY TARGET
**Estimated Effort:** 6-8 hours

Current state: Uses custom Playwright initialization, custom auth logic, custom retry logic
Target: Extend BaseScraper, use PlaywrightManager, AuthenticationManager, RetryManager

**Changes:**
1. Inherit from BaseScraper instead of standalone class
2. Replace `extract_text_with_playwright()` with `text_extractor.extract_from_html()`
3. Replace `init_browser()` with `browser_manager.launch_browser()` and `browser_manager.new_context()`
4. Replace custom login logic with `auth_manager.login()` methods
5. Replace custom retry logic with `retry_manager.execute_with_retry()`
6. Use `url_navigator` for URL validation and tracking
7. Use `db_writer` for database operations
8. Use `pdf_extractor` for PDF extraction

**Expected reduction:** ~250 lines of code (~33% of current 764 lines)

### Phase 11B-2: ebs.py (EventbriteScraperRaw) - SECONDARY TARGET
**Estimated Effort:** 4-6 hours

Current state: Standalone requests-based scraper
Target: Extend BaseScraper, use utility managers

**Changes:**
1. Inherit from BaseScraper
2. Use `text_extractor` for HTML parsing
3. Use `url_navigator` for URL handling
4. Use `db_writer` for database operations
5. Use `retry_manager` for resilience

**Expected reduction:** ~100 lines of code (~29% of current 347 lines)

### Phase 11B-3: scraper.py (EventSpider) - OPTIONAL/FUTURE
**Estimated Effort:** 10-12 hours (complex due to Scrapy framework integration)

Note: Scrapy's architecture is fundamentally different from BaseScraper. Full integration would require:
- Wrapping BaseScraper utilities to work within Scrapy's spider lifecycle
- Careful handling of async/sync boundaries
- Potential creation of Scrapy middleware/extensions

**Recommendation:** May be better to create separate Scrapy-specific utilities in Phase 12+

### Phase 11B-4: fb.py (FacebookScraper) - OPTIONAL/FUTURE
**Estimated Effort:** 8-10 hours (complex due to custom auth and antibot measures)

Note: Facebook scraping involves complex anti-bot handling and session management. Would need:
- Custom auth_manager extensions for Facebook-specific flows
- Integration of CAPTCHA handling
- Preservation of current session persistence patterns

**Recommendation:** May be better addressed after Scraper.py in Phase 12+

## Proposed Implementation Order

### Sprint 1: ReadExtract Refactoring (6-8 hours)
1. Create new `ReadExtractV2` class extending BaseScraper
2. Migrate all methods from ReadExtract to use utility modules
3. Ensure backward compatibility through wrapper layer
4. Test all scraper functionality
5. Commit: "Phase 11B-1: Refactor ReadExtract to use BaseScraper utilities"

### Sprint 2: ebs.py Refactoring (4-6 hours)
1. Refactor EventbriteScraperRaw to extend BaseScraper
2. Integrate utility managers
3. Test EventBrite scraping workflows
4. Commit: "Phase 11B-2: Refactor EventbriteScraperRaw to use BaseScraper utilities"

### Sprint 3: Integration and Testing (3-4 hours)
1. Update all imports in dependent modules
2. Update tests for refactored scrapers
3. Run full test suite
4. Document changes
5. Commit: "Phase 11B-3: Update imports and tests for refactored scrapers"

### Phase 12 (Future): Advanced Scrapers
- Phase 12A: Scrapy integration (EventSpider in scraper.py)
- Phase 12B: Facebook scraper improvements (fb.py)

## Code Organization

After Phase 11B, file structure will be:

```
src/
├── base_scraper.py              (BaseScraper - abstract base)
├── browser_utils.py             (PlaywrightManager)
├── text_utils.py                (TextExtractor)
├── auth_manager.py              (AuthenticationManager)
├── resilience.py                (RetryManager, CircuitBreaker)
├── url_nav.py                   (URLNavigator)
├── pdf_utils.py                 (PDFExtractor)
├── db_utils.py                  (DBWriter)
│
├── rd_ext.py                    (ReadExtract - REFACTORED)
├── ebs.py                       (EventbriteScraperRaw - REFACTORED)
├── scraper.py                   (EventSpider - unchanged, Phase 12)
├── fb.py                        (FacebookScraper - unchanged, Phase 12)
│
└── ... other modules
```

## Benefits

### Code Reduction
- ReadExtract: ~250 lines removed (33%)
- ebs.py: ~100 lines removed (29%)
- **Total: ~350 lines of duplicate code eliminated**

### Maintainability
- Common patterns consolidated
- Single source of truth for browser management
- Single source of truth for error handling/retries
- Single source of truth for database operations

### Features Gained
- Built-in circuit breaker for fault tolerance
- Consistent retry logic across all scrapers
- Unified logging and statistics
- URL tracking and validation
- Better error messages

## Testing Strategy

### Phase 11B-1 (ReadExtract)
```bash
# Unit tests for refactored ReadExtract
pytest tests/test_coda_scraper.py -v
pytest tests/test_loft_scraper.py -v
pytest tests/test_duke_saloon_scraper.py -v

# Integration tests
pytest tests/integration/ -v -k "read_extract or rd_ext"
```

### Phase 11B-2 (ebs.py)
```bash
# Eventbrite-specific tests
pytest tests/ -v -k "eventbrite or ebs"
```

### Phase 11B-3 (Full integration)
```bash
# Full test suite
pytest tests/unit/ -v
pytest tests/ -v -k "scraper"
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Breaking existing tests | Create new refactored classes alongside old ones initially |
| Loss of functionality | Comprehensive manual testing of each scraper |
| Database integration issues | Mock database in unit tests, real DB in integration tests |
| Backward compatibility | Keep wrapper methods if needed for legacy code |

## Success Criteria

- ✅ All ReadExtract functionality preserved
- ✅ All ebs.py functionality preserved
- ✅ No regression in existing tests
- ✅ Code volume reduced by ~350 lines
- ✅ No new dependencies introduced
- ✅ Documentation updated for refactored modules
- ✅ New test coverage for utility integration

## Timeline

- **Phase 11B-1:** 1 day (6-8 hours)
- **Phase 11B-2:** 0.5 day (4-6 hours)
- **Phase 11B-3:** 0.5 day (3-4 hours)
- **Total:** 2 days (13-18 hours)

## References

- Phase 11A: Created 7 utility modules and BaseScraper
- Base architecture: `src/base_scraper.py`
- Utility interfaces in respective files

---

**Document Version:** 1.0
**Status:** Planning Phase
**Target Start:** Immediately after Phase 11A completion
