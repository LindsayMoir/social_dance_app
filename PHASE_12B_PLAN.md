# Phase 12B Plan: Facebook Scraper Refactoring
**Refactoring FacebookEventScraper with BaseScraper Utilities**

**Status:** Planning Phase
**Estimated Effort:** 6-8 hours
**Branch:** `refactor/code-cleanup-phase2`

---

## Overview

Phase 12B refactors the FacebookEventScraper class in `fb.py` to integrate with BaseScraper patterns and utility modules while maintaining Facebook-specific functionality.

### Key Challenge
FacebookEventScraper is the largest scraper (1019 lines) with specialized Facebook authentication and event extraction logic. The refactoring must preserve all FB-specific patterns while eliminating code duplication.

---

## Current FacebookEventScraper Analysis

### File: src/fb.py
- **Size:** 1019 lines
- **Class:** FacebookEventScraper()
- **Purpose:** Scrape Facebook events with Playwright browser automation

### Methods (15 total):
1. `__init__()` - Initialize with Playwright browser
2. `login_to_facebook()` - FB authentication (155 lines)
3. `normalize_facebook_url()` - URL normalization
4. `navigate_and_maybe_login()` - Navigation with login handling
5. `extract_event_links()` - Extract event URLs from search results
6. `extract_event_text()` - Extract content from event pages
7. `extract_relevant_text()` - Filter text based on keywords
8. `append_df_to_excel()` - Write to Excel file
9. `scrape_events()` - Main scraping orchestration
10. `process_fb_url()` - Process individual FB URL
11. `driver_fb_search()` - Search-based extraction driver
12. `driver_fb_urls()` - URL-based extraction driver (from database)
13. `driver_no_urls()` - Handle events without URLs
14. `write_run_statistics()` - Log statistics
15. `checkpoint_events()` - Checkpoint events to Excel

### Dependencies & Responsibilities:
```
Browser Management:       ✓ Playwright (can use PlaywrightManager)
Text Extraction:          ✓ BeautifulSoup (can use TextExtractor)
Authentication:          ✓ Facebook login (use AuthenticationManager)
Error Handling/Retries:   ✓ Manual try-catch (can use RetryManager)
URL Navigation:          ✓ Manual URL handling (can use URLNavigator)
Database Operations:     ✓ DatabaseHandler (can use DBWriter)
LLM Processing:          ✓ LLMHandler (keep shared)
```

---

## Integration Strategy

### Approach: Composition with FB-Specific Methods
```python
class FacebookScraperV2(BaseScraper):
    """
    Refactored FacebookEventScraper using BaseScraper utilities.

    Maintains all FB-specific functionality while leveraging
    utility managers for common patterns.
    """

    def __init__(self, config_path="config/config.yaml"):
        super().__init__(config_path)

        # Initialize utility managers from BaseScraper
        self.text_extractor = TextExtractor(config)
        self.url_navigator = URLNavigator(config)
        self.auth_manager = AuthenticationManager(config)
        self.retry_manager = RetryManager(config)
        self.circuit_breaker = CircuitBreaker(config)

        # FB-specific setup
        self.facebook_auth_path = get_auth_file('facebook')
        self.context = self.browser.new_context(
            storage_state=self.facebook_auth_path
        )
        self.page = self.context.new_page()

        # Statistics
        self.stats = {
            'unique_urls': 0,
            'urls_extracted': 0,
            'urls_with_keywords': 0,
            'events_written': 0
        }
```

---

## Key Refactoring Points

### 1. Browser Management
**Before:**
```python
self.playwright = sync_playwright().start()
self.browser = self.playwright.chromium.launch(headless=config['crawling']['headless'])
```

**After:**
```python
# Use PlaywrightManager from BaseScraper
self.browser = self.browser_manager.browser
self.playwright = self.browser_manager.playwright
```

### 2. Text Extraction
**Before:**
```python
soup = BeautifulSoup(self.page.content(), 'html.parser')
text = soup.get_text(separator="\n", strip=True)
```

**After:**
```python
content = await self.page.content()
text = self.text_extractor.extract_from_html(content)
```

### 3. Authentication
**Before:**
```python
# 150+ lines of manual login logic
async with self.page.expect_navigation():
    await self.page.fill('input[name="email"]', email)
    # ... manual FB login flow
```

**After:**
```python
success = await self.auth_manager.login(
    page=self.page,
    platform="facebook"
)
```

### 4. Error Handling
**Before:**
```python
try:
    await self.page.goto(url, timeout=15000)
except PlaywrightTimeoutError:
    logging.error(f"Timeout for {url}")
except Exception as e:
    logging.error(f"Error: {e}")
```

**After:**
```python
async def _navigate():
    await self.page.goto(url, timeout=15000)

success = await self.retry_manager.execute_with_retry(
    _navigate,
    max_retries=3
)
```

### 5. URL Validation
**Before:**
```python
# Manual URL validation and normalization
if url.startswith("http"):
    normalized = url
elif url.startswith("/"):
    normalized = f"https://facebook.com{url}"
else:
    normalized = f"https://facebook.com/{url}"
```

**After:**
```python
if self.url_navigator.is_valid_url(url):
    normalized = self.url_navigator.normalize_url(url)
```

---

## Code Reduction Estimate

### Original: ~1019 lines
### V2 Estimate: ~800-850 lines

**Expected reduction: 169-219 lines (16.6-21.5%)**

Breakdown:
- Browser initialization: -50 lines
- Authentication logic: -30 lines (use AuthenticationManager)
- Error handling: -40 lines (use RetryManager)
- Text extraction: -20 lines (use TextExtractor)
- URL handling: -20 lines (use URLNavigator)
- Documentation additions: +50 lines
- Net reduction: ~150-170 lines

---

## Implementation Phases

### Phase 1: Identify Refactoring Opportunities (30 min)
- [x] Analyze current code structure
- [x] Identify utility module integration points
- [x] Plan code reorganization

### Phase 2: Create FacebookScraperV2 (3-4 hours)
- Create base class structure
- Integrate utility managers
- Refactor key methods:
  - __init__() with utilities
  - login_to_facebook() → auth_manager
  - extract_event_text() → text_extractor
  - navigate_and_maybe_login() → retry_manager
  - extract_event_links() → url_navigator
- Add comprehensive docstrings

### Phase 3: Test & Verify (1-2 hours)
- Import verification
- Method validation
- Error handling verification
- Statistics tracking verification

### Phase 4: Commit & Document (30 min)
- Create completion report
- Document refactoring decisions
- Prepare for next phase

---

## Challenges & Mitigations

### Challenge 1: FB-Specific Authentication
**Issue:** Facebook login is complex with multiple steps and often requires human interaction
**Mitigation:**
- Preserve specialized FB login logic in separate method
- Use AuthenticationManager as wrapper where possible
- Keep facebook_auth.json storage state management

### Challenge 2: Excel File Output
**Issue:** Excel operations are FB-scraper specific
**Mitigation:**
- Keep append_df_to_excel() as-is (not part of utilities)
- This is DB output, not extraction logic

### Challenge 3: Multiple Driver Methods
**Issue:** driver_fb_search(), driver_fb_urls(), driver_no_urls() are complex orchestration methods
**Mitigation:**
- Refactor to use shared utilities where possible
- Keep orchestration logic intact
- Focus on core extraction method refactoring

### Challenge 4: Global Handler Dependencies
**Issue:** Code relies on global llm_handler and db_handler
**Mitigation:**
- Initialize through constructor
- Store as instance variables
- Use BaseScraper's db_writer where possible

---

## Success Criteria

- ✅ FacebookScraperV2 class created and documented
- ✅ All utility managers properly integrated
- ✅ Imports verify without errors
- ✅ Class instantiation works
- ✅ Key methods refactored with utilities
- ✅ Error handling via RetryManager working
- ✅ Backward compatibility maintained (original fb.py unchanged)
- ✅ Code reduced by 15-20%
- ✅ Full documentation with docstrings
- ✅ Changes committed to git

---

## Testing Strategy

### 1. Import Verification
```bash
python -c "from src.fb_v2 import FacebookScraperV2; print('✓')"
```

### 2. Class Instantiation
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from fb_v2 import FacebookScraperV2
scraper = FacebookScraperV2()
print('✓ FacebookScraperV2 instantiated')
"
```

### 3. Method Verification
- Verify all original methods are preserved
- Verify utility managers are accessible
- Verify error handling works

### 4. Integration Tests
- Run with GeneralScraper (can be added later)
- Verify statistics tracking
- Test error recovery

---

## Implementation Checklist

- [ ] Read and analyze entire fb.py file
- [ ] Create Phase_12B_PLAN.md (THIS FILE)
- [ ] Create src/fb_v2.py with FacebookScraperV2 class
- [ ] Add PlaywrightManager integration
- [ ] Add TextExtractor integration
- [ ] Add AuthenticationManager integration (where applicable)
- [ ] Add RetryManager integration
- [ ] Add CircuitBreaker integration
- [ ] Add URLNavigator integration
- [ ] Document all methods with comprehensive docstrings
- [ ] Test imports and instantiation
- [ ] Verify all utility managers are accessible
- [ ] Verify all statistics tracking works
- [ ] Run full test suite
- [ ] Create Phase 12B completion report
- [ ] Commit changes with appropriate message

---

## Estimated Timeline

| Step | Estimated Time |
|------|----------------|
| Analysis & planning | 30 min ✓ |
| Code creation | 3-4 hours |
| Integration & refactoring | 1.5-2 hours |
| Testing & verification | 1 hour |
| Documentation & commit | 30 min |
| **Total** | **6-8 hours** |

---

## Files to Create/Modify

### New Files:
1. **src/fb_v2.py** (800-850 lines)
   - FacebookScraperV2 class
   - Refactored methods using utilities
   - Full docstrings

2. **PHASE_12B_COMPLETION.md**
   - Implementation report
   - Code metrics
   - Refactoring decisions

### Modified Files:
- **No breaking changes** to existing files
- Original fb.py preserved for backward compatibility

---

## Architecture Comparison

### Before (fb.py):
```
FacebookEventScraper (1019 lines)
├── Manual browser management
├── Manual authentication
├── Manual text extraction
├── Manual error handling
├── Manual URL validation
└── Statistics tracking
```

### After (fb_v2.py):
```
FacebookScraperV2 (800-850 lines) extends BaseScraper
├── PlaywrightManager (shared)
├── AuthenticationManager (shared)
├── TextExtractor (shared)
├── RetryManager (shared)
├── CircuitBreaker (shared)
├── URLNavigator (shared)
└── Statistics tracking (improved)
```

---

## References

- **BaseScraper:** src/base_scraper.py (12K)
- **Utility Modules:** Documented in Phase 11A
- **Original FacebookScraper:** src/fb.py (1019 lines)
- **Phase 11B Pattern:** V2 classes extending utilities

---

**Document Version:** 1.0
**Date:** October 24, 2025
**Status:** Plan Complete, Ready for Implementation
