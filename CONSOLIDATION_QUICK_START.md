# Consolidation Initiative - Quick Start Guide

## TL;DR

**8 prioritized tasks identified | 800-1200 lines to save | 35-50 hours effort | 4-6 weeks to complete**

Priority 1 tasks are ready to start **today**. Focus on Task 1-3 first (14 hours total) for immediate impact.

---

## Priority 1: Start This Week (14 hours total, ready now)

### Task 1: Migrate ImageScraper to HandlerFactory (4 hours)
```
File: /src/images.py (783 → ~720 lines)
Impact: HIGH | Effort: 4h | Lines: 60-80
Status: READY NOW ✓

Current Issue:
  - Duplicate handler initialization (~60 lines of boilerplate)
  - Not using HandlerFactory (unlike 4 other scrapers)
  - Manual PlaywrightManager, AuthManager setup

Solution:
  - Call HandlerFactory.create_web_scraper_handlers()
  - Get pre-configured handlers for browser, text extraction, auth, DB
  - Remove ~60 lines of duplicate initialization

Expected Outcome:
  - 7% code reduction
  - 100% consistency with other scrapers (fb_v2, ebs_v2, rd_ext_v2)
  - Better error handling from factory defaults
```

### Task 2: Add Error Decorators to Repositories (4 hours)
```
Files: 9 repository files in /src/repositories/
Impact: HIGH | Effort: 3-4h | Lines: 40-60
Status: READY NOW ✓

Current Issue:
  - 13+ manual try-except blocks across repositories
  - Inconsistent error handling patterns
  - No recovery/retry logic

Solution:
  - Apply @resilient_execution decorator to all DB methods
  - Replaces 3-5 line try-except blocks with 1-line decorator
  - Automatically handles retries, logging, error classification

Example:
  FROM: try: ... except Exception as e: logger.error(...) return False
  TO:   @resilient_execution(...) def method(...)

Expected Outcome:
  - ~90% of try-except blocks replaced
  - Consistent error recovery across all repos
  - Better observability with structured logging
```

### Task 3: Consolidate DatabaseHandler Error Handling (6 hours)
```
File: /src/db.py
Impact: MEDIUM-HIGH | Effort: 5-6h | Lines: 80-120
Status: READY NOW ✓

Current Issue:
  - 13+ try-except blocks with inconsistent error handling
  - Connection errors not standardized
  - No unified retry strategy for DB operations

Solution:
  - Apply @resilient_execution to 8-10 DB methods
  - Add specific catch_exceptions for SQLAlchemy errors
  - Use CircuitBreaker for connection pool management

Methods to Update:
  - write_events_to_db()
  - write_url_to_db()
  - query_database()
  - get_events_by_similarity()
  - Plus 5-7 more critical DB operations

Expected Outcome:
  - Unified DB error handling
  - Automatic retry with exponential backoff
  - Better connection pool management
```

---

## Priority 2: Start After Priority 1 (19 hours total)

### Task 4: Create HTTPHandler (8 hours) - NEW FILE
```
Create: /src/http_handler.py
Impact: HIGH | Effort: 6-8h | Lines: 100-150
Dependencies: None (but do after P1)

Problem: 30 manual HTTP retry loops across 6+ files

Solution:
  class HTTPHandler:
      @http_retry(max_retries=3)
      def fetch_json(url): ...
      
      @http_retry(max_retries=3, retriable_statuses=[408, 429, 503])
      def fetch_html(url): ...

Use in: fb_v2.py, ebs_v2.py, rd_ext_v2.py, images.py

Expected: 100-150 lines of duplicate HTTP logic consolidated
```

### Task 5: Enhance AuthenticationHandler (6 hours)
```
File: /src/auth_manager.py (enhance) + scrapers (refactor)
Impact: MEDIUM-HIGH | Effort: 5-6h | Lines: 70-100

Problem: 70-100 lines of login code duplicated in 3-4 files

Solution:
  - Extract Instagram cookie loading logic from images.py
  - Extract session verification patterns
  - Add login retry with exponential backoff
  - Create OAuth/session refresh methods

Expected: Consolidated auth patterns across all scrapers
```

### Task 6: Create ConfigurationValidator (5 hours) - NEW FILE
```
Create: /src/configuration_validator.py
Impact: MEDIUM | Effort: 4-5h | Lines: 50-80

Problem: 8+ files with manual config validation (.get() chains)

Solution:
  @require_valid_config(['crawling', 'database', 'llm'])
  def __init__(self, config_path): ...

Expected: Prevent runtime config-related failures
```

---

## Priority 3: Strategic Work (Week 3, 7 hours)

### Task 7: LLM Error Recovery (7 hours)
```
File: /src/llm.py
Impact: MEDIUM | Effort: 6-7h | Lines: 60-100

Add:
  - @llm_retry decorator with exponential backoff
  - CircuitBreaker for LLM API endpoints
  - Fallback/cache mechanisms

Expected: Robust LLM operation handling with automatic recovery
```

---

## Priority 4: Advanced Work (Week 4, 10 hours)

### Task 8: AsyncOperationHandler (10 hours) - NEW FILE
```
Create: /src/async_handler.py
Impact: MEDIUM | Effort: 8-10h | Lines: 120-180

Consolidate async patterns from:
  - gen_scraper.py (manual semaphore management)
  - ebs_v2.py (async/await patterns)
  - rd_ext_v2.py (async browser management)
  - images.py (async event loop management)

Expected: 120-180 lines of async boilerplate removed
```

---

## Implementation Timeline

```
WEEK 1 (14 hours)
├─ Mon: Task 1 - ImageScraper (4h)
├─ Tue: Task 2 - Repository decorators (4h)
├─ Wed-Thu: Task 3 - DatabaseHandler (6h)
└─ STATUS: 0 days behind

WEEK 2 (19 hours)
├─ Mon-Tue: Task 4 - HTTPHandler (8h)
├─ Wed: Task 5 - AuthenticationHandler (6h)
├─ Thu: Task 6 - ConfigurationValidator (5h)
└─ STATUS: 0 days behind

WEEK 3-4 (23 hours)
├─ Week 3: Task 7 - LLM error recovery (7h)
├─ Week 4: Task 8 - AsyncOperationHandler (10h)
├─ Quick wins & testing (6h)
└─ STATUS: 0 days behind

TOTAL: 4-6 weeks for full implementation
```

---

## Files Created (3 new)
- `/src/http_handler.py` (Task 4)
- `/src/configuration_validator.py` (Task 6)
- `/src/async_handler.py` (Task 8)

## Files Refactored (16 existing)
**Priority 1:**
- `/src/images.py`
- `/src/repositories/event_repository.py` (+ 8 other repos)
- `/src/db.py`

**Priority 2:**
- `/src/auth_manager.py`
- `/src/fb_v2.py`, `/src/ebs_v2.py`, `/src/rd_ext_v2.py`

**Priority 3:**
- `/src/llm.py`, `/src/dedup_llm.py`
- `/src/gen_scraper.py`

---

## Success Criteria

| Metric | Target | Current | Success |
|--------|--------|---------|---------|
| Lines saved | 800-1200 | N/A | ~1000 lines |
| Error handling standardization | 90% | <20% | 45 of 50 try-except blocks replaced |
| HandlerFactory adoption | 100% | 67% (4/6 scrapers) | 6/6 scrapers |
| Decorator utilization | 90%+ | <20% | 40+ methods use decorators |

---

## Bonus Quick Wins (6 hours)
After Priority 1 tasks:
1. Standardize error logging (30 min)
2. Add DB query timeouts (1-2h)
3. Resource cleanup context manager (1h)
4. Structured logging for errors (2-3h)

---

## How to Start

### Today:
1. Read `/CONSOLIDATION_NEXT_TASKS.md` (detailed specs)
2. Read `/CONSOLIDATION_TASKS_SUMMARY.txt` (comprehensive analysis)
3. Review Task 1 code patterns in `handler_factory.py` and `fb_v2.py`

### Tomorrow:
1. Create feature branch: `git checkout -b feature/consolidation-phase2-task1`
2. Start Task 1: Migrate ImageScraper
3. Reference: `HandlerFactory.create_web_scraper_handlers()`

### This Week:
- Complete Tasks 1-3
- Run all tests: `pytest tests/`
- Create PR with summary of changes

---

## Key Resources

**Existing Consolidation Work:**
- `handler_factory.py` - Already has 3 factory methods ready to use
- `resilience.py` - 5 decorators ready (@http_retry, @resilient_execution, etc.)
- `logging_config.py` - Centralized logging already in place
- `base_scraper.py` - Base class with manager utilities

**Reference Implementations:**
- `fb_v2.py` - Shows correct HandlerFactory usage (lines 88-110)
- `ebs_v2.py` - Shows database writer integration (lines 50-68)
- `rd_ext_v2.py` - Shows async browser management (lines 66-86)

---

## Questions?

Refer to:
1. `/CONSOLIDATION_NEXT_TASKS.md` - Detailed specifications for each task
2. `/CONSOLIDATION_TASKS_SUMMARY.txt` - Comprehensive analysis & metrics
3. Existing files like `handler_factory.py`, `resilience.py` for patterns

---

## Expected Benefits

After completing all 8 tasks:
- **800-1200 fewer lines** of code to maintain
- **90%+ standardization** of error handling patterns
- **Better code reuse** across all scrapers
- **Improved observability** with structured logging & metrics
- **Faster development** with pre-built patterns and decorators
- **Reduced bugs** from standardized error recovery

---

**Total Time Investment: 35-50 hours**
**Expected ROI: Significant long-term maintainability improvement**
**Risk Level: LOW (building on proven consolidation patterns)**

---

Last updated: 2025-10-26
Created by: Claude Code Analysis
