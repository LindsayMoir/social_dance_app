# Error Handling Quick Reference Guide

## Current Resilience Implementation Locations

### resilience.py - Core Module
**File:** `/mnt/d/GitHub/social_dance_app/src/resilience.py` (358 lines)

**Key Classes:**
1. RetryStrategy (Enum) - Lines 25-30
   - FIXED, LINEAR, EXPONENTIAL, EXPONENTIAL_WITH_JITTER

2. RetryManager - Lines 36-286
   - `calculate_delay()` - Lines 79-110
   - `retry_sync()` - Lines 112-149
   - `retry_async()` - Lines 151-190
   - `@retry_with_backoff()` decorator - Lines 192-212
   - `@retry_with_exponential_backoff()` decorator - Lines 214-230
   - `handle_error()` - Lines 232-255
   - `classify_http_status()` - Lines 261-273

3. CircuitBreaker - Lines 288-358
   - `can_execute()` - Lines 331-349
   - `record_success()` - Lines 313-317
   - `record_failure()` - Lines 319-329
   - `reset()` - Lines 351-357

---

## Error Handling by File (Detailed Locations)

### 1. fb_v2.py (950+ lines)
**Resilience Imports:** Line 51
**Usage Status:** MINIMAL - Imports but doesn't use decorators
**Error Handling Locations:**
- Lines 177-184: Manual try-except for login page navigation
- Lines 197-200: Login reload timeout handling
- Lines 219-224: Session state save error handling
- Lines 303-314: Navigate_and_maybe_login timeout chain (8+ nested try-except blocks)
- Line 443-445: Event text extraction timeout
- Lines 450-455: "See more" button click error handling

**Pattern:** Mix of PlaywrightTimeoutError and generic Exception
**Backoff Strategy:** Random timeout variation (line 181, 301, 327, 348)

### 2. images.py (600+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO - No resilience module usage
**Error Handling Locations - Retry Loops:**
- Lines 282-320: `_download_image_playwright()` - Exponential backoff retry (2^attempt + random)
  - Specific handling for 403 (line 300), 429 (line 305)
  - 3+ nested try-except blocks per attempt
  
- Lines 378-416: `download_image()` - requests.Session retry loop
  - Retry count: max_retries + 1 (line 378)
  - Error types: HTTPError (396), generic Exception (410)
  - Specific handling: 403 Forbidden (397), 429 Rate Limited (402)
  - Sleep logic: exponential backoff `(2**attempt) + random.uniform(0, 1)` (line 382)

**Error Handling Locations - Single Attempts:**
- Lines 140-142: `_load_instagram_cookies()` generic Exception
- Lines 151-167: `_verify_instagram_session()` generic Exception
- Lines 211-212: `_login_to_instagram()` generic Exception catch for pre-saved cookie failures
- Lines 437-439: `ocr_image_to_text()` bare except for Image.open
- Lines 446-450: OCR processing exception handling

**Pattern Duplication:** Download retry logic repeated twice (Playwright vs requests)

### 3. browser_utils.py (245+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO
**Error Handling Locations:**
- Lines 188-215: `navigate_safe()` - Manual retry loop
  - Exception types: PlaywrightTimeoutError (205), generic Exception (209)
  - Delay strategy: `apply_delay()` call with random 2-5 second range
  - Loop: `for attempt in range(max_retries)` (200)

- Lines 217-244: `navigate_safe_async()` - Async version of same logic
  - Uses `await asyncio.sleep(self.get_random_delay())` instead of `time.sleep`
  - Identical retry loop structure

**Timeout Management:**
- Lines 92-106: `get_timeout()` - Random variation ±50% of base
  - Calculation: `base_timeout + random.uniform(-variation, variation)` (105)
  - Default: 20000ms navigation, 10000ms wait

### 4. read_pdfs_v2.py (462 lines)
**Resilience Imports:** Line 36 (RetryManager, CircuitBreaker)
**Usage Status:** MINIMAL - Stores instance but doesn't use retry methods
**Error Handling Locations:**
- Lines 179-210: `read_write_pdf()` loop with broad error catching
  - Catches general Exception (207)
  - Calls `self.circuit_breaker.record_failure()` (209) but never checks `can_execute()`

- Lines 274-287: `_fetch_and_parse_pdf_sync()` PDF download
  - Checks for 404 status (276)
  - Calls `raise_for_status()` (279)
  - Generic exception handling (284-286)
  - Calls `self.circuit_breaker.record_failure()` (286)

**Pattern:** Uses CircuitBreaker but not RetryManager decorators

### 5. base_scraper.py (371 lines)
**Resilience Imports:** Line 31 (RetryManager, CircuitBreaker)
**Usage Status:** PARTIAL - Creates instances but limited use
**Error Handling Locations:**
- Line 68: RetryManager instantiation (but never called)
- Line 94: CircuitBreaker instantiation
- Line 143-146: `can_execute()` check using circuit breaker state
- Line 332: Logs circuit breaker state in statistics
- Lines 249-256: `write_events_to_db()` exception handling
  - Catches Exception (253)
  - Calls `record_failure()` (255)

**Pattern:** Inconsistent - checks circuit breaker state but doesn't prevent execution on open state

### 6. auth_manager.py (500+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO
**Error Handling Locations - Timeout Handling:**
- Lines 86-87: `login_to_facebook_sync()` - Navigate with timeout
- Lines 162-164: Login page navigation with timeout and generic exception catch
- Lines 167-172: Selector wait with timeout and specific PlaywrightTimeoutError handling
- Lines 178-181: Form submission and wait-for-timeout

**Pattern:** All timeout handling done with explicit parameters, no centralized decorator

### 7. pipeline.py (400+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO
**Error Handling Locations:**
- Lines 159-164: File move with broad exception handling
- Lines 188-244: Subprocess retry loop for database operations
  - Loop structure: `for attempt in range(1, attempts + 1)` (not using RetryManager)
  - No exponential backoff, just linear retry
  - CalledProcessError handling (191, 242)

### 8. db.py (1000+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO
**Error Handling Locations - Database Operations:**
- Lines 249-256: Broad generic Exception catching in database write operations
- Multiple instances of `except Exception as e:` without specific error classification
- SQLAlchemyError imported (line 26) but rarely caught specifically

**Pattern:** Reactive error handling without proactive retry strategy

### 9. llm.py (600+ lines)
**Resilience Imports:** None
**Usage Status:** ZERO
**Error Handling Locations:**
- Multiple generic `except Exception as e:` blocks
- Timeout specified for OpenAI (line 117: 60.0 seconds)
- Timeout specified for Mistral (line 123: 60000 milliseconds)
- No retry logic for API calls

---

## Error Handling Pattern Summary

### Retry Loop Pattern (Used in 3+ files)
```python
# Pattern A: Time-based
for attempt in range(max_retries):
    try:
        operation()
    except SpecificError as e:
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay = calculate_next_delay(attempt)
```

### Timeout Pattern (Used in 5+ files)
```python
# Pattern B: Playwright
try:
    page.goto(url, timeout=random_timeout())
except PlaywrightTimeoutError:
    # retry or fallback
except Exception:
    pass  # generic handling
```

### HTTP Error Pattern (Used in 2-3 files)
```python
# Pattern C: Specific status handling
if response.status_code == 403:
    handle_forbidden()
elif response.status_code == 429:
    handle_rate_limit()
```

---

## Consolidation Opportunities by Priority

### HIGH PRIORITY - Quick Wins

#### 1. HTTP Request Retry Decorator
**Current Code Locations:** 
- images.py lines 378-416
- images.py lines 282-320
- read_pdfs_v2.py lines 274-287

**Duplication:** ~70 lines of similar retry logic

**Proposed Replacement:**
```python
@http_retry(max_retries=3, backoff_strategy=EXPONENTIAL_WITH_JITTER)
def download_resource(url):
    return requests.get(url, timeout=30)
```

#### 2. Timeout Handler Decorator
**Current Code Locations:**
- browser_utils.py lines 188-244
- auth_manager.py lines 162-181
- fb_v2.py lines 177-351 (scattered)
- images.py lines 375, 386

**Duplication:** ~80 lines of navigation/timeout logic

**Proposed Replacement:**
```python
@with_timeout(timeout_ms=20000, variation_percent=50, max_retries=2)
async def navigate_url(page, url):
    await page.goto(url, wait_until='networkidle')
```

### MEDIUM PRIORITY - Core Improvements

#### 3. Resilient Execution Decorator
**Current Code Locations:**
- fb_v2.py: 6 instances of broad Exception handling
- images.py: 4 instances
- read_pdfs_v2.py: 3 instances
- db.py: Multiple instances

**Total Duplication:** ~120 lines

#### 4. Database Operation Decorator
**Current Code Locations:**
- db.py: Scattered exception handling without retry strategy

### LOW PRIORITY - Specialized

#### 5. Parser Error Decorator
**Current Code Locations:**
- llm.py: JSON parsing without retry
- gen_scraper.py: Parse error handling

---

## Testing Coverage Gaps

### Files with Tests:
- test_fb_v2_scraper.py

### Files WITHOUT error handling tests:
- images.py (retry logic untested)
- browser_utils.py (navigate_safe untested)
- pipeline.py (subprocess retry untested)
- auth_manager.py (timeout handling untested)

**Recommendation:** Add test cases for:
1. Retry count validation
2. Backoff calculation accuracy
3. Circuit breaker state transitions
4. Timeout behavior on slow operations
5. HTTP error code handling

---

## Implementation Checklist

### Step 1: Extend resilience.py
- [ ] Add `@http_retry()` decorator
- [ ] Add `@with_timeout()` decorator
- [ ] Add `@resilient_execution()` decorator
- [ ] Add HTTP error classification enhancement
- [ ] Add RetryMetrics class for tracking

### Step 2: Migrate browser_utils.py
- [ ] Replace navigate_safe() retry loop with decorator
- [ ] Replace navigate_safe_async() with decorator
- [ ] Add tests for timeout variation

### Step 3: Migrate images.py
- [ ] Replace _download_image_playwright() retry with decorator
- [ ] Replace download_image() retry loop with decorator
- [ ] Consolidate 403/429 handling
- [ ] Add tests

### Step 4: Migrate auth_manager.py
- [ ] Apply @with_timeout() decorator to login methods
- [ ] Consolidate timeout handling
- [ ] Add tests

### Step 5: Migrate pipeline.py
- [ ] Replace subprocess retry with @http_retry() or custom @subprocess_retry()
- [ ] Add exponential backoff
- [ ] Add tests

### Step 6: Database Operations
- [ ] Create @db_operation_retry() decorator
- [ ] Apply to db.py operations
- [ ] Handle SQLAlchemyError specifically
- [ ] Add tests

---

## Metrics to Track After Consolidation

1. **Lines of Code Reduction:** Target 240-410 line reduction
2. **Retry Effectiveness:** Track actual retry success rates
3. **Performance Impact:** Monitor timeout variance effectiveness
4. **Circuit Breaker Effectiveness:** Track state transitions and half-open recovery
5. **Error Classification Accuracy:** Monitor how many errors are correctly classified

