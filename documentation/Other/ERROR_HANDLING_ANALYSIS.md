# Error Handling and Resilience Analysis Report

## Executive Summary

The codebase has a good foundation with `resilience.py` providing centralized retry and circuit breaker patterns, but there are significant opportunities for consolidation. Currently, error handling is fragmented across multiple files with repeated patterns that could be abstracted into decorator-based utilities.

---

## 1. Current Resilience Infrastructure

### A. resilience.py (Core Resilience Module)
**Location:** `/mnt/d/GitHub/social_dance_app/src/resilience.py`

**Existing Implementations:**

1. **RetryManager Class**
   - `retry_sync()`: Synchronous retry with configurable strategies
   - `retry_async()`: Asynchronous retry support
   - Decorators: `retry_with_backoff()`, `retry_with_exponential_backoff()`
   - Strategies: FIXED, LINEAR, EXPONENTIAL, EXPONENTIAL_WITH_JITTER
   - Retriable errors: ConnectionError, TimeoutError, OSError, IOError
   - Features:
     - Configurable max_retries (default: 3)
     - Base delay: 1.0s, max delay: 60s
     - Jitter support (±10%) to avoid thundering herd
     - Detailed logging at each retry attempt

2. **CircuitBreaker Class**
   - States: "closed", "open", "half-open"
   - Failure threshold: 0.5 (50%)
   - Timeout: 60 seconds before attempting recovery
   - Methods: `can_execute()`, `record_success()`, `record_failure()`, `reset()`
   - Tracks failure_rate and transitions states

3. **HTTP Status Classification**
   - `classify_http_status()`: Treats 408, 429, 500, 502, 503, 504 as retriable

---

## 2. Current Error Handling Patterns (Fragmented)

### A. Direct Retry Loops (Not Using RetryManager)
**Files:**
- `browser_utils.py`: Lines 188-244 (navigate_safe, navigate_safe_async)
- `images.py`: Lines 282-320 (exponential backoff with duplicate logic)
- `pipeline.py`: Lines 250-244 (subprocess retry logic)
- `auth_manager.py`: Multiple timeout handling but no centralized retry

**Pattern Found:**
```python
for attempt in range(max_retries):
    try:
        # operation
    except (SpecificError, AnotherError) as e:
        if attempt < max_retries - 1:
            time.sleep(calculated_delay)
        else:
            raise/log
```

**Issues:**
1. Duplicated delay calculation logic
2. No consistent exponential backoff formula
3. Inconsistent error classification
4. Manual attempt counting and logging

### B. Timeout Handling (Scattered)
**Files:**
- `fb_v2.py`: Lines 177-351 (random timeout variation)
- `images.py`: Lines 386-416 (requests timeout handling)
- `browser_utils.py`: Lines 92-106 (timeout with randomization)
- `auth_manager.py`: Multiple timeout parameters without unified handling

**Current Patterns:**
```python
# Random timeout variation (different in each file)
timeout = random.randint(20000 // 2, int(20000 * 1.5))  # 50% variation
timeout = self.DEFAULT_NAVIGATION_TIMEOUT * 0.5 + variation  # PlaywrightManager
```

### C. HTTP Error Handling (Inconsistent)
**Files:**
- `images.py`: Lines 396-409 (specific handling for 403, 429)
- `read_pdfs_v2.py`: Lines 274-287 (basic raise_for_status)
- `fb_v2.py`: No explicit HTTP error handling

**Missing:** 
- Unified HTTP error classification
- Consistent status code → retriable decision mapping
- Rate limit (429) specific backoff

### D. Generic Exception Handling (Over-broad)
**Files:** Widespread `except Exception as e:` without specific handling
- `fb_v2.py`: Lines 183, 199, 210, 223, 258, 303-350
- `images.py`: Lines 211, 314, 410
- `read_pdfs_v2.py`: Lines 207-210

**Issues:**
1. Catches all exceptions, logs, and continues
2. No differentiation between retriable and fatal errors
3. Lost opportunity for circuit breaker integration
4. Inconsistent logging levels (warning vs error)

---

## 3. Files Currently Using resilience.py

**Importing resilience module:** 4 files
1. `/mnt/d/GitHub/social_dance_app/src/fb_v2.py` - Lines 51
2. `/mnt/d/GitHub/social_dance_app/src/base_scraper.py` - Line 31
3. `/mnt/d/GitHub/social_dance_app/src/read_pdfs_v2.py` - Line 36
4. `/mnt/d/GitHub/social_dance_app/tests/test_fb_v2_scraper.py` - Test file

**Usage Analysis:**
- `fb_v2.py`: Imports but doesn't use (only instantiates CircuitBreaker at line 109)
- `base_scraper.py`: Imports both RetryManager and CircuitBreaker
  - Creates instance at line 68
  - Uses circuit breaker state at line 332
  - But doesn't use retry_with_backoff decorator
- `read_pdfs_v2.py`: Imports but stores instance (line 107-108)
  - Never actually calls retry methods
  - Only uses CircuitBreaker.record_failure()

**Gap:** Files are importing resilience utilities but not actively using retry decorators

---

## 4. Common Error Patterns Being Handled

### A. Connection Errors (9 occurrences)
- ConnectionError, TimeoutError, OSError, IOError
- Handled in: resilience.py, browser_utils.py, auth_manager.py

### B. Timeout Errors (15+ occurrences)
- PlaywrightTimeoutError, requests.ConnectTimeout, socket.timeout
- Handled in: fb_v2.py, images.py, browser_utils.py, auth_manager.py
- **Issue:** Each file implements its own retry logic

### C. HTTP Errors (8+ occurrences)
- 403 Forbidden (Instagram CDN token expiration)
- 404 Not Found
- 429 Rate Limiting
- 500-504 Server Errors
- Handled in: images.py, read_pdfs_v2.py

### D. Database Errors (3+ occurrences)
- SQLAlchemyError
- Handled in: db.py only
- **Gap:** No database-specific error handling decorators

### E. JSON/Parse Errors (Multiple)
- json.JSONDecodeError, regex failures, missing fields
- Handled in: llm.py, gen_scraper.py
- **Gap:** No parser error retry pattern

### F. File I/O Errors (5+ occurrences)
- FileNotFoundError, IOError, PermissionError
- Handled in: images.py, pipeline.py
- **Gap:** No I/O specific retry decorator

---

## 5. Opportunities for Decorator-Based Consolidation

### A. HIGH PRIORITY: HTTP Request Retry Decorator

**Current State:** 8+ separate try-except blocks with similar logic
**Files Affected:** images.py, read_pdfs_v2.py, browser_utils.py, auth_manager.py

**Proposed Decorator:**
```python
@http_retry(
    max_retries=3,
    backoff_strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
    retriable_statuses={408, 429, 500, 502, 503, 504},
    timeout=30
)
def fetch_url(url):
    return requests.get(url, timeout=30)
```

**Benefits:**
- Consolidate 8+ retry loops into single decorator
- Centralized HTTP status classification
- Automatic rate limit handling (429 → exponential backoff)
- Consistent timeout management

### B. HIGH PRIORITY: Timeout Handling Decorator

**Current State:** Timeout logic scattered across 5+ files with inconsistent delay calculations
**Files Affected:** fb_v2.py, images.py, browser_utils.py, auth_manager.py, browser_utils.py

**Proposed Decorator:**
```python
@with_timeout(
    timeout_ms=20000,
    variation_percent=50,  # Random ±50% variation for anti-detection
    retry_on_timeout=True,
    max_retries=2
)
async def navigate_to_url(page, url):
    await page.goto(url, wait_until='networkidle')
```

**Benefits:**
- Single source for timeout randomization logic
- Unified variation calculation (currently: 3 different approaches)
- Consistent retry behavior on timeout
- Works for both sync and async code

### C. MEDIUM PRIORITY: Generic Resilient Execution Decorator

**Current State:** 15+ `except Exception as e:` blocks with generic error handling
**Files Affected:** fb_v2.py (6 instances), images.py (4 instances), read_pdfs_v2.py

**Proposed Decorator:**
```python
@resilient_execution(
    circuit_breaker=cb_instance,
    log_level='warning',
    default_return=None,
    record_failure_on_exception=True
)
def process_event(event_data):
    # operation that may fail
    return result
```

**Benefits:**
- Automatic circuit breaker integration
- Consistent failure recording
- Unified logging at catch point
- Configurable fallback return values

### D. MEDIUM PRIORITY: Database Operation Retry Decorator

**Current State:** Database operations have minimal error handling
**Files Affected:** db.py, repositories/*.py

**Proposed Decorator:**
```python
@db_operation_retry(
    max_retries=3,
    retriable_errors={OperationalError, TimeoutError},
    backoff_strategy=RetryStrategy.LINEAR
)
def write_to_database(conn, data):
    # database operation
    pass
```

**Benefits:**
- Handle transient DB connection issues
- Automatic retry on deadlocks
- Consistent logging for database operations
- Works with SQLAlchemy exceptions

### E. LOW PRIORITY: Parser Error Retry Decorator

**Current State:** LLM/JSON parsing errors caught generically
**Files Affected:** llm.py, gen_scraper.py

**Proposed Decorator:**
```python
@parse_with_retry(
    max_retries=2,
    fallback_value=None,
    log_errors=True
)
def parse_llm_response(response_text):
    return json.loads(response_text)
```

---

## 6. Error Handling Smell Detection

### A. Inconsistent Error Severity Logging
**Files:** fb_v2.py, images.py, pipeline.py
**Issue:** Same error conditions logged at different levels (warning vs error)

Example:
- `fb_v2.py` line 183: `logger.warning(f"...timed out")`
- `browser_utils.py` line 206: `logger.warning(f"...timeout")`
- `images.py` line 300: `logger.warning(f"...403 Forbidden")`
- But `images.py` line 408: `logger.error(f"...HTTP error")`

### B. Silent Failures
**Pattern:** `return None` without logging error details
- `read_pdfs_v2.py`: Lines 259, 287 (exception logged but no context)
- `images.py`: Lines 312, 320, 416
- **Risk:** Difficult to debug why operations failed

### C. Unhandled Async Timeout
**Location:** `images.py` lines 282-320
**Issue:** Async code uses `await asyncio.sleep()` but no handling for `asyncio.TimeoutError`

### D. Retry Logic Without Backoff Metrics
**Pattern:** Retries happen but no metric tracking
- No tracking of actual retry counts
- No performance impact analysis
- No retry effectiveness measurement

---

## 7. Circuit Breaker Integration Gaps

### Current Usage:
- `base_scraper.py`: Creates CircuitBreaker but only reads state (line 332)
- `fb_v2.py`: Creates instance but never uses (line 109)
- `read_pdfs_v2.py`: Calls `record_failure()` but never checks `can_execute()`

**Missing:**
1. Pre-flight checks using `can_execute()` before operations
2. Automatic state transitions on consecutive failures
3. Half-open state testing strategy
4. Circuit breaker metrics exposure

**Recommended Pattern:**
```python
if not scraper.circuit_breaker.can_execute():
    logger.critical("Circuit breaker open, aborting")
    return

try:
    result = operation()
    scraper.circuit_breaker.record_success()
except Exception as e:
    scraper.circuit_breaker.record_failure()
    raise
```

---

## 8. Summary of Consolidation Opportunities

| Opportunity | Files Affected | Pattern Count | Priority | Est. Lines Saved |
|------------|----------------|---------------|----------|-----------------|
| HTTP Request Retry Decorator | 5 files | 8+ | HIGH | 50-80 |
| Timeout Handling Decorator | 5 files | 15+ | HIGH | 60-100 |
| Generic Resilient Execution | 10+ files | 20+ | MEDIUM | 80-150 |
| DB Operation Retry Decorator | 3 files | 5+ | MEDIUM | 30-50 |
| Parser Error Retry Decorator | 2 files | 3+ | LOW | 20-30 |
| **TOTAL CONSOLIDATION OPPORTUNITY** | | | | **240-410 lines** |

---

## 9. Recommended Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. Extend `resilience.py` with new decorators
   - `@http_retry()` - handle HTTP requests
   - `@with_timeout()` - handle timeout + retry
   - Update `RetryManager` to support async context managers

2. Add HTTP status classification enhancement
   ```python
   def classify_http_error(status_code):
       """Return (is_retriable, error_category, recommended_backoff)"""
   ```

### Phase 2: Migration (Week 2-3)
1. Replace direct retry loops in `browser_utils.py` with decorator
2. Consolidate timeout logic in `images.py` and `fb_v2.py`
3. Refactor `auth_manager.py` to use timeout decorator
4. Update `pipeline.py` to use HTTP retry decorator

### Phase 3: Advanced (Week 4)
1. Add `@resilient_execution()` decorator for generic error handling
2. Implement database operation retry decorator
3. Add circuit breaker state metrics
4. Create error handling testing utilities

### Phase 4: Optimization (Ongoing)
1. Add retry metrics collection
2. Create resilience dashboard/logging
3. Implement adaptive retry strategies
4. Add distributed rate limiting support

---

## 10. Code Quality Improvements

### Currently Missing:
1. Retry metric collection (counts, durations)
2. Error classification consistency
3. Distributed rate limit awareness
4. Adaptive backoff based on error history
5. Integration with monitoring/alerting

### Recommended Additions to resilience.py:
```python
class RetryMetrics:
    def __init__(self):
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_operations = 0
        self.retry_durations = []
        self.error_histogram = {}
    
    def record_retry(self, attempt_num, duration, success):
        # Track retry effectiveness
        pass
```

---

## Files by Error Handling Maturity

### Mature (Using resilience.py properly):
- None currently - all files underutilize the resilience infrastructure

### Intermediate (Some error handling, could use decorators):
- `browser_utils.py` - Good navigation retry logic, could use decorator
- `read_pdfs_v2.py` - Uses RetryManager, but only stores instance
- `auth_manager.py` - Timeout handling present, not systematic

### Immature (Ad-hoc error handling):
- `images.py` - 8+ error handling blocks, significant code duplication
- `fb_v2.py` - 6+ try-except blocks, imports resilience but doesn't use
- `pipeline.py` - Subprocess retry without exponential backoff
- `llm.py` - Generic exception catching, no retry strategy
- `db.py` - SQLAlchemy errors not consistently handled

---

## Conclusion

The codebase has a solid foundation with `resilience.py` providing RetryManager and CircuitBreaker, but **significant opportunities exist for consolidation**:

1. **240-410 lines of duplicated error handling code** could be eliminated
2. **5+ different retry implementations** could be unified into 2-3 decorators
3. **15+ timeout handling approaches** could become 1 standardized pattern
4. **20+ generic exception catches** could be unified with circuit breaker pattern

**Quick Win:** Implement `@http_retry()` and `@with_timeout()` decorators - would eliminate ~150 lines and improve code maintainability significantly.

**Long-term Win:** Migrate all error handling to decorator-based approach with metrics collection for production-grade resilience monitoring.
