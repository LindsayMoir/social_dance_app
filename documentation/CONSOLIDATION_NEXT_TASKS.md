# Consolidation Initiative - Next High-Impact Tasks

## Executive Summary

Based on analysis of the completed consolidation work (HandlerFactory, error handling decorators, logging consolidation, and repository patterns), I've identified 8 prioritized tasks that can deliver significant value with manageable effort.

**Total Potential Lines Saved: ~800-1200 lines**
**Average Task Effort: 2-8 hours**
**Total Estimated Effort: 35-50 hours**

---

## PRIORITY 1: IMMEDIATE QUICK WINS (Do First)

### Task 1: Migrate ImageScraper to use HandlerFactory
**Impact Level:** HIGH | **Effort:** 4 hours | **Lines Saved:** 60-80

**Current State:**
- ImageScraper (783 lines) manually initializes handlers directly
- Contains duplicate browser/auth/DB initialization code
- Uses old patterns not integrated with PlaywrightManager

**Expected Outcome:**
- Reduces ImageScraper to ~720 lines (~7% reduction)
- Eliminates handler initialization boilerplate (~60 lines)
- Improves consistency with other scrapers (fb_v2, ebs_v2, rd_ext_v2)
- Better error handling through factory defaults

**Implementation:**
```python
# Current pattern:
self.llm_handler = LLMHandler(config)
self.db_handler = self.llm_handler.db_handler

# New pattern:
handlers = HandlerFactory.create_web_scraper_handlers(config, logger)
self.llm_handler = LLMHandler(config)
self.db_writer = handlers['db_writer']
# ... with auth_manager, retry_manager, etc pre-configured
```

**Files Affected:**
- `/src/images.py` (primary)
- `/src/handler_factory.py` (reference)

**Dependencies:** None - ready to implement immediately

---

### Task 2: Add Error Handling Decorators to Repository Classes
**Impact Level:** HIGH | **Effort:** 3-4 hours | **Lines Saved:** 40-60

**Current State:**
- 9 repository files with manual try/except blocks
- EventRepository (150+ lines) has repeated error handling patterns
- No centralized error recovery strategy
- 13+ try-except blocks across repositories

**Expected Outcome:**
- Decorators applied to all repository methods
- Consistent error logging and recovery
- ~40-60 lines of boilerplate removed
- Better observability with structured logging

**Implementation:**
```python
# Current pattern:
def write_events_to_db(self, df):
    try:
        # ... 20 lines of logic
    except Exception as e:
        self.logger.error(f"Failed: {e}")
        return False

# New pattern:
@resilient_execution(max_retries=3, catch_exceptions=(SQLAlchemyError, IOError))
def write_events_to_db(self, df):
    # ... 20 lines of logic (no error handling needed)
```

**Files Affected:**
- `/src/repositories/event_repository.py` (main candidate)
- `/src/repositories/address_repository.py`
- `/src/repositories/event_management_repository.py`
- `/src/repositories/address_resolution_repository.py`
- `/src/resilience.py` (reference)

**Dependencies:** None - decorators already exist

---

### Task 3: Consolidate Database Error Handling in DatabaseHandler
**Impact Level:** MEDIUM-HIGH | **Effort:** 5-6 hours | **Lines Saved:** 80-120

**Current State:**
- DatabaseHandler (db.py) has 13+ try-except blocks
- Connection errors handled inconsistently
- No unified error classification
- Manual retry logic in several methods

**Expected Outcome:**
- Apply @resilient_execution decorator to 8-10 DB methods
- Unified connection error handling
- ~80-120 lines of boilerplate removed
- Better logging through decorator infrastructure

**Implementation Targets:**
- `write_events_to_db()` method
- `write_url_to_db()` method  
- `query_database()` method
- `get_events_by_similarity()` method
- Database operation methods that interact with connections

**Files Affected:**
- `/src/db.py` (primary)
- `/src/resilience.py` (reference)

**Dependencies:** Task 2 (but can be done in parallel)

---

## PRIORITY 2: HIGH-VALUE STANDARDIZATION (Do Next)

### Task 4: Create Unified HTTPHandler for API/Web Calls
**Impact Level:** HIGH | **Effort:** 6-8 hours | **Lines Saved:** 100-150

**Current State:**
- Http request logic scattered across 6+ files
- Manual retry/timeout logic in multiple places
- RequestHandler uses different patterns than HTTP decorator
- ~30 manual retry loops across codebase

**Expected Outcome:**
- New HTTPHandler class encapsulating all HTTP patterns
- Uses @http_retry decorator internally
- Replaces 100-150 lines of manual request code
- Consistent timeout/retry/header handling

**Implementation:**
```python
# New HTTPHandler approach:
class HTTPHandler:
    @http_retry(max_retries=3, base_delay=1.0)
    def fetch_json(self, url, headers=None):
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    
    @http_retry(max_retries=3, retriable_statuses=[408, 429, 503])
    def fetch_html(self, url):
        # ...
```

**Current Problematic Patterns to Replace:**
- Manual timeout handling in `fb_v2.py` (15-20 lines)
- Retry logic in `ebs_v2.py` (10-15 lines)
- RequestHandler patterns (30-40 lines)
- Manual header construction (20-30 lines)

**Files Affected:**
- NEW: `/src/http_handler.py` (create)
- `/src/fb_v2.py` (refactor)
- `/src/ebs_v2.py` (refactor)
- `/src/rd_ext_v2.py` (refactor)
- `/src/images.py` (refactor)
- `/src/resilience.py` (reference)

**Dependencies:** None - can be done immediately after Priority 1 tasks

---

### Task 5: Extract Authentication Patterns into AuthenticationHandler
**Impact Level:** MEDIUM-HIGH | **Effort:** 5-6 hours | **Lines Saved:** 70-100

**Current State:**
- AuthenticationManager exists but isn't fully utilized
- Manual login logic duplicated in scrapers
- Cookie/session management scattered
- ~70-100 lines of login code across 3-4 files

**Expected Outcome:**
- Enhanced AuthenticationHandler with common patterns
- Login retry logic with exponential backoff
- Cookie validation/refresh patterns
- ~70-100 lines of boilerplate eliminated

**Implementation Targets:**
- Extract Instagram cookie loading (images.py, ~50 lines)
- Extract session verification (images.py, ~30 lines)
- Extract generic login patterns (rd_ext_v2.py, ~40 lines)
- Create AuthenticationHandler wrapper

**Files Affected:**
- `/src/auth_manager.py` (enhance)
- `/src/images.py` (refactor)
- `/src/rd_ext_v2.py` (refactor)
- `/src/fb_v2.py` (refactor)

**Dependencies:** Task 1 (HandlerFactory update)

---

## PRIORITY 3: STRATEGIC CONSOLIDATIONS (Do After Core Tasks)

### Task 6: Create ConfigurationValidator and Consolidate Config Access
**Impact Level:** MEDIUM | **Effort:** 4-5 hours | **Lines Saved:** 50-80

**Current State:**
- 8+ files have manual config loading (`.get()` chains)
- No unified validation of required config keys
- Inconsistent config access patterns
- ~50-80 lines of config boilerplate

**Expected Outcome:**
- ConfigurationValidator class with schema validation
- Decorator for validating config before method execution
- Consistent config access through factory
- Prevents runtime config-related failures

**Implementation:**
```python
# Create validator:
class ConfigurationValidator:
    REQUIRED_KEYS = {
        'crawling': ['headless', 'max_retries'],
        'database': ['connection_string'],
        'llm': ['api_key', 'provider']
    }
    
    @staticmethod
    def validate(config: dict) -> bool:
        # ...

# Use in constructors:
@require_valid_config(['crawling', 'database'])
def __init__(self, config_path):
    # ...
```

**Files Affected:**
- NEW: `/src/configuration_validator.py` (create)
- `/src/base_scraper.py` (integrate)
- `/src/handler_factory.py` (integrate)
- Multiple scraper files (refactor config access)

**Dependencies:** None - can be done in parallel

---

### Task 7: Implement Error Recovery Strategies for LLM Operations
**Impact Level:** MEDIUM | **Effort:** 6-7 hours | **Lines Saved:** 60-100

**Current State:**
- LLMHandler has minimal retry logic
- No circuit breaker for LLM API calls
- Fallback strategies not standardized
- 10+ try-except blocks specific to LLM errors

**Expected Outcome:**
- LLM-specific retry strategy with jitter
- Circuit breaker for LLM API endpoints
- Fallback prompt/cache mechanism
- ~60-100 lines of error handling consolidated

**Implementation:**
```python
# Create LLM-specific decorator:
@llm_retry(max_retries=5, base_delay=2.0, max_delay=120.0)
def generate_prompt_response(self, prompt):
    # ...

# Create circuit breaker:
self.llm_circuit_breaker = CircuitBreaker(
    failure_threshold=0.5,
    timeout=300
)
```

**Files Affected:**
- `/src/llm.py` (primary - integrate decorators)
- `/src/resilience.py` (add LLM-specific strategies)
- `/src/dedup_llm.py` (refactor error handling)
- `/src/repositories/address_resolution_repository.py` (refactor)

**Dependencies:** Task 2-3 (decorators already implemented)

---

## PRIORITY 4: ADVANCED CONSOLIDATIONS (Nice-to-Have)

### Task 8: Create AsyncOperationHandler for Coordinating Async Scrapers
**Impact Level:** MEDIUM | **Effort:** 8-10 hours | **Lines Saved:** 120-180

**Current State:**
- Three async scrapers with different coordination patterns
- Manual semaphore/lock management in gen_scraper.py
- No unified async resource pooling
- ~120-180 lines of async boilerplate

**Expected Outcome:**
- AsyncOperationHandler managing multiple async operations
- Unified semaphore/resource management
- Consistent timeout handling across async operations
- Better performance through coordinated parallelism

**Implementation:**
```python
# Create handler:
class AsyncOperationHandler:
    def __init__(self, max_concurrent=5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.circuit_breaker = CircuitBreaker()
    
    async def execute_with_limits(self, coro):
        async with self.semaphore:
            if not self.circuit_breaker.can_execute():
                raise CircuitBreakerOpenError()
            try:
                result = await asyncio.wait_for(coro, timeout=30)
                self.circuit_breaker.record_success()
                return result
            except Exception as e:
                self.circuit_breaker.record_failure()
                raise
```

**Files Affected:**
- NEW: `/src/async_handler.py` (create)
- `/src/gen_scraper.py` (refactor coordination)
- `/src/ebs_v2.py` (simplify async management)
- `/src/rd_ext_v2.py` (simplify async management)
- `/src/images.py` (simplify async management)

**Dependencies:** Tasks 1-3 (foundation work)

---

## TECHNICAL DEBT & BUG FIXES (Bonus Tasks)

### Quick Wins Identified:

1. **Fix inconsistent error logging across repositories** (30 mins)
   - Some repos use `logger.error()`, others use `logging.error()`
   - Create standardized error logging wrapper
   - Files: All repositories

2. **Add missing timeout handling to database queries** (1-2 hours)
   - Some DB operations have no timeout protection
   - Add default 60s timeout to all queries
   - Files: `/src/db.py`

3. **Implement context manager for resource cleanup** (1 hour)
   - Several scrapers have incomplete cleanup
   - Create ResourceManager context manager
   - Files: `/src/base_scraper.py`, scraper files

4. **Add structured logging to all error paths** (2-3 hours)
   - Use ProductionLogger for consistency
   - Standardize error context fields
   - Files: Multiple

---

## RECOMMENDED IMPLEMENTATION SEQUENCE

**Week 1 (Priority 1):**
- Task 1: Migrate ImageScraper (4 hours)
- Task 2: Add decorators to repositories (4 hours)
- Task 3: DatabaseHandler error handling (6 hours)

**Week 2 (Priority 2):**
- Task 4: HTTPHandler creation (8 hours)
- Task 5: AuthenticationHandler enhancement (6 hours)
- Task 6: ConfigurationValidator (5 hours)

**Week 3-4 (Priority 3-4):**
- Task 7: LLM error recovery (7 hours)
- Task 8: AsyncOperationHandler (10 hours)
- Quick wins & bug fixes (6 hours)

---

## MEASURING SUCCESS

**Metrics to Track:**

1. **Code Reduction:**
   - Target: 800-1200 lines eliminated
   - Tracking: Line counts before/after for each task

2. **Error Handling Coverage:**
   - Target: 90%+ of try-except blocks replaced with decorators
   - Current: ~35 try-except blocks across priority files
   - Success: <5 manual try-except blocks remaining

3. **Consistency:**
   - Target: 100% of scrapers using HandlerFactory
   - Current: 4/6 using it (fb_v2, ebs_v2, rd_ext_v2, gen_scraper)
   - Remaining: images.py, read_pdfs_v2.py (ready for Task 1)

4. **Test Coverage Improvement:**
   - New tests for HTTPHandler, ConfigValidator
   - Decorator testing improvements
   - Mock improvements for async operations

---

## RISKS & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Refactoring breaks existing functionality | Medium | High | Comprehensive test suite before refactoring |
| LLM retry decorator causes rate limiting | Low | Medium | Exponential backoff with jitter already in place |
| Async handler coordination issues | Medium | Medium | Thorough testing with multiple concurrent operations |
| Breaking existing code using manual patterns | Medium | Medium | Backward compatibility wrappers for 2-3 releases |

---

## CONCLUSION

The consolidation initiative has created a strong foundation. These 8 prioritized tasks build directly on that work and can deliver:
- **800-1200 lines of eliminated technical debt**
- **90%+ standardization of error handling**
- **Improved code maintainability and readability**
- **Better error recovery and observability**

Priority 1-2 tasks (5 tasks) should be completed first (~30 hours) to maximize immediate value. Priority 3-4 tasks can be scheduled based on business priorities.
