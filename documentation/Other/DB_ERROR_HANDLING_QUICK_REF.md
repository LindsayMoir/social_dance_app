# DatabaseHandler Error Handling - Quick Reference

## Try-Except Blocks at a Glance

| Line(s) | Method | Pattern | Type | Priority | Decorator Rec. |
|---------|--------|---------|------|----------|----------------|
| 209-220 | `get_db_connection()` | Try-Log-Return None | Exception | HIGH | @resilient_execution |
| 371-381 | `create_urls_df()` | Try-Log-Return Empty | SQLAlchemyError | HIGH | @resilient_execution |
| 411-474 | `execute_query()` | Try-Connect-Log-Return | SQLAlchemyError | CRITICAL | @resilient_execution |
| 487-493 | `close_connection()` | Nested Try-Except | Exception | LOW | None (cleanup) |
| 769-783 | `cache_raw_location()` | Try-Execute-Log | Exception | MEDIUM | @resilient_execution |
| 790-802 | `lookup_raw_location()` | Try-Query-Return | Exception | MEDIUM | @resilient_execution |
| 838-852 | `create_raw_locations_table()` | Try-Multi-Execute | Exception | LOW | None (init) |
| 995-998 | `fuzzy_duplicates()` | Nested Silent Try | Exception | MEDIUM | Extract + log |
| 1085-1103 | `multiple_db_inserts()` | Try-Batch-Upsert | Exception | HIGH | @resilient_execution |
| 1346-1352 | `sql_input()` | Try-JSON-Load | Exception | LOW | None (config) |
| 1505-1685 | `reset_address_id_sequence()` | Try-Complex-Op | Exception | HIGH | Refactor + @timeout |
| 1697-1753 | `update_full_address_with_building_names()` | Try-Batch-Update | Exception | MEDIUM | @resilient_execution |

## Critical Methods (Use These to Focus Efforts)

### 1. execute_query() - MOST USED
- **Location**: Lines 384-474
- **Calls**: 50+ times
- **Current Try-Except**: Lines 411-474
- **Issue**: No retry for transient errors
- **Action**: Add `@resilient_execution(max_retries=3, catch_exceptions=(SQLAlchemyError, TimeoutError))`
- **Savings**: ~20 lines

### 2. reset_address_id_sequence() - MOST COMPLEX
- **Location**: Lines 1491-1685 (194 lines)
- **Calls**: Rarely (maintenance)
- **Current Try-Except**: Lines 1505-1685, nested cleanup
- **Issue**: No retry, bare except, no rollback
- **Action**: Split into 3-4 methods with `@resilient_execution`, add `@with_timeout`
- **Savings**: ~30-40 lines

### 3. multiple_db_inserts() - BATCH OPERATION
- **Location**: Lines 1062-1103
- **Calls**: During bulk writes
- **Current Try-Except**: Lines 1085-1103
- **Issue**: No partial success handling
- **Action**: Add `@resilient_execution(max_retries=2)`
- **Savings**: ~18 lines

### 4. fuzzy_duplicates() - SILENT FAILURES
- **Location**: Lines 943-1011
- **Calls**: During cleanup
- **Current Try-Except**: Lines 995-998 (SILENT!)
- **Issue**: No logging in exception (line 997)
- **Action**: Add logging, extract updates/deletes to separate methods
- **Savings**: ~5-10 lines + adds 2 lines for logging

## Silent Exception Issues (MUST FIX)

1. **fuzzy_duplicates() Line 997**
   ```python
   try:
       value = value.item() if hasattr(value, 'item') else value.tolist()
   except Exception:
       pass  # <- SILENT! No logging
   ```
   **Fix**: Add logging
   ```python
   except Exception as e:
       logging.warning(f"fuzzy_duplicates: Failed to convert numpy value: {e}")
   ```

2. **reset_address_id_sequence() Line 1683**
   ```python
   except:  # <- BARE EXCEPT! Too broad
       pass
   ```
   **Fix**: Use specific exception
   ```python
   except Exception as e:
       logging.error(f"reset_address_id_sequence: Cleanup failed: {e}")
   ```

## Timeout Issues (MISSING)

| Method | Max Runtime | Current Timeout | Recommended |
|--------|-------------|-----------------|-------------|
| `execute_query()` | Varies | None | 30 seconds |
| `create_urls_df()` | Varies | None | 60 seconds |
| `reset_address_id_sequence()` | Minutes | None | 5 minutes |
| `fuzzy_duplicates()` | Minutes | None | 10 minutes |
| LLM operations | 10-30s | None | 30 seconds |

## Pattern Usage Distribution

```
Try-Log-Return None:        7 methods (50%)
Try-Execute-Log-Continue:   4 methods (29%)
Try-Execute-Re-raise:       3 methods (21%)
```

## Quick Implementation Checklist

- [ ] Add `@resilient_execution` to `execute_query()`
- [ ] Add `@resilient_execution` to `create_urls_df()`
- [ ] Add `@resilient_execution` to `multiple_db_inserts()`
- [ ] Fix silent exception in `fuzzy_duplicates()` line 997
- [ ] Fix bare except in `reset_address_id_sequence()` line 1683
- [ ] Add `@with_timeout` to long-running operations
- [ ] Extract numpy conversion to separate method with logging
- [ ] Refactor `reset_address_id_sequence()` into smaller methods
- [ ] Add circuit breaker for LLM operations
- [ ] Write integration tests for retry logic

## Existing Infrastructure

Already available in `/mnt/d/GitHub/social_dance_app/src/resilience.py`:
- `@resilient_execution` - Generic resilience decorator
- `@http_retry` - HTTP-specific retry
- `@with_timeout` - Timeout decorator
- `RetryManager` - Centralized retry logic
- `CircuitBreaker` - Failure prevention pattern
- `RetryStrategy` - FIXED, LINEAR, EXPONENTIAL, EXPONENTIAL_WITH_JITTER

**Action**: Import and use these instead of writing new error handling code!

## External Service Dependencies

### LLM (AddressResolutionRepository)
- **Methods**: `_resolve_via_llm()` (lines 168-228)
- **Timeout**: Not explicit
- **Error Handling**: Try-except at line 226 (warning log)
- **Recommendation**: Add `@http_retry` or `@with_timeout` decorator

### Database (PostgreSQL)
- **Core Method**: `execute_query()` (lines 384-474)
- **Connection Pool**: Not tracked
- **Timeout**: Not explicit
- **Recommendation**: Add `@resilient_execution` with timeout

### Cache (PostgreSQL raw_locations table)
- **Methods**: `cache_raw_location()`, `lookup_raw_location()`
- **Criticality**: Non-critical (graceful degrade)
- **Error Handling**: Appropriate (fire-and-forget)
- **Recommendation**: Optional retry, appropriate as-is

## Consolidation by Impact

| Method | Lines Saved | Impact | Effort |
|--------|------------|--------|--------|
| execute_query() | 20 | CRITICAL | Low |
| reset_address_id_sequence() | 30-40 | HIGH | High |
| multiple_db_inserts() | 18 | HIGH | Low |
| create_urls_df() | 10 | MEDIUM | Low |
| fuzzy_duplicates() | 5-10 | MEDIUM | Low |
| cache_raw_location() | 15 | LOW | Low |
| lookup_raw_location() | 13 | LOW | Low |
| **TOTAL** | **~150** | - | - |

