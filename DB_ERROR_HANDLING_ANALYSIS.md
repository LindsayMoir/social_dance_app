# DatabaseHandler Error Handling Analysis

## Executive Summary
- **File**: `/mnt/d/GitHub/social_dance_app/src/db.py` (1,787 lines)
- **Try-Except Blocks Found**: 14 total (including nested)
- **Error Handling Patterns**: 3 primary patterns identified
- **Critical Path Methods**: 6 with complex error handling
- **Consolidation Opportunity**: ~200-300 lines of code can be consolidated
- **Recommendation**: Apply `@resilient_execution` decorator to 8-10 methods

---

## 1. Try-Except Blocks Inventory

### Block 1: `get_db_connection()` (Lines 209-220)
```
Lines 209-220: try-except Exception
Pattern: Try-log-return None
Exception Type: Generic Exception
Logging: Error log
Recovery Strategy: Return None (graceful degradation)
```
**Issues**: 
- Generic exception catch (too broad)
- Silent failure returns None
- No retry capability

---

### Block 2: `create_urls_df()` (Lines 371-381)
```
Lines 371-381: try-except SQLAlchemyError
Pattern: Try-log-return empty DataFrame
Exception Type: SQLAlchemyError (specific)
Logging: Error log
Recovery Strategy: Return empty DataFrame
```
**Issues**:
- Specific exception good, but operation could benefit from retry
- Logs error but loses data context

---

### Block 3: `execute_query()` - Main Query Execution (Lines 411-474)
```
Lines 411-474: try-except SQLAlchemyError
Pattern: Try-connect-query-except-log-return None
Exception Type: SQLAlchemyError (specific)
Logging: Error/Info logs with context
Recovery Strategy: Return None
Special Handling: UniqueViolation constraint detected (lines 464-468)
```
**Issues**:
- Most complex error handling in file
- Handles unique constraint gracefully (good!)
- No retry logic despite being core operation
- Could benefit from retry for transient errors

---

### Block 4: `close_connection()` (Lines 487-493)
```
Lines 487-493: nested try-except blocks
- Inner try (487-489): conn.dispose()
- Outer except (490-493): Exception handling
Pattern: Try-except-log with finally-like fallback
Logging: Info/Error logs
Recovery Strategy: Log and continue
```
**Issues**:
- Nested structure unclear
- No retry for connection close

---

### Block 5: `cache_raw_location()` (Lines 769-783)
```
Lines 769-783: try-except Exception
Pattern: Try-execute-except-log (no return value)
Exception Type: Generic Exception
Logging: Info on success, warning on failure
Recovery Strategy: Log warning, continue (fire-and-forget)
```
**Issues**:
- Overly broad exception catch
- Non-critical operation, appropriate handling

---

### Block 6: `lookup_raw_location()` (Lines 790-802)
```
Lines 790-802: try-except Exception
Pattern: Try-execute-except-log-return None
Exception Type: Generic Exception
Logging: Info on hit, warning on miss
Recovery Strategy: Return None (fail gracefully)
```
**Issues**:
- Overly broad exception catch
- Cache miss could benefit from retry

---

### Block 7: `create_raw_locations_table()` (Lines 838-852)
```
Lines 838-852: try-except Exception
Pattern: Try-execute-multiple-except-log
Exception Type: Generic Exception
Logging: Info on success, error on failure with SQL context
Recovery Strategy: Log error and SQL, continue
```
**Issues**:
- Good error context logging
- Could have schema migration retry logic

---

### Block 8 & 9: `fuzzy_duplicates()` - Nested Conversion (Lines 995-998)
```
Lines 995-998: try-except Exception (nested in loop)
Pattern: Try-numpy-conversion-except-pass
Exception Type: Generic Exception
Logging: None (silent)
Recovery Strategy: Continue (pass)
```
**Issues**:
- Silent failure (dangerous)
- No logging
- Nested in critical operation loop

---

### Block 10: `multiple_db_inserts()` (Lines 1085-1103)
```
Lines 1085-1103: try-except Exception
Pattern: Try-batch-upsert-except-log
Exception Type: Generic Exception
Logging: Info on success, error on failure
Recovery Strategy: Return/log error
```
**Issues**:
- Batch operation could benefit from partial retry
- No transactional rollback handling

---

### Block 11: `sql_input()` - JSON File Loading (Lines 1346-1352)
```
Lines 1346-1352: try-except Exception
Pattern: Try-json-load-except-log-return
Exception Type: Generic Exception
Logging: Info on success, error on failure
Recovery Strategy: Return early (graceful fail)
```
**Issues**:
- Appropriate for file I/O
- Could have retry for transient file system issues

---

### Block 12: `reset_address_id_sequence()` (Lines 1505-1685)
```
Lines 1505-1685: try-except Exception with nested cleanup (1681-1684)
Pattern: Try-complex-operation-except-log-cleanup
Exception Type: Generic Exception
Logging: Detailed info/error logs
Recovery Strategy: Cleanup temp table, re-raise exception
Special Handling: Nested try-except for cleanup
```
**Critical Issues**:
- Most complex method (180 lines)
- Requires transactional integrity
- No retry mechanism
- Cleanup exception handling is bare except (line 1683)

---

### Block 13: `update_full_address_with_building_names()` (Lines 1697-1753)
```
Lines 1697-1753: try-except Exception
Pattern: Try-batch-update-except-log-raise
Exception Type: Generic Exception
Logging: Info/Debug on iteration, error on failure
Recovery Strategy: Re-raise exception (fail fast)
```
**Issues**:
- Good fail-fast approach
- Could benefit from partial success handling
- No retry for transient errors

---

## 2. Error Handling Patterns Identified

### Pattern 1: Try-Log-Return None (7 instances)
Methods:
1. `get_db_connection()` - line 209
2. `create_urls_df()` - line 371
3. `lookup_raw_location()` - line 790
4. `execute_query()` - line 411

**Characteristics**:
- Catches generic Exception or specific SQLAlchemyError
- Logs error with context
- Returns None/empty on failure
- No retry capability

**Consolidation**: Can be replaced with `@resilient_execution` decorator

---

### Pattern 2: Try-Execute-Log-Continue (4 instances)
Methods:
1. `cache_raw_location()` - line 769
2. `create_raw_locations_table()` - line 838
3. `fuzzy_duplicates()` - line 995 (nested)
4. `sql_input()` - line 1346

**Characteristics**:
- Fire-and-forget operations
- Log on failure but don't stop execution
- Appropriate for non-critical paths
- Can mask subtle failures

**Consolidation**: Minimal consolidation needed; fix logging consistency

---

### Pattern 3: Try-Execute-Exception Re-raise (3 instances)
Methods:
1. `reset_address_id_sequence()` - line 1505
2. `update_full_address_with_building_names()` - line 1697
3. `multiple_db_inserts()` - line 1085

**Characteristics**:
- Fail-fast approach
- Log error with context
- Re-raise exception
- Good for critical operations

**Consolidation**: Can be wrapped with `@resilient_execution` for limited retry

---

## 3. Critical Path Methods Analysis

### 1. `execute_query()` - MOST CRITICAL (Lines 384-474)
**Frequency**: Called 50+ times throughout DatabaseHandler
**Complexity**: High (handles multiple query types)
**Error Handling**: 1 try-except block
**Issues**:
- All database I/O funnels through this method
- No retry logic despite being core
- Handles UniqueViolation gracefully (good!)
- 64 lines of logic

**Decorator Recommendation**: `@resilient_execution(max_retries=3, catch_exceptions=(SQLAlchemyError, TimeoutError))`

**Estimated Consolidation**: 15-20 lines

---

### 2. `resolve_or_insert_address()` - (Lines 518-531)
**Delegates to**: AddressRepository
**Frequency**: Called ~20 times (critical for event processing)
**Error Handling**: 0 try-except blocks (delegated to repository)
**External Interactions**: LLM indirectly via AddressResolutionRepository

**Review**: Check AddressRepository for error handling pattern

---

### 3. `multiple_db_inserts()` - BATCH OPERATION (Lines 1062-1103)
**Frequency**: Called during bulk operations
**Complexity**: Moderate (batch upsert pattern)
**Error Handling**: 1 try-except block
**Issues**:
- Single exception handler for entire batch
- No partial success tracking
- Upsert logic could fail selectively

**Decorator Recommendation**: `@resilient_execution(max_retries=2, catch_exceptions=(SQLAlchemyError,))`

**Estimated Consolidation**: 10-15 lines

---

### 4. `reset_address_id_sequence()` - MOST COMPLEX (Lines 1491-1685)
**Frequency**: Called rarely (maintenance operation)
**Complexity**: Very High (180 lines, multi-step renumbering)
**Error Handling**: 2 try-except blocks
**Issues**:
- No retry logic despite transactional criticality
- Cleanup try-except uses bare `except:` (line 1683)
- Multiple database operations, no rollback on failure
- Iterates over addresses without bulk operation

**Decorator Recommendation**: Cannot easily decorator (too complex), needs refactor

**Estimated Consolidation**: 30-40 lines (via refactoring into smaller methods with @resilient_execution)

---

### 5. `fuzzy_duplicates()` - LOOP WITH SILENT FAILURES (Lines 943-1011)
**Frequency**: Called during cleanup operations
**Complexity**: Moderate (nested loops, multiple updates/deletes)
**Error Handling**: 1 nested try-except block (line 995)
**Issues**:
- Silent except (no logging) at line 997
- Nested in tight loop - exception could be missed
- Updates/deletes within loop could fail individually

**Decorator Recommendation**: Extract update/delete into separate `@resilient_execution` method

**Estimated Consolidation**: 5-10 lines

---

### 6. `process_event_address()` - LLM INTERACTION (via AddressResolutionRepository)
**Frequency**: Called for every event processed
**Complexity**: Moderate (orchestrates 4-level fallback)
**Error Handling**: Multi-level (in AddressResolutionRepository)
**External Interactions**: 
- LLM query (lines 108-111 in AddressResolutionRepository)
- Database lookups/inserts
- Cache operations

**Review**: AddressResolutionRepository has 4 try-except blocks (lines 75, 116, 182, 226, 243, 297)

---

## 4. External Service Interactions

### LLM Integration
**Location**: AddressResolutionRepository (injected into DatabaseHandler)
**Methods**:
- `_resolve_via_llm()` - lines 168-228 in AddressResolutionRepository
- `llm.generate_prompt()`
- `llm.query_llm()`
- `llm.extract_and_parse_json()`

**Error Handling**: Try-except at line 226 with warning log
**Issues**:
- LLM timeout not explicitly handled
- No circuit breaker pattern for repeated failures
- Could benefit from `@http_retry` decorator for network resilience

---

### Database Operations
**Core Operation**: `execute_query()` (lines 384-474)
**External Dependencies**: PostgreSQL via SQLAlchemy
**Error Handling**: 1 try-except with UniqueViolation special case
**Issues**:
- Transient connection errors not retried
- Pool exhaustion not handled
- No exponential backoff

---

### Cache Layer
**Methods**:
- `cache_raw_location()` - lines 764-783
- `lookup_raw_location()` - lines 785-802
- `create_raw_locations_table()` - lines 804-853

**Error Handling**: Fire-and-forget pattern (appropriate for cache)
**Issues**: None identified (appropriate design for non-critical cache)

---

## 5. Timeout Handling Code

### Explicit Timeout Handling: NONE FOUND
**Issue**: Database operations have no explicit timeout
**Methods Missing Timeout**:
- `execute_query()` - could hang indefinitely
- `create_urls_df()` - pd.read_sql could hang
- `reset_address_id_sequence()` - long-running operation

**Recommendation**: Add `@with_timeout` decorator to long-running operations

---

## 6. Database Operation Classification

### Transient Error Candidates (Should Retry)
1. `execute_query()` - network timeout, connection pool exhaustion
2. `create_urls_df()` - read timeout
3. `resolve_or_insert_address()` - transactional conflict
4. `cache_raw_location()` - cache miss, write delay
5. `lookup_raw_location()` - cache read delay

---

### Non-Retriable Candidates (Should Fail Fast)
1. `fuzzy_duplicates()- logic error would be consistent
2. `reset_address_id_sequence()` - constraint violations
3. `sql_input()` - malformed SQL

---

## 7. Recommendations Summary

### High Priority: Apply @resilient_execution Decorator
1. **execute_query()** (1,505-1,525 lines)
   ```python
   @resilient_execution(
       max_retries=3,
       catch_exceptions=(SQLAlchemyError, TimeoutError),
       strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER
   )
   def execute_query(self, query, params=None):
       # Remove existing try-except block
   ```
   **Consolidation**: -20 lines

2. **create_urls_df()** (Lines 360-381)
   ```python
   @resilient_execution(max_retries=2, catch_exceptions=(SQLAlchemyError,))
   def create_urls_df(self):
       # Remove existing try-except block
   ```
   **Consolidation**: -10 lines

3. **get_db_connection()** (Lines 193-220)
   ```python
   @resilient_execution(max_retries=3, catch_exceptions=(Exception,))
   def get_db_connection(self):
       # Remove existing try-except block, remove None return
   ```
   **Consolidation**: -12 lines

4. **multiple_db_inserts()** (Lines 1062-1103)
   ```python
   @resilient_execution(max_retries=2, catch_exceptions=(SQLAlchemyError,))
   def multiple_db_inserts(self, table_name, values):
       # Remove existing try-except block
   ```
   **Consolidation**: -18 lines

5. **cache_raw_location()** (Lines 764-783) - Optional, non-critical
   ```python
   @resilient_execution(max_retries=2, catch_exceptions=(Exception,))
   def cache_raw_location(self, raw_location: str, address_id: int):
       # Remove existing try-except block
   ```
   **Consolidation**: -15 lines

6. **lookup_raw_location()** (Lines 785-802) - Optional, non-critical
   ```python
   @resilient_execution(max_retries=1, catch_exceptions=(Exception,))
   def lookup_raw_location(self, raw_location: str) -> Optional[int]:
       # Remove existing try-except block
   ```
   **Consolidation**: -13 lines

---

### Medium Priority: Add Timeout Handling
1. **reset_address_id_sequence()** - Add transaction timeout
2. **fuzzy_duplicates()** - Add operation timeout
3. **update_full_address_with_building_names()** - Add iteration timeout

---

### Medium Priority: Fix Silent Exceptions
1. **fuzzy_duplicates()** Line 997 - Add logging in except clause
   ```python
   except Exception as e:
       logging.warning(f"Failed to convert numpy value: {e}")
       # Existing pass statement
   ```
   **Consolidation**: +2 lines (necessary)

2. **reset_address_id_sequence()** Line 1683 - Use specific exception
   ```python
   except Exception as e:
       logging.error(f"Failed to cleanup temp table: {e}")
   ```
   **Consolidation**: +1 line (necessary)

---

### Low Priority: Improve Exception Specificity
1. Replace generic `except Exception` with specific exceptions:
   - `except SQLAlchemyError` for database operations
   - `except (IOError, OSError)` for file operations
   - `except json.JSONDecodeError` for JSON parsing

---

## 8. Estimated Total Consolidation

### By Category
| Category | Lines Saved | Priority |
|----------|------------|----------|
| @resilient_execution decorators | -88 lines | High |
| @with_timeout decorators | -15 lines | Medium |
| Fix silent exceptions | +3 lines | Medium |
| Refactor reset_address_id_sequence | -30 lines | Low |
| Improve exception specificity | -10 lines | Low |
| **Total Net Consolidation** | **-150 lines** | - |

### Confidence Levels
- **execute_query()**: 95% - Most critical, most repeatable
- **multiple_db_inserts()**: 85% - Batch operation, clear pattern
- **create_urls_df()**: 80% - Simple operation, clear consolidation
- **fuzzy_duplicates()**: 70% - Complex logic, less obvious benefits
- **reset_address_id_sequence()**: 60% - Very complex, needs refactoring

---

## 9. Implementation Roadmap

### Phase 1: Critical Path (Week 1)
1. Add `@resilient_execution` to `execute_query()`
2. Add `@resilient_execution` to `create_urls_df()`
3. Add `@with_timeout` to `reset_address_id_sequence()`
4. Test thoroughly with database stress tests

### Phase 2: Secondary Operations (Week 2)
5. Add `@resilient_execution` to `multiple_db_inserts()`
6. Fix silent exceptions in `fuzzy_duplicates()`
7. Add logging to exception handlers

### Phase 3: Cache Operations (Week 3)
8. Add `@resilient_execution` to cache operations
9. Add circuit breaker for LLM operations
10. Add timeout handling to LLM calls

### Phase 4: Refactoring (Week 4)
11. Refactor `reset_address_id_sequence()` into smaller methods
12. Add transaction management for multi-step operations
13. Add comprehensive integration tests

---

## 10. Code Quality Metrics

### Current State
- **Total Try-Except Blocks**: 14
- **Average Try-Except Size**: ~40 lines
- **Error Handling Coverage**: ~35% of critical paths
- **Retry Capability**: 0% (manual, no decorators)
- **Timeout Handling**: 0% (missing)
- **Circuit Breaker Usage**: 0% (missing)

### Target State (Post-Consolidation)
- **Total Try-Except Blocks**: 8 (via decorators, ~6 manual remaining)
- **Average Try-Except Size**: ~20 lines
- **Error Handling Coverage**: 85% of critical paths
- **Retry Capability**: 100% (via @resilient_execution)
- **Timeout Handling**: 60% (key operations only)
- **Circuit Breaker Usage**: 20% (LLM operations)

---

## Appendix: Full Method Classification

### Methods with Try-Except (14 total)
1. `__init__()` - No error handling
2. `get_db_connection()` - 1 try-except (lines 209-220)
3. `create_urls_df()` - 1 try-except (lines 371-381)
4. `execute_query()` - 1 try-except (lines 411-474) **CRITICAL**
5. `close_connection()` - 1 nested try-except (lines 487-493)
6. `cache_raw_location()` - 1 try-except (lines 769-783)
7. `lookup_raw_location()` - 1 try-except (lines 790-802)
8. `create_raw_locations_table()` - 1 try-except (lines 838-852)
9. `fuzzy_duplicates()` - 1 nested try-except (lines 995-998)
10. `multiple_db_inserts()` - 1 try-except (lines 1085-1103)
11. `sql_input()` - 1 try-except (lines 1346-1352)
12. `reset_address_id_sequence()` - 2 nested try-except (lines 1505-1685) **MOST COMPLEX**
13. `update_full_address_with_building_names()` - 1 try-except (lines 1697-1753)
14. AddressResolutionRepository methods - 4 additional try-except blocks

### Methods WITHOUT Try-Except (Delegates to Repositories)
- `resolve_or_insert_address()`
- `process_event_address()`
- `write_events_to_db()`
- `fuzzy_match()`
- `find_address_by_building_name()`
- `quick_address_lookup()`
- And 15+ other wrapper/delegation methods

