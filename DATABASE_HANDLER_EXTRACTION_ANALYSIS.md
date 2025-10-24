# DatabaseHandler Class - Repository Extraction Analysis
## File: /mnt/d/GitHub/social_dance_app/src/db.py

**Analysis Date:** October 23, 2025  
**Total Methods:** 72 methods  
**Current Class Size:** 2,249 lines

---

## Executive Summary

The DatabaseHandler class exhibits significant violations of the Single Responsibility Principle (SRP) and contains multiple distinct concerns that should be extracted into specialized repositories. This analysis identifies **5 major extraction opportunities** affecting approximately **700+ lines of code** and affecting multiple business domains.

### Current State
- Methods already delegated to repositories: ~25 wrapper methods (backward compatibility)
- Methods still directly implemented: ~47 core methods
- Violation severity: **HIGH** - Mixed concerns spanning data transformation, error handling, caching, and LLM operations

---

## GROUP 1: DATA TRANSFORMATION & NORMALIZATION OPERATIONS
**Priority Level:** HIGH (Violates SRP - Data Transformation Concern)  
**SOLID Principles Violated:** Single Responsibility, Open/Closed

### Methods to Extract

| Method | Lines | Responsibility | Current Pattern |
|--------|-------|-----------------|-----------------|
| `normalize_nulls()` | 1860-1887 | Convert null-like strings/NaN to Python None | Direct implementation |
| `clean_null_strings_in_address()` | 1890-1905 | SQL-based null cleanup in address table | Direct implementation |
| `standardize_postal_codes()` | 1908-1926 | Standardize postal code format (V8N 1S3) | Direct implementation |
| `clean_up_address_basic()` | 542-562 | Clean events using regex/fuzzy matching (no external APIs) | Direct implementation |
| `extract_canadian_postal_code()` | 651-671 | Extract postal code from location string | Direct implementation |
| `is_canadian_postal_code()` | 674-688 | Validate Canadian postal code format | Direct implementation |
| `format_address_from_db_row()` | 810-837 | Format database address row to string | Wrapper (delegates to AddressRepository) |

**Estimated Lines to Extract:** ~180 lines

**Suggested Repository Name:** `AddressDataRepository` or `AddressTransformRepository`

**Rationale:**
- These methods handle data cleaning and transformation specific to addresses
- They are utility/transformation operations, not core address resolution logic
- Should be separated from AddressRepository (which focuses on resolution/matching)
- Used by both database operations and event processing pipeline

**Extraction Recommendation:**
```python
# NEW: repositories/address_data_repository.py
class AddressDataRepository:
    - normalize_nulls(record: dict) -> dict
    - standardize_postal_codes() -> int
    - clean_null_strings_in_address() -> None
    - extract_canadian_postal_code(location_str: str) -> Optional[str]
    - is_canadian_postal_code(postal_code: str) -> bool
    - format_address_from_db_row(db_row) -> str
    - clean_up_address_basic(events_df: DataFrame) -> DataFrame
```

**Dependencies to Inject:**
- db_handler (for execute_query)
- config (for access to configuration)

**Test Coverage Priority:** HIGH - These affect data integrity

---

## GROUP 2: LLM/AI OPERATIONS & PROMPT HANDLING
**Priority Level:** CRITICAL (New Concern - LLM Integration)  
**SOLID Principles Violated:** Single Responsibility, Dependency Inversion

### Methods to Extract

| Method | Lines | Responsibility | Current Pattern |
|--------|-------|-----------------|-----------------|
| `process_event_address()` | 1055-1212 | Complex LLM-based address processing orchestration | Direct implementation |
| `set_llm_handler()` | 152-156 | Inject LLM handler dependency | Direct implementation |
| `_extract_address_from_event_details()` | 997-1053 | Extract building names and fuzzy match to addresses | Direct implementation |

**Estimated Lines to Extract:** ~220 lines

**Suggested Repository Name:** `AddressResolutionRepository` or `LLMAddressRepository`

**Rationale:**
- `process_event_address()` is the most complex method in the entire class (~160 lines)
- Contains multiple concerns: caching, LLM querying, fallback logic, minimal address creation
- Directly couples DatabaseHandler to LLMHandler
- Uses LLM for address inference - distinct from other address operations
- Has 3-level fallback strategy that deserves its own orchestration

**Current LLM Integration:**
```python
# Lines 1155-1161: Direct LLM calls
prompt, schema_type = self.llm_handler.generate_prompt(...)
llm_response = self.llm_handler.query_llm(...)
parsed_results = self.llm_handler.extract_and_parse_json(...)
```

**Extraction Recommendation:**
```python
# ENHANCED: repositories/address_resolution_repository.py
class AddressResolutionRepository:
    def __init__(self, db_handler, llm_handler=None):
        self.db = db_handler
        self.llm = llm_handler
    
    - process_event_address(event: dict) -> dict
    - _extract_address_from_event_details(event: dict) -> Optional[int]
    - _resolve_via_cache(location: str) -> Optional[int]
    - _resolve_via_quick_lookup(location: str) -> Optional[int]
    - _resolve_via_llm(event: dict, location: str) -> Optional[dict]
    - _create_minimal_fallback_address(event: dict, source: str) -> dict
```

**Dependencies to Inject:**
- db_handler (required)
- llm_handler (optional - can be None for non-LLM paths)

**Key Methods Currently Used:**
- `lookup_raw_location()` (caching)
- `quick_address_lookup()` (delegated to AddressRepository)
- `resolve_or_insert_address()` (delegated to AddressRepository)
- `get_full_address_from_id()` (delegated to AddressRepository)
- `find_address_by_building_name()` (delegated to AddressRepository)
- `_extract_address_from_event_details()` (internal)

**Test Coverage Priority:** CRITICAL - Complex business logic

---

## GROUP 3: CACHING & LOOKUP OPERATIONS
**Priority Level:** MEDIUM (Performance Concern)  
**SOLID Principles Violated:** Single Responsibility, Open/Closed

### Methods to Extract

| Method | Lines | Responsibility | Current Pattern |
|--------|-------|-----------------|-----------------|
| `cache_raw_location()` | 1244-1263 | Insert raw_location → address_id mapping to cache | Direct implementation |
| `lookup_raw_location()` | 1265-1282 | Query raw_location cache for address_id | Direct implementation |
| `_get_building_name_dictionary()` | 974-995 | Create and cache building name → address_id lookup | Direct implementation |
| `create_raw_locations_table()` | 1284-1333 | Create raw_locations caching table (DDL) | Direct implementation |

**Estimated Lines to Extract:** ~110 lines

**Suggested Repository Name:** `LocationCacheRepository` or `AddressCacheRepository`

**Rationale:**
- These methods manage location caching (both in-memory and database-backed)
- Form a cohesive caching layer separate from core address operations
- The caching strategy deserves centralization for maintenance and optimization
- Could be replaced with Redis/distributed cache in the future
- Currently spread across multiple concerns

**Extraction Recommendation:**
```python
# NEW: repositories/location_cache_repository.py
class LocationCacheRepository:
    def __init__(self, db_handler):
        self.db = db_handler
        self._building_name_cache = {}
        
    - cache_raw_location(raw_location: str, address_id: int) -> None
    - lookup_raw_location(raw_location: str) -> Optional[int]
    - get_building_name_dictionary() -> dict
    - create_raw_locations_table() -> None
    - invalidate_building_cache() -> None
    - cache_stats() -> dict  # NEW: Monitoring method
```

**Database Tables Managed:**
- `raw_locations` (raw_location_id, raw_location, address_id, created_at)

**Test Coverage Priority:** MEDIUM - Performance and consistency important

---

## GROUP 4: DATA CLEANING & QUALITY OPERATIONS (NOT YET EXTRACTED)
**Priority Level:** MEDIUM (Quality Assurance - Partially Extracted)  
**SOLID Principles Violated:** Single Responsibility

### Methods ALREADY Extracted (Wrappers)
- `dedup()` → EventManagementRepository
- `delete_old_events()` → EventManagementRepository
- `delete_likely_dud_events()` → EventManagementRepository
- `delete_events_with_nulls()` → EventManagementRepository
- `check_dow_date_consistent()` → EventManagementRepository
- `update_dow_date()` → EventManagementRepository
- `sync_event_locations_with_address_table()` → EventAnalysisRepository
- `clean_orphaned_references()` → EventAnalysisRepository

### Methods STILL DIRECTLY IMPLEMENTED

| Method | Lines | Responsibility | Current Pattern |
|--------|-------|-----------------|-----------------|
| `fuzzy_duplicates()` | 1423-1491 | Find and merge fuzzy duplicate events | Direct implementation |
| `is_foreign()` | 1586-1655 | Identify and delete events outside BC/Canada | Direct implementation |
| `check_image_events_exist()` | 1758-1829 | Disable: Force re-scraping of images | Direct implementation |
| `match_civic_number()` | 947-972 | Match civic number from location to address | Direct implementation |

**Estimated Lines to Extract:** ~250 lines

**Suggested Repository Name:** `EventQualityRepository` or Enhanced `EventManagementRepository`

**Rationale:**
- These operations focus on event-level deduplication and quality
- Complement existing EventManagementRepository which handles age/validity filtering
- Should be consolidated for maintainability
- `fuzzy_duplicates()` is particularly complex (~70 lines) and involves data merging logic

**Current Delegation Gap:**
EventManagementRepository only handles:
- Age-based deletion
- Invalid event deletion (missing fields, out-of-region)
- Day-of-week validation

Missing:
- Fuzzy duplicate detection and merging
- Geographic filtering (foreign location detection)
- Image event existence checking

**Extraction Recommendation:**
```python
# ENHANCE: repositories/event_management_repository.py
class EventManagementRepository:
    # Existing methods...
    
    # ADD NEW METHODS:
    - fuzzy_duplicates() -> None
    - is_foreign() -> DataFrame
    - match_civic_number(df: DataFrame, numbers: List[str]) -> Optional[int]
    - check_image_events_exist(image_url: str) -> bool  # Always returns False
```

**Test Coverage Priority:** HIGH - Affects event deduplication

---

## GROUP 5: ADMINISTRATIVE/MAINTENANCE OPERATIONS
**Priority Level:** LOW (Administrative - Specialized Use)  
**SOLID Principles Violated:** Single Responsibility

### Methods to Extract

| Method | Lines | Responsibility | Current Pattern |
|--------|-------|-----------------|-----------------|
| `reset_address_id_sequence()` | 1986-2180 | Renumber addresses sequentially (1, 2, 3...) | Direct implementation |
| `update_full_address_with_building_names()` | 2182-2248 | Rebuild full_address for all records | Direct implementation |
| `sql_input()` | 1832-1857 | Execute SQL statements from JSON file | Direct implementation |
| `standardize_postal_codes()` | 1908-1926 | (Also in Group 1) | Direct implementation |

**Estimated Lines to Extract:** ~320 lines

**Suggested Repository Name:** `DatabaseMaintenanceRepository` or `AddressMaintenanceRepository`

**Rationale:**
- These are specialized admin operations, not core business logic
- Rarely called (mostly for maintenance/cleanup cycles)
- Should be clearly separated from operational code
- Deserve their own test suite and monitoring
- Carry high risk (affects entire address table)

**Risk Profile:**
- `reset_address_id_sequence()`: HIGH RISK - Requires database lock, affects all tables
- `update_full_address_with_building_names()`: MEDIUM RISK - Read-only then batch update
- `sql_input()`: MEDIUM RISK - Executes arbitrary SQL
- `standardize_postal_codes()`: LOW RISK - Single table, idempotent

**Extraction Recommendation:**
```python
# NEW: repositories/database_maintenance_repository.py
class DatabaseMaintenanceRepository:
    def __init__(self, db_handler, config):
        self.db = db_handler
        self.config = config
        
    - reset_address_id_sequence() -> int
    - update_full_address_with_building_names() -> int
    - sql_input(file_path: str) -> None
    - standardize_postal_codes() -> int
    - clean_orphaned_references() -> int
    - get_maintenance_status() -> dict  # NEW: Health check
```

**Test Coverage Priority:** LOW - Admin only, needs integration tests

---

## GROUP 6: SUPPORTING UTILITY METHODS
**Priority Level:** LOW (Infrastructure/Utility)  
**SOLID Principles Violated:** Single Responsibility (mild)

### Methods to Keep or Deprecate

| Method | Lines | Responsibility | Recommendation |
|--------|-------|-----------------|-----------------|
| `fuzzy_match()` | 936-942 | Wrapper around fuzz.token_sort_ratio() | Move to FuzzyMatcher utility |
| `create_address_dict()` | 511-539 | Create dict from address components | Keep in AddressRepository |
| `decide_preferred_row()` | 1365-1398 | Compare event rows for dedup | Move to EventQualityRepository |
| `update_preferred_row_from_other()` | 1400-1420 | Merge fields from duplicate event | Move to EventQualityRepository |
| `get_address_update_for_event()` | 618-648 | Extract street number and match to address | Keep in AddressRepository |
| `match_civic_number()` | 947-972 | Match civic number in list to DataFrame | Keep in AddressRepository or utilities |
| `populate_from_db_or_fallback()` | 691-747 | Lookup address by postal code | Keep in AddressRepository |
| `groupby_source()` | 1658-1670 | Count events per source | Move to EventAnalysisRepository |

**Estimated Lines to Relocate:** ~150 lines

**Rationale:**
- These utility methods should follow their primary data entity
- `fuzzy_match()` should use centralized FuzzyMatcher class
- Event comparison methods belong in EventQualityRepository
- Address construction methods belong in AddressRepository

---

## METHODS ALREADY DELEGATED (BACKWARD COMPATIBILITY WRAPPERS)
**Status:** GOOD - Already following repository pattern

| Method | Delegates To | Lines |
|--------|-------------|-------|
| `load_blacklist_domains()` | URLRepository | 159-168 |
| `avoid_domains()` | URLRepository | 170-183 |
| `write_url_to_db()` | URLRepository | 491-508 |
| `write_events_to_db()` | EventRepository | 840-857 |
| `_rename_google_calendar_columns()` | EventRepository | 860-873 |
| `_convert_datetime_fields()` | EventRepository | 875-885 |
| `_clean_day_of_week_field()` | EventRepository | 887-900 |
| `_filter_events()` | EventRepository | 902-915 |
| `update_event()` | EventRepository | 918-933 |
| `resolve_or_insert_address()` | AddressRepository | 750-763 |
| `build_full_address()` | AddressRepository | 766-799 |
| `get_full_address_from_id()` | AddressRepository | 801-807 |
| `format_address_from_db_row()` | AddressRepository | 810-837 |
| `find_address_by_building_name()` | AddressRepository | 1215-1229 |
| `quick_address_lookup()` | AddressRepository | 1231-1242 |
| `sync_event_locations_with_address_table()` | EventAnalysisRepository | 1336-1341 |
| `clean_orphaned_references()` | EventAnalysisRepository | 1343-1348 |
| `dedup()` | EventManagementRepository | 1350-1355 |
| `fetch_events_dataframe()` | EventRepository | 1358-1363 |
| `delete_old_events()` | EventManagementRepository | 1494-1499 |
| `delete_likely_dud_events()` | EventManagementRepository | 1502-1507 |
| `delete_event()` | EventRepository | 1510-1515 |
| `delete_events_with_nulls()` | EventManagementRepository | 1518-1523 |
| `delete_event_with_event_id()` | EventRepository | 1526-1531 |
| `delete_multiple_events()` | EventRepository | 1534-1539 |
| `stale_date()` | URLRepository | 1689-1702 |
| `normalize_url()` | URLRepository | 1705-1718 |
| `should_process_url()` | URLRepository | 1720-1739 |
| `update_dow_date()` | EventManagementRepository | 1742-1747 |
| `check_dow_date_consistent()` | EventManagementRepository | 1750-1755 |
| `count_events_urls_start()` | EventAnalysisRepository | 1673-1678 |
| `count_events_urls_end()` | EventAnalysisRepository | 1681-1686 |

**Total:** 31 wrapper methods (good coverage of core operations)

---

## ARCHITECTURAL RECOMMENDATIONS

### Extraction Priority (Recommended Sequence)

**Phase 1 - CRITICAL (Do First):** LLM & Process Operations
1. Extract `AddressResolutionRepository` with `process_event_address()` orchestration
2. Extract `LocationCacheRepository` for location caching
3. Move `AddressDataRepository` for normalization/transformation

**Phase 2 - HIGH (Do Second):** Event Quality Operations
1. Enhance `EventManagementRepository` with fuzzy_duplicates() and is_foreign()
2. Extract supporting methods (match_civic_number, etc.)

**Phase 3 - MEDIUM (Do Third):** Maintenance Operations
1. Extract `DatabaseMaintenanceRepository` for admin operations
2. Consolidate reset_address_id_sequence() and full_address updates

**Phase 4 - LOW (Polish):** Utility Methods
1. Relocate fuzzy_match() to FuzzyMatcher utility
2. Clean up remaining edge cases

### Code Removal Estimate

| Repository | Lines Removed | Methods | Priority |
|------------|----------------|---------|----------|
| AddressResolutionRepository | 220 | 3 | CRITICAL |
| LocationCacheRepository | 110 | 4 | CRITICAL |
| AddressDataRepository | 180 | 7 | HIGH |
| Enhanced EventManagementRepository | 250 | 4 | HIGH |
| DatabaseMaintenanceRepository | 320 | 4 | MEDIUM |
| **TOTAL** | **~1,080** | **22** | - |

**Resulting db.py Size:** 2,249 → ~1,169 lines (48% reduction)

### Updated DatabaseHandler Responsibilities

After extraction, DatabaseHandler should focus on:
1. Database connection management (init, close_connection, get_db_connection)
2. Low-level query execution (execute_query)
3. Table creation (create_tables)
4. URL DataFrame management (create_urls_df)
5. Repository initialization and delegation

---

## SOLID PRINCIPLES ANALYSIS

### Single Responsibility Violations
- **CRITICAL:** `process_event_address()` handles 3 distinct concerns:
  - Cache lookup
  - LLM processing
  - Fallback address creation
  
- **HIGH:** Data normalization mixed with address operations
- **HIGH:** Event quality operations spread across multiple methods
- **MEDIUM:** Caching logic intertwined with address resolution

### Open/Closed Principle Violations
- Cannot extend LLM behavior without modifying `process_event_address()`
- Caching strategy cannot be replaced (e.g., with Redis) without modifying DatabaseHandler
- Normalization rules hardcoded - difficult to extend with new types

### Dependency Inversion Issues
- DatabaseHandler directly depends on LLMHandler
- No interface abstraction for LLM operations
- Circular dependency requires `set_llm_handler()` injection

### Cohesion Issues
- Low cohesion between address operations and event operations
- Mixing operational code with maintenance code
- Database schema operations (DDL) mixed with data operations (DML)

---

## IMPLEMENTATION NOTES

### Migration Strategy
1. Create new repositories with full method implementations
2. Update DatabaseHandler methods to delegate to repositories
3. Keep wrapper methods for backward compatibility (initially)
4. Update all callers to use repositories directly
5. Remove wrapper methods in final cleanup

### Testing Strategy
- Each new repository needs unit tests for extracted methods
- Integration tests for `process_event_address()` (complex orchestration)
- Performance tests for caching operations
- Admin operations need separate integration test suite

### Documentation Requirements
- Update docstrings for method moves
- Document new repository responsibilities
- Add architecture diagram showing repository relationships
- Document LLM integration patterns

---

## DEPENDENCIES & INTERACTIONS

### AddressResolutionRepository Dependencies
```
AddressResolutionRepository
  ├─ db_handler (for execute_query, get_full_address_from_id)
  ├─ llm_handler (optional, for LLM operations)
  ├─ AddressRepository (for resolve_or_insert_address, quick_address_lookup)
  ├─ LocationCacheRepository (for cache operations)
  └─ AddressDataRepository (for data normalization)
```

### LocationCacheRepository Dependencies
```
LocationCacheRepository
  ├─ db_handler (for execute_query, access to metadata)
  └─ No LLM or address resolution dependencies
```

### EventManagementRepository (Enhanced) Dependencies
```
EventManagementRepository
  ├─ db_handler (for execute_query, config)
  ├─ EventRepository (for fetch_events_dataframe)
  ├─ FuzzyMatcher (for fuzzy_duplicates)
  └─ AddressRepository (for geography checks)
```

---

## METRICS & IMPACT

### Before Extraction
- DatabaseHandler: 2,249 lines
- Methods: 72 total
- Cyclomatic Complexity: HIGH (estimated >50)
- SRP Violations: 15+ distinct concerns

### After Extraction
- DatabaseHandler: ~1,169 lines (48% reduction)
- Methods: ~50 core methods
- New repositories: 5 additional classes
- Estimated complexity reduction: 30-40%

### Code Duplication Reduction
- Current wrappers could be removed once migration complete
- ~300 lines of wrapper methods become candidates for deprecation

---

## RISK ASSESSMENT

### HIGH RISK
- `process_event_address()` extraction - Complex logic, many edge cases
- `reset_address_id_sequence()` - Requires database lock, affects referential integrity

### MEDIUM RISK
- LocationCacheRepository - Affects performance if not done correctly
- Fuzzy duplicates consolidation - Needs thorough testing
- LLM integration - Circular dependency handling

### LOW RISK
- Data transformation utilities - Straightforward, testable
- Admin operations extraction - Low usage frequency

---

## CONCLUSION

The DatabaseHandler class suffers from mixing too many distinct concerns. The recommended extraction of 5 new repositories will:

1. **Improve Maintainability:** Clear separation of concerns
2. **Enable Scaling:** Each repository can evolve independently
3. **Enhance Testability:** Smaller, focused test suites per repository
4. **Reduce Complexity:** ~48% reduction in DatabaseHandler size
5. **Support Extensibility:** Caching and LLM strategies can be swapped
6. **Follow SOLID:** Better adherence to all SOLID principles

**Recommended First Step:** Extract AddressResolutionRepository to decompose the most complex method (`process_event_address()` at 160 lines), which will immediately improve code clarity and testability.

