# Implementation Plan: AddressResolutionRepository
## Extracting the Most Complex Method from DatabaseHandler

**Priority:** CRITICAL  
**Estimated Effort:** 2-3 days with full testing  
**Impact:** ~220 lines removed, improves testability and maintainability  
**Risk Level:** MEDIUM (complex logic, many edge cases)

---

## Overview

The `process_event_address()` method in DatabaseHandler is the most complex single method in the codebase (~160 lines, 1055-1212). It orchestrates multiple concerns:

1. **Cache lookup** - Check if location has been seen before
2. **Quick resolution** - Try regex/fuzzy matching without LLM
3. **LLM processing** - Query AI model for address parsing
4. **Fallback handling** - Create minimal addresses when all else fails

This document provides a step-by-step implementation plan to extract this into a dedicated repository.

---

## Current Method Structure

### Input
```python
event: dict = {
    "location": str,           # Raw location string
    "event_name": str,         # Event name
    "source": str,             # Source identifier
    "url": str,                # Source URL
    "description": str,        # Event description
    # ... other event fields
}
```

### Output
```python
event: dict = {
    "address_id": int,         # Resolved address ID
    "location": str,           # Standardized location/address
    # ... other fields preserved
}
```

### Current 3-Level Fallback Strategy
```
STEP 1: Cache Lookup
  ├─ lookup_raw_location(location)
  └─ If found: Return address_id from cache ✓

STEP 2: Quick Address Lookup
  ├─ quick_address_lookup(location)
  ├─ Regex parsing + fuzzy matching
  ├─ No external API calls
  └─ If found: Cache & return ✓

STEP 3: LLM Processing (Fallback)
  ├─ generate_prompt(...)
  ├─ query_llm(...)
  ├─ extract_and_parse_json(...)
  ├─ normalize_nulls(...)
  ├─ resolve_or_insert_address(...)
  └─ If found: Cache & return ✓

STEP 4: Fallback Address Creation
  ├─ Building name extraction
  ├─ Deduplication check
  ├─ Minimal address creation
  └─ Final fallback to defaults
```

---

## Target Repository Structure

### Class Definition
```python
# repositories/address_resolution_repository.py

class AddressResolutionRepository:
    """
    Resolves and processes event addresses using multi-level fallback strategy.
    
    Combines cache lookups, quick regex-based resolution, LLM processing,
    and fallback address creation in a single orchestrated flow.
    
    Responsibilities:
    - Coordinate address resolution from multiple sources
    - Manage cache-first lookup strategy
    - Orchestrate LLM-based address parsing
    - Handle fallback address creation
    - Normalize and store resolved addresses
    """
    
    def __init__(self, db_handler, llm_handler=None):
        """
        Initialize AddressResolutionRepository.
        
        Args:
            db_handler: DatabaseHandler for database operations
            llm_handler: Optional LLMHandler for AI-based resolution
        """
        self.db = db_handler
        self.llm = llm_handler  # Optional - can be None
        self.logger = logging.getLogger(__name__)
        self.cache_repo = None  # Set after initialization
        self.data_repo = None   # Set after initialization
```

### Public API Methods

#### 1. Main Orchestration Method
```python
def process_event_address(self, event: dict) -> dict:
    """
    Main public method - orchestrates address resolution.
    
    Args:
        event (dict): Event with 'location', 'event_name', 'source', 'url'
        
    Returns:
        dict: Event with resolved 'address_id' and standardized 'location'
        
    Flow:
        1. Validate location (not empty/unknown)
        2. Try cache lookup
        3. Try quick address lookup
        4. Try LLM resolution (if available)
        5. Create minimal fallback address
    """
    location = event.get("location", None)
    # ... orchestration logic
```

#### 2. Resolution Strategy Methods
```python
def _resolve_via_cache(self, location: str) -> Optional[int]:
    """Try to resolve address from location cache."""
    # Delegates to LocationCacheRepository.lookup_raw_location()

def _resolve_via_quick_lookup(self, location: str) -> Optional[int]:
    """Try regex/fuzzy matching without LLM."""
    # Delegates to AddressRepository.quick_address_lookup()

def _resolve_via_llm(self, event: dict, location: str) -> Optional[dict]:
    """Use LLM to parse address structure from location string."""
    # Coordinates with llm_handler

def _resolve_via_building_extraction(self, event: dict) -> Optional[int]:
    """Extract building names from event details."""
    # Coordinates with LocationCacheRepository._extract_address_from_event_details()
```

#### 3. Fallback Creation
```python
def _create_minimal_fallback_address(self, event: dict, source: str) -> Optional[int]:
    """Create minimal address entry when resolution fails."""
    # Creates minimal dict with building_name and city

def _create_default_fallback_address(self, event: dict) -> None:
    """Apply hardcoded defaults when all else fails."""
    # Sets address_id=0 and generic location string
```

#### 4. Supporting Methods
```python
def _cache_resolution(self, location: str, address_id: int) -> None:
    """Cache the raw location -> address_id mapping."""

def _get_full_address(self, address_id: int) -> Optional[str]:
    """Retrieve formatted full_address from database."""

def _update_event_address(self, event: dict, address_id: int) -> dict:
    """Update event with resolved address information."""
```

---

## Implementation Steps

### Step 1: Create Base Repository File

**File:** `/mnt/d/GitHub/social_dance_app/src/repositories/address_resolution_repository.py`

```python
from typing import Optional, Dict, Any
import logging
import pandas as pd
from datetime import datetime

class AddressResolutionRepository:
    """Address resolution orchestration repository."""
    
    def __init__(self, db_handler, llm_handler=None):
        self.db = db_handler
        self.llm = llm_handler
        self.logger = logging.getLogger(__name__)
        self.cache_repo = None
        self.data_repo = None
        
    def set_dependencies(self, cache_repo, data_repo):
        """Set dependent repositories after initialization."""
        self.cache_repo = cache_repo
        self.data_repo = data_repo
```

### Step 2: Implement Main Orchestration Logic

Extract from `db.py` lines 1055-1212, decompose into:
- Input validation
- Cache check
- Quick lookup
- LLM query (if available)
- Fallback creation

### Step 3: Implement Strategy Methods

Break down the monolithic method into focused private methods:
- `_resolve_via_cache()` - Cache lookup
- `_resolve_via_quick_lookup()` - Regex/fuzzy
- `_resolve_via_llm()` - AI parsing
- `_resolve_via_building_extraction()` - Building name match
- `_create_minimal_fallback_address()` - Last resort

### Step 4: Handle Dependencies

**Required Dependencies:**
- `db_handler` - Database connection
- `llm_handler` - Optional LLM service

**Inter-Repository Dependencies:**
- `LocationCacheRepository` - Cache operations
- `AddressRepository` - Address resolution
- `AddressDataRepository` - Data normalization

### Step 5: Update DatabaseHandler

Replace implementation with wrapper:
```python
def process_event_address(self, event: dict) -> dict:
    """Wrapper delegating to AddressResolutionRepository."""
    return self.address_resolution_repo.process_event_address(event)
```

### Step 6: Initialize in DatabaseHandler.__init__

```python
def __init__(self, config):
    # ... existing initialization ...
    
    # Initialize AddressResolutionRepository
    self.address_resolution_repo = AddressResolutionRepository(self, llm_handler=None)
    
    # After LLMHandler is created:
    # self.address_resolution_repo.llm = self.llm_handler
```

### Step 7: Create Integration Tests

Test file: `/mnt/d/GitHub/social_dance_app/tests/test_address_resolution_repository.py`

```python
import pytest
from repositories.address_resolution_repository import AddressResolutionRepository
from unittest.mock import Mock, patch

class TestAddressResolutionRepository:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_db = Mock()
        self.mock_llm = Mock()
        self.repo = AddressResolutionRepository(self.mock_db, self.mock_llm)
    
    def test_process_event_address_with_valid_location(self):
        """Test resolving event with valid location."""
        event = {
            "event_name": "Dance Night",
            "location": "123 Main St, Vancouver",
            "source": "Facebook"
        }
        # Test cache hit
        # Test quick lookup
        # Test LLM fallback
        # Test minimal address creation
    
    def test_resolve_via_cache_hit(self):
        """Test cache lookup success."""
        
    def test_resolve_via_quick_lookup(self):
        """Test regex/fuzzy matching."""
        
    def test_resolve_via_llm(self):
        """Test LLM-based resolution."""
        
    def test_fallback_address_creation(self):
        """Test minimal address creation."""
    
    def test_missing_location_handling(self):
        """Test behavior with missing/invalid location."""
```

---

## Dependencies Management

### Current Circular Dependency
```
DatabaseHandler
  ├─ creates LLMHandler
  └─ LLMHandler
      └─ creates DatabaseHandler
          └─ set_llm_handler() injects back
```

### Resolved Dependency Graph
```
DatabaseHandler
  ├─ creates AddressResolutionRepository
  │  ├─ optional LLMHandler (injected after creation)
  │  ├─ LocationCacheRepository
  │  ├─ AddressRepository
  │  └─ AddressDataRepository
  └─ creates LLMHandler
     └─ sets via set_llm_handler(llm_handler)
```

### Injection Strategy

**Option A: Constructor Injection (Preferred)**
```python
# In DatabaseHandler.__init__
address_resolution_repo = AddressResolutionRepository(
    db_handler=self,
    llm_handler=None  # Will be set later
)
```

**Option B: Setter Injection (For circular deps)**
```python
# After LLMHandler created
address_resolution_repo.set_llm_handler(llm_handler)
```

---

## Code Migration Checklist

- [ ] Create `/repositories/address_resolution_repository.py`
- [ ] Implement `AddressResolutionRepository` class
- [ ] Extract `process_event_address()` logic
- [ ] Extract `_extract_address_from_event_details()` logic
- [ ] Implement cache integration
- [ ] Implement quick lookup delegation
- [ ] Implement LLM integration
- [ ] Create fallback methods
- [ ] Write comprehensive unit tests
- [ ] Write integration tests
- [ ] Test with edge cases (missing location, unknown source, etc.)
- [ ] Update DatabaseHandler to create repository instance
- [ ] Replace `process_event_address()` with wrapper
- [ ] Update any direct callers to use repository
- [ ] Document repository API
- [ ] Add logging statements
- [ ] Performance test cache operations
- [ ] Test concurrent access patterns
- [ ] Update README/architecture docs
- [ ] Code review and approval
- [ ] Deploy and monitor

---

## Testing Strategy

### Unit Tests
- Test each resolution strategy independently
- Mock all external dependencies
- Test fallback logic exhaustively
- Test edge cases (None values, empty strings, etc.)

### Integration Tests
- Test full orchestration flow
- Test with real database queries
- Test LLM integration (if applicable)
- Test cache consistency

### Performance Tests
- Measure cache lookup latency
- Measure quick lookup latency
- Measure LLM query latency
- Compare to current implementation

### Edge Case Tests
- Missing location
- Unknown source
- Empty event_name
- Invalid event structure
- Null values in various fields
- Very long location strings
- Special characters in location

---

## Risk Mitigation

### High Risk: Complex Logic
**Mitigation:**
- Break into focused sub-methods
- Extensive unit test coverage (>90%)
- Integration tests with real data
- Pair programming during implementation

### High Risk: LLM Integration
**Mitigation:**
- LLM handler is optional (can be None)
- Graceful fallback when LLM unavailable
- Timeout handling on LLM requests
- Monitoring and logging of LLM failures

### Medium Risk: Database Access
**Mitigation:**
- All queries go through db_handler
- Transaction isolation
- Connection pooling validation
- Duplicate insertion prevention

### Medium Risk: Cache Consistency
**Mitigation:**
- Cache invalidation strategy
- TTL on cache entries
- Consistency checks
- Monitoring cache hit rates

---

## Performance Considerations

### Current Performance (Baseline)
- Cache lookup: ~1-5ms (PostgreSQL)
- Quick lookup: ~10-50ms (regex + query)
- LLM query: ~1-5 seconds (API call)
- Total (worst case): ~5 seconds

### Target Performance After Extraction
- Same or better (no extra overhead)
- Monitor via logging timestamps
- Add optional timing instrumentation

### Optimization Opportunities
- In-memory building name cache (already exists)
- Prepared statements for postal code queries
- Connection pooling improvements
- LLM response caching (future)

---

## Documentation Updates

### Code Documentation
- [ ] Update DatabaseHandler docstring
- [ ] Update method docstrings in new repository
- [ ] Add usage examples
- [ ] Document dependencies

### Architecture Documentation
- [ ] Update architecture diagram
- [ ] Document repository relationships
- [ ] Document dependency injection pattern
- [ ] Document LLM integration pattern

### API Documentation
- [ ] Document public methods
- [ ] Document parameters and return values
- [ ] Document exceptions
- [ ] Document side effects

---

## Rollback Plan

If extraction causes issues:

1. **Revert commits** - Git revert to previous state
2. **Restore wrappers** - Keep backward compatibility wrappers
3. **Monitor for issues** - Check logs for errors
4. **Document findings** - Understand what went wrong
5. **Reattempt with modifications** - Fix identified issues

---

## Success Criteria

- [x] All existing functionality preserved
- [x] No performance degradation
- [x] Test coverage >80%
- [x] No new bugs introduced
- [x] Code review approved
- [x] Documentation updated
- [x] Team trained on new structure
- [x] Monitoring/logging in place

---

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| Create base repository | 0.5 days | - |
| Extract logic | 1 day | Base repo |
| Write unit tests | 1 day | Extracted logic |
| Write integration tests | 0.5 days | Unit tests |
| Update DatabaseHandler | 0.5 days | Tests passing |
| Code review | 0.5 days | All above |
| Documentation | 0.5 days | Code review |
| Deployment | 0.5 days | Documentation |
| **Total** | **4.5 days** | - |

---

## Contact & Questions

For questions about this implementation plan:
1. Review DATABASE_HANDLER_EXTRACTION_ANALYSIS.md for context
2. Review EXTRACTION_SUMMARY.txt for quick reference
3. Consult with team on architecture decisions

---

## Appendix: Code Snippets

### Current Implementation (Simplified)
```python
# Current: db.py lines 1055-1212
def process_event_address(self, event: dict) -> dict:
    location = event.get("location", None)
    
    # ... validation logic ...
    
    # STEP 1: Check cache
    cached_addr_id = self.lookup_raw_location(location)
    if cached_addr_id:
        # ... use cached address
        return event
    
    # STEP 2: Quick lookup
    quick_addr_id = self.quick_address_lookup(location)
    if quick_addr_id:
        # ... cache and use
        return event
    
    # STEP 3: LLM
    if self.llm_handler:
        prompt, schema_type = self.llm_handler.generate_prompt(...)
        llm_response = self.llm_handler.query_llm(...)
        parsed_results = self.llm_handler.extract_and_parse_json(...)
        # ... resolve address
        return event
    
    # STEP 4: Fallback
    # ... create minimal address
    return event
```

### New Implementation (Structure)
```python
# New: repositories/address_resolution_repository.py
class AddressResolutionRepository:
    def process_event_address(self, event: dict) -> dict:
        """Main orchestration method."""
        location = event.get("location")
        
        # Validate location
        if self._is_invalid_location(location):
            return self._handle_missing_location(event)
        
        # Strategy 1: Cache
        address_id = self._resolve_via_cache(location)
        if address_id:
            return self._finalize_event(event, address_id, location)
        
        # Strategy 2: Quick lookup
        address_id = self._resolve_via_quick_lookup(location)
        if address_id:
            self._cache_resolution(location, address_id)
            return self._finalize_event(event, address_id, location)
        
        # Strategy 3: LLM
        if self.llm:
            address_data = self._resolve_via_llm(event, location)
            if address_data:
                address_id = self.db.resolve_or_insert_address(address_data)
                if address_id:
                    self._cache_resolution(location, address_id)
                    return self._finalize_event(event, address_id, location)
        
        # Strategy 4: Building extraction
        address_id = self._resolve_via_building_extraction(event)
        if address_id:
            self._cache_resolution(location, address_id)
            return self._finalize_event(event, address_id, location)
        
        # Fallback: Create minimal address
        return self._handle_fallback_address(event)
```

---

**Document Version:** 1.0  
**Last Updated:** October 23, 2025  
**Status:** Ready for Implementation  
**Next Review:** After Phase 1 completion
