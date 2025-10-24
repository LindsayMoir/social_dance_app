# Phase 10: Direct Repository Migration Plan

**Objective:** Remove all wrapper methods from DatabaseHandler and update calling code to use repositories directly.

**Status:** Planning phase
**Total Calls to Migrate:** 50
**Files Affected:** 12
**Estimated Effort:** 4-6 hours

---

## Overview

This phase removes the intermediary wrapper methods in DatabaseHandler, requiring calling code to directly access the specialized repositories. This completes the refactoring by eliminating the final layer of indirection.

### Impact Summary

| Category | Count |
|----------|-------|
| Total wrapper calls to migrate | 50 |
| Files to modify | 12 |
| Wrapper methods to remove | 6 |
| Pipeline files affected | 4 |
| Supporting modules affected | 4 |
| Repository files affected | 3 |

---

## Migration Strategy

### Phase 10a: Pipeline Files (Highest Priority)
**Files:** fb.py, scraper.py, ebs.py, rd_ext.py
**Calls:** 24 total
**Priority:** HIGH - These are the main entry points

### Phase 10b: Supporting Modules (Medium Priority)
**Files:** images.py, llm.py, read_pdfs.py, clean_up.py
**Calls:** 18 total
**Priority:** MEDIUM - Support the pipeline

### Phase 10c: Repository Interdependencies (Lowest Priority)
**Files:** event_repository.py, address_repository.py, address_resolution_repository.py
**Calls:** 5 total
**Priority:** LOW - Internal to refactored code, handle last

### Phase 10d: Cleanup
**Action:** Remove wrapper methods from DatabaseHandler
**Files:** src/db.py
**Priority:** FINAL - Do after all migrations complete

---

## Detailed Migration Plan

### Phase 10a: Pipeline Files

#### 1. src/fb.py (13 calls)

**Current calls (using wrappers):**
```python
# Line 607, 624, 632, 645, 652, 657, 718, 722, 941, 948, 954
db.write_url_to_db(url_row)

# Line 665, 962
db.write_events_to_db(events_df)
```

**After migration (direct repository calls):**
```python
# Lines where write_url_to_db is called
db.url_repo.write_url_to_db(url_row)

# Lines where write_events_to_db is called
db.event_repo.write_events_to_db(events_df)
```

**Implementation:**
- Replace 11 instances of `db.write_url_to_db()` with `db.url_repo.write_url_to_db()`
- Replace 2 instances of `db.write_events_to_db()` with `db.event_repo.write_events_to_db()`

---

#### 2. src/scraper.py (9 calls)

**Current calls:**
```python
# Line 118, 161, 184, 187, 190, 217, 225, 253
db.write_url_to_db(url_row)

# Line 329
db.write_events_to_db(events_df)
```

**After migration:**
```python
db.url_repo.write_url_to_db(url_row)
db.event_repo.write_events_to_db(events_df)
```

---

#### 3. src/ebs.py (1 call)

**Current call:**
```python
# Line 507
db.write_events_to_db(events_df)
```

**After migration:**
```python
db.event_repo.write_events_to_db(events_df)
```

---

#### 4. src/rd_ext.py (1 call)

**Current call:**
```python
# Line 676
db.write_events_to_db(events_df)
```

**After migration:**
```python
db.event_repo.write_events_to_db(events_df)
```

---

### Phase 10b: Supporting Modules

#### 5. src/images.py (9 calls)

**Current calls:**
```python
# Line 543, 563, 571, 578, 676, 682, 690, 699, 708
db.write_url_to_db(url_row)
```

**After migration:**
```python
db.url_repo.write_url_to_db(url_row)
```

---

#### 6. src/llm.py (5 calls)

**Current calls:**
```python
# Line 187, 192, 197, 203
db.write_url_to_db(url_row)

# Line 269
db.write_events_to_db(events_df)
```

**After migration:**
```python
db.url_repo.write_url_to_db(url_row)
db.event_repo.write_events_to_db(events_df)
```

---

#### 7. src/read_pdfs.py (3 calls)

**Current calls:**
```python
# Line 99, 105, 143
db.write_url_to_db(url_row)
```

**After migration:**
```python
db.url_repo.write_url_to_db(url_row)
```

---

#### 8. src/clean_up.py (1 call)

**Current call:**
```python
# Line 453
db.write_url_to_db(url_row)
```

**Note:** This call may have incorrect parameter order - needs investigation before migration.

**After migration:**
```python
db.url_repo.write_url_to_db(url_row)  # After verifying parameter order
```

---

### Phase 10c: Repository Interdependencies

#### 9. src/repositories/event_repository.py (4 calls)

**Current calls:**
```python
# Line 102 - normalize_nulls
db.normalize_nulls(record)

# Line 103 - process_event_address
db.process_event_address(event)

# Line 116, 128 - write_url_to_db
db.write_url_to_db(url_row)
```

**After migration:**
```python
# Normalize nulls
db.address_data_repo.normalize_nulls(record)

# Process event address
db.address_resolution_repo.process_event_address(event)

# Write URL to database
db.url_repo.write_url_to_db(url_row)
```

---

#### 10. src/repositories/address_repository.py (1 call)

**Current call:**
```python
# Line 182 - normalize_nulls
db.normalize_nulls(full_address)
```

**After migration:**
```python
db.address_data_repo.normalize_nulls(full_address)
```

---

#### 11. src/repositories/address_resolution_repository.py (3 calls)

**Current calls:**
```python
# Line 103, 110 - cache_raw_location
db.cache_raw_location(raw_location, address_id)

# Line 214 - normalize_nulls
db.normalize_nulls(record)
```

**After migration:**
```python
db.location_cache_repo.cache_raw_location(raw_location, address_id)
db.address_data_repo.normalize_nulls(record)
```

---

### Phase 10d: DatabaseHandler Cleanup

#### 12. src/db.py

**Remove wrapper methods:**
```python
# DELETE THESE METHODS (after migration completes):

def write_url_to_db(self, url_row):
    """Remove this wrapper - direct calls will use db.url_repo"""
    return self.url_repo.write_url_to_db(url_row)

def write_events_to_db(self, events_df):
    """Remove this wrapper - direct calls will use db.event_repo"""
    return self.event_repo.write_events_to_db(events_df)

def process_event_address(self, event):
    """Remove this wrapper - direct calls will use db.address_resolution_repo"""
    return self.address_resolution_repo.process_event_address(event)

def normalize_nulls(self, record):
    """Remove this wrapper - direct calls will use db.address_data_repo"""
    return self.address_data_repo.normalize_nulls(record)

def cache_raw_location(self, raw_location, address_id):
    """Remove this wrapper - direct calls will use db.location_cache_repo"""
    return self.location_cache_repo.cache_raw_location(raw_location, address_id)

def sql_input(self, file_path):
    """Remove this wrapper - direct calls will use db.database_maintenance_repo"""
    return self.database_maintenance_repo.sql_input(file_path)
```

---

## Implementation Steps

### Step 1: Prepare (30 minutes)
- [ ] Read this migration plan
- [ ] Review the codebase changes needed
- [ ] Create Phase 10 branch
- [ ] Ensure all current tests pass

### Step 2: Phase 10a - Migrate Pipeline Files (1 hour)
- [ ] Update src/fb.py (13 calls)
- [ ] Update src/scraper.py (9 calls)
- [ ] Update src/ebs.py (1 call)
- [ ] Update src/rd_ext.py (1 call)
- [ ] Run tests after each file
- [ ] Commit: "refactor: Migrate pipeline files to direct repository calls (Phase 10a)"

### Step 3: Phase 10b - Migrate Supporting Modules (1 hour)
- [ ] Update src/images.py (9 calls)
- [ ] Update src/llm.py (5 calls)
- [ ] Update src/read_pdfs.py (3 calls)
- [ ] Investigate and fix src/clean_up.py (1 call)
- [ ] Run tests after each file
- [ ] Commit: "refactor: Migrate supporting modules to direct repository calls (Phase 10b)"

### Step 4: Phase 10c - Migrate Repository Interdependencies (45 minutes)
- [ ] Update src/repositories/event_repository.py (4 calls)
- [ ] Update src/repositories/address_repository.py (1 call)
- [ ] Update src/repositories/address_resolution_repository.py (3 calls)
- [ ] Run tests after each file
- [ ] Commit: "refactor: Migrate repository interdependencies (Phase 10c)"

### Step 5: Phase 10d - Remove Wrapper Methods (30 minutes)
- [ ] Remove write_url_to_db() wrapper from src/db.py
- [ ] Remove write_events_to_db() wrapper from src/db.py
- [ ] Remove process_event_address() wrapper from src/db.py
- [ ] Remove normalize_nulls() wrapper from src/db.py
- [ ] Remove cache_raw_location() wrapper from src/db.py
- [ ] Remove sql_input() wrapper from src/db.py
- [ ] Run full test suite
- [ ] Commit: "refactor: Remove DatabaseHandler wrapper methods (Phase 10d)"

### Step 6: Finalization (15 minutes)
- [ ] Run full test suite (265 tests)
- [ ] Verify all migrations complete
- [ ] Push to remote
- [ ] Update SESSION_CONTINUATION.md
- [ ] Create Phase 10 summary

---

## Testing Strategy

**After Each File Migration:**
```bash
# Run tests to ensure no breakage
pytest tests/unit/ -v --tb=short
```

**Before Removing Wrappers:**
```bash
# Final verification
pytest tests/unit/ -v
```

**Expected:** All 265 tests passing

---

## Potential Issues & Mitigations

### Issue 1: Missing Repository Initialization
**Risk:** Some code paths might not have `db` initialized
**Mitigation:** Ensure all imports are correct and `db` is available where needed

### Issue 2: Parameter Mismatches
**Risk:** Repository method signatures might differ from wrapper signatures
**Mitigation:** Review each migration carefully, run tests after each change

### Issue 3: clean_up.py Parameter Order
**Risk:** Line 453 may have incorrect parameter order
**Mitigation:** Investigate before migration, fix if needed

### Issue 4: Circular Dependencies
**Risk:** Repository interdependencies might create circular imports
**Mitigation:** Tested in Phase 10c, but monitor carefully

---

## Rollback Plan

If issues arise:

1. **Revert to last passing commit:**
   ```bash
   git revert HEAD
   git push origin refactor/code-cleanup-phase2
   ```

2. **Individual file rollback:**
   ```bash
   git checkout HEAD~1 -- src/fb.py
   ```

3. **Full phase rollback:**
   - Switch to main branch
   - Create new branch from pre-Phase-10 state

---

## Success Criteria

✅ **Phase 10 Complete When:**
- [ ] All 50 wrapper calls migrated to direct repository calls
- [ ] All 265 unit tests passing
- [ ] All 6 wrapper methods removed from DatabaseHandler
- [ ] Code compiles and runs without errors
- [ ] Pipeline.py runs successfully
- [ ] All changes committed and pushed
- [ ] Documentation updated

---

## Files Modified Summary

```
Total files to modify: 12
- Pipeline files: 4 (fb.py, scraper.py, ebs.py, rd_ext.py)
- Supporting modules: 4 (images.py, llm.py, read_pdfs.py, clean_up.py)
- Repository files: 3 (event_repo, address_repo, address_resolution_repo)
- DatabaseHandler: 1 (db.py - remove wrappers)
- Documentation: 1 (SESSION_CONTINUATION.md)

Commits planned: 5
- Phase 10a (pipeline files)
- Phase 10b (supporting modules)
- Phase 10c (repository interdependencies)
- Phase 10d (wrapper removal)
- Phase 10 summary/docs
```

---

## Timeline

**Estimated Total Time: 4-6 hours**

- Phase 10a: 1 hour
- Phase 10b: 1 hour
- Phase 10c: 45 minutes
- Phase 10d: 30 minutes
- Testing & Finalization: 1-1.5 hours

---

## Phase 10 Execution Order

```
1. Create Phase 10 branch
2. Migrate Phase 10a (pipeline files)
   → Test
   → Commit
3. Migrate Phase 10b (supporting modules)
   → Test
   → Commit
4. Migrate Phase 10c (repository interdependencies)
   → Test
   → Commit
5. Migrate Phase 10d (remove wrappers)
   → Test
   → Commit
6. Finalize & Push
   → Update documentation
   → Push to remote
```

---

## Notes

- This is the final phase of the refactoring project
- After Phase 10, DatabaseHandler will no longer have wrapper methods
- Code will directly use specialized repositories
- Maximum clean architecture achieved
- All 265 tests must pass

---

**Document Version:** 1.0
**Created:** 2025-10-23
**Status:** Ready for Implementation
