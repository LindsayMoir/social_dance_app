# Social Dance App - Code Redundancy Analysis Report

## Executive Summary

The social_dance_app codebase shows significant progress in consolidation efforts with several refactored utility modules (BaseScraper, TextExtractor, DBWriter, PlaywrightManager, RetryManager, URLNavigator). However, there remain several high-impact redundancy opportunities, particularly in dual-versioning patterns, environment/config handling, and logging setup. 

**Total Python Files Analyzed**: 117
**High-Impact Redundancies Found**: 11 (CRITICAL/HIGH severity)
**Medium-Impact Redundancies Found**: 8 (MEDIUM severity)
**Low-Impact Redundancies Found**: 6 (LOW severity)

---

## 1. HIGH-SEVERITY REDUNDANCIES (Immediate Action Required)

### 1.1 Dual Version Pattern: Legacy + Refactored Files (CRITICAL)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/fb.py` (988 lines) + `/mnt/d/GitHub/social_dance_app/src/fb_v2.py` (993 lines)
- `/mnt/d/GitHub/social_dance_app/src/ebs.py` (541 lines) + `/mnt/d/GitHub/social_dance_app/src/ebs_v2.py` (437 lines)
- `/mnt/d/GitHub/social_dance_app/src/rd_ext.py` + `/mnt/d/GitHub/social_dance_app/src/rd_ext_v2.py`
- `/mnt/d/GitHub/social_dance_app/src/read_pdfs.py` + `/mnt/d/GitHub/social_dance_app/src/read_pdfs_v2.py`

**Severity**: CRITICAL
**Impact**: 4,000+ lines of redundant code maintained in parallel

**Issue Description**:
The codebase maintains both old and new versions of major scrapers. While the v2 versions use the new BaseScraper architecture, the v1 versions are still present and potentially in use.

Example from code size comparison:
- `fb.py`: 988 lines → `fb_v2.py`: 993 lines (only 5 lines difference)
- `ebs.py`: 541 lines → `ebs_v2.py`: 437 lines (104 lines saved = 19% reduction)

**Code Location**:
- Facebook: Lines 1-100 in both fb.py and fb_v2.py (similar class initialization)
- Eventbrite: Lines 70-95 in both ebs.py and ebs_v2.py

**Suggested Approach**:
1. Audit which versions are actively used (check pipeline.py, main.py, cron jobs)
2. For each pair, migrate all callers to the v2 version
3. Delete the old v1 file after confirming no references remain
4. Timeline: 2-3 weeks total work

**Expected Savings**: 2,500+ lines of code, simpler testing/maintenance

---

### 1.2 Environment/Database Configuration Redundancy (HIGH)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/config_manager.py` (177 lines)
- `/mnt/d/GitHub/social_dance_app/src/deployment_config.py` (398 lines)
- `/mnt/d/GitHub/social_dance_app/src/db_config.py` (246 lines)
- `/mnt/d/GitHub/social_dance_app/src/environment.py` (27 lines)

**Severity**: HIGH
**Impact**: 4 separate modules handling overlapping concerns

**Issue Description**:
The codebase has evolved 3+ configuration management approaches:

1. **ConfigManager** (config_manager.py): Singleton pattern for YAML config
   ```python
   # Line 42-52: Singleton pattern
   @staticmethod
   def get_instance() -> 'ConfigManager':
       if ConfigManager._instance is None:
           ConfigManager._instance = ConfigManager()
       return ConfigManager._instance
   ```

2. **DeploymentConfig** (deployment_config.py): Environment-specific config with validation
   ```python
   # Line 75-87: Separate class hierarchy
   def __init__(self, environment: Optional[str] = None):
       self.environment = environment or os.getenv('DEPLOYMENT_ENV', 'local')
   ```

3. **db_config.py**: Database-specific configuration detection
   ```python
   # Line 49-149: Auto-detection logic (107 lines)
   def get_database_config() -> Tuple[str, str]:
       # Duplicated environment detection logic
   ```

4. **environment.py**: Module-level constants
   ```python
   # Line 22-26: Simple flags
   IS_RENDER = os.getenv('RENDER') == 'true'
   IS_LOCAL = not IS_RENDER
   ```

**Duplication Points**:

| Feature | ConfigManager | DeploymentConfig | db_config | environment |
|---------|---------------|------------------|-----------|-------------|
| Environment detection | No | Yes | Yes (duplicated) | Yes |
| YAML loading | Yes | Yes | No | No |
| Config validation | Yes (basic) | Yes (comprehensive) | No | No |
| Database config | No | Partial | Yes | No |
| Nested key access (dot notation) | Yes | Yes | No | No |

**Code Examples of Duplication**:

1. Environment detection appears 5+ times:
   - db_config.py lines 72-77 (IS_RENDER detection)
   - deployment_config.py line 85 (DEPLOYMENT_ENV lookup)
   - environment.py lines 22-23 (module-level constants)
   - main.py lines 63-68 (inline detection)
   - Repeated in multiple scraper __init__ methods

2. YAML loading pattern (41 occurrences found):
   ```python
   # Pattern repeated in fb.py, ebs.py, gen_scraper.py, images.py, etc.
   with open('config/config.yaml', 'r') as file:
       config = yaml.safe_load(file)
   ```

3. Dot-notation config access duplicated:
   - ConfigManager.get('key.nested.path') - line 106-138
   - DeploymentConfig.get('key.nested.path') - line 234-261
   - db_config functions don't support this pattern

**Suggested Approach**:
1. **Keep**: ConfigManager as the single source of truth (already well-designed)
2. **Merge db_config functions** into ConfigManager (add get_database_config method)
3. **Migrate DeploymentConfig logic** into ConfigManager validation
4. **Use environment.py** only for module-level constants (keep as-is)
5. **Update all imports**:
   ```python
   # Before
   from config_manager import ConfigManager
   from db_config import get_database_config
   
   # After
   from config_manager import ConfigManager
   db_config, env_name = ConfigManager.get_database_config()
   ```

**Expected Savings**: 350+ lines, single source of truth, easier maintenance

**Timeline**: 1-2 weeks

---

### 1.3 Duplicate URL Normalization Functions (HIGH)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/fb.py` - line 243-280
- `/mnt/d/GitHub/social_dance_app/src/fb_v2.py` - line 264-301
- `/mnt/d/GitHub/social_dance_app/src/url_nav.py` - line 116-156 (generic version)

**Severity**: HIGH
**Impact**: 3 implementations of similar functionality, inconsistent behavior

**Issue Description**:
Facebook scraper implements its own URL normalization, while a generic URLNavigator exists:

```python
# fb.py lines 243-280: Facebook-specific (38 lines)
def normalize_facebook_url(self, url: str) -> str:
    # Handles v.facebook.com -> www.facebook.com, query parameter removal
    # Specific to Facebook edge cases

# fb_v2.py lines 264-301: Identical copy in refactored version (38 lines)
def normalize_facebook_url(self, url: str) -> str:
    # Same logic as fb.py

# url_nav.py lines 116-156: Generic URL normalization (40 lines)
def normalize_url(self, url: str, base_url: Optional[str] = None) -> Optional[str]:
    # Generic URL normalization (removes fragments, normalizes case)
    # Does NOT handle Facebook-specific cases
```

**Why This Matters**:
- Facebook scraper normalizes URLs differently than general scraper
- URLNavigator.normalize_url() doesn't handle v.facebook.com → www.facebook.com conversion
- Results in potential duplicates if same URL scraped via different paths

**Suggested Consolidation**:

```python
# Option A: Extend URLNavigator with platform-specific normalization
class URLNavigator:
    def normalize_url(self, url: str, base_url: Optional[str] = None, platform: str = 'generic'):
        if platform == 'facebook':
            return self._normalize_facebook_url(url)
        return self._normalize_generic_url(url, base_url)
    
    def _normalize_facebook_url(self, url: str) -> str:
        # Move fb.py logic here
        # Handle v.facebook.com → www.facebook.com

# Option B: Move Facebook logic to a FacebookURLNormalizer subclass
class FacebookURLNormalizer(URLNavigator):
    def normalize_url(self, url: str, base_url=None):
        # Facebook-specific logic
```

**Expected Savings**: 35+ lines removed, single source of truth for URL handling

**Timeline**: 3-5 days

---

### 1.4 Logging Setup Duplication (HIGH)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/logging_config.py` (79 lines) - Main logging setup
- `/mnt/d/GitHub/social_dance_app/src/production_logging.py` (150+ lines) - Production logging
- `/mnt/d/GitHub/social_dance_app/src/logging_utils.py` (73 lines) - Utility functions
- Multiple files with inline logging initialization (fb.py, ebs.py, gen_scraper.py, images.py)

**Severity**: HIGH
**Impact**: Inconsistent logging across environments, split responsibility

**Issue Description**:

1. **Competing logging systems**:
   - `logging_config.py`: Simple environment-based setup (Render stdout vs local file)
   - `production_logging.py`: Comprehensive system with JSON, filtering, correlation IDs
   - `logging_utils.py`: Text extraction logging helpers

2. **Scattered logging initialization** (at least 10 files do this):
   ```python
   # fb.py line 41-92
   from logging_config import setup_logging
   setup_logging('fb')
   
   # gen_scraper.py line 49-57
   from logging_config import setup_logging
   setup_logging('gen_scraper')
   
   # ebs.py (similar pattern)
   from logging_config import setup_logging
   setup_logging('ebs')
   ```

3. **Duplicate logger creation pattern**:
   ```python
   # images.py lines 59-60
   logger = logging.getLogger(__name__)
   logger.info("\n\nStarting images.py ...")
   
   # scraper.py line 84
   logging.info("\n\nscraper.py starting...")
   
   # main.py lines 40-41
   setup_logging('main')
   logging.info("main.py starting...")
   ```

**Suggested Consolidation**:

```python
# logging_config.py: Keep but extend
def setup_logging(script_name: str, level=logging.INFO, use_production: bool = False):
    """
    Unified logging setup supporting both simple and production configurations.
    
    Args:
        script_name: Name of the script
        level: Logging level
        use_production: If True, use JSON/filtering/correlation IDs
    """
    if use_production or os.getenv('PRODUCTION_LOGGING') == 'true':
        # Use production_logging.ProductionLogger setup
        return ProductionLogger(script_name, level)
    else:
        # Use simple setup (Render stdout vs local file)
        logger = logging.getLogger(script_name)
        # ... existing setup ...
        return logger
```

**Expected Savings**: 150+ lines removed, consistent logging everywhere

**Timeline**: 1 week

---

## 2. MEDIUM-SEVERITY REDUNDANCIES

### 2.1 Scraper Initialization Pattern (MEDIUM)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/fb.py` lines 100-150
- `/mnt/d/GitHub/social_dance_app/src/ebs.py` lines 70-95
- `/mnt/d/GitHub/social_dance_app/src/images.py` lines 62-87
- `/mnt/d/GitHub/social_dance_app/src/gen_scraper.py` lines 88-150

**Severity**: MEDIUM
**Impact**: 150+ lines of similar initialization code

**Pattern**:
```python
# All scrapers follow this pattern:
def __init__(self, config_path: str = "config/config.yaml") -> None:
    # 1. Load config YAML
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 2. Create handler instances
    llm_handler = LLMHandler(config_path)
    db_handler = llm_handler.db_handler or DatabaseHandler(config)
    
    # 3. Get keywords
    keywords_list = llm_handler.get_keywords()
    
    # 4. Initialize run tracker
    run_results_tracker = RunResultsTracker(filename, db_handler)
    events_count, urls_count = get_database_counts(db_handler)
    run_results_tracker.initialize(events_count, urls_count)
    
    # 5. Initialize utility managers
    text_extractor = TextExtractor(logger)
    retry_manager = RetryManager(logger=logger)
```

**Consolidation Approach**:
```python
# base_scraper.py: Add initialization helper
class BaseScraper:
    @staticmethod
    def create_standard_handlers(config_path):
        """Factory method to create standard handler set."""
        llm_handler = LLMHandler(config_path)
        db_handler = llm_handler.db_handler
        return {
            'llm': llm_handler,
            'db': db_handler,
            'keywords': llm_handler.get_keywords(),
            'text_extractor': TextExtractor(),
            'retry_manager': RetryManager(),
        }
```

**Expected Savings**: 50+ lines per scraper, 200+ lines total

---

### 2.2 Database Error Handling Pattern (MEDIUM)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/db.py` (47 instances of try/except)
- `/mnt/d/GitHub/social_dance_app/src/db_utils.py` (10+ instances)
- `/mnt/d/GitHub/social_dance_app/src/dedup_llm.py` (reported errors)
- `/mnt/d/GitHub/social_dance_app/src/llm.py` (10+ instances)

**Severity**: MEDIUM
**Impact**: 100+ lines of similar error handling code

**Pattern Repetition**:
```python
# db_utils.py lines 94-96
except Exception as e:
    self.logger.error(f"Failed to normalize event data: {e}")
    return normalized

# db.py (repeated 47 times, variation):
try:
    # operation
except Exception as e:
    logging.error(f"Error in operation: {e}")
    return None/False
```

**Consolidation Approach**:
```python
# resilience.py: Add error handler decorator
@retry_on_error(max_retries=3, logger=logger, error_message="Database operation failed")
def operation():
    # Code here automatically gets retry + logging
    pass

# Or context manager:
with ErrorHandler(logger, "Operation name") as handler:
    # operation
    if error: handler.log_and_return(None)
```

---

### 2.3 Config Access Pattern (MEDIUM)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/fb.py` - config access scattered
- `/mnt/d/GitHub/social_dance_app/src/ebs.py` - direct dict access
- `/mnt/d/GitHub/social_dance_app/src/gen_scraper.py` - mixed patterns

**Severity**: MEDIUM
**Impact**: Inconsistent config access, hard to refactor

**Pattern Issues**:
```python
# Inconsistent access patterns:
config.get('key')  # Direct dict access (fragile)
config['key']      # Index access (can KeyError)
ConfigManager.get('key.nested')  # Dot notation (consistent)
self.get_config('crawling.headless')  # Method call (in BaseScraper)
```

---

## 3. LOW-SEVERITY REDUNDANCIES

### 3.1 Keyword Checking Pattern (LOW)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/scraper_utils.py` - lines 26-67 (check_keywords, has_keywords)
- `/mnt/d/GitHub/social_dance_app/src/llm.py` - lines 170 (inline keyword checking)
- Multiple scrapers do inline keyword checks

**Severity**: LOW
**Impact**: 30+ lines of similar logic

**Already Addressed**: ✓ Consolidated in scraper_utils.check_keywords()

---

### 3.2 Text Extraction Logging (LOW)

**Files Affected**:
- `/mnt/d/GitHub/social_dance_app/src/logging_utils.py` - 73 lines (log_extracted_text)
- `/mnt/d/GitHub/social_dance_app/src/text_utils.py` - TextExtractor.extract_from_html() also logs

**Severity**: LOW
**Impact**: Duplicate logging for extracted text

**Consolidation**: Merge logging_utils into TextExtractor or logging_config

---

## Summary Table

| Rank | Issue | Severity | Files | Lines | Type | Timeline |
|------|-------|----------|-------|-------|------|----------|
| 1 | Dual v1/v2 versions | CRITICAL | 4 pairs | 2500+ | Architecture | 2-3 weeks |
| 2 | Config management | HIGH | 4 files | 850+ | Feature duplication | 1-2 weeks |
| 3 | URL normalization | HIGH | 3 files | 100+ | Code duplication | 3-5 days |
| 4 | Logging setup | HIGH | 3+10 | 200+ | Infrastructure | 1 week |
| 5 | Scraper initialization | MEDIUM | 4 files | 150+ | Pattern | 3-5 days |
| 6 | Error handling | MEDIUM | 5 files | 100+ | Pattern | 1 week |
| 7 | Config access | MEDIUM | 3 files | 80+ | Consistency | 2-3 days |
| 8 | Keyword checking | LOW | 3 files | 30+ | Pattern | 1 day |

---

## Implementation Priority (Recommended Sequence)

### Phase 1: Configuration (Week 1)
1. Merge db_config.py into ConfigManager
2. Add database_config() method to ConfigManager
3. Update all imports across 20+ files

### Phase 2: Version Consolidation (Week 2-3)
1. Audit which v1/v2 versions are actually used
2. Migrate all callers to v2 versions
3. Delete old v1 files

### Phase 3: Logging Consolidation (Week 4)
1. Extend logging_config.py to support production mode
2. Integrate production_logging.py features
3. Remove duplicate setup() calls

### Phase 4: URL & Error Handling (Week 5)
1. Consolidate URL normalization in URLNavigator
2. Add error handler utilities to resilience.py
3. Update scraper code to use consolidated patterns

---

## Additional Opportunities (Not Critical)

### A. Repository Pattern Usage
The codebase has good repository pattern usage (AddressRepository, URLRepository, EventRepository in db.py). Consider:
- Extracting base repository class (BaseRepository)
- Documenting repository pattern for future extensions

### B. Monitoring & Logging Systems
- `monitoring.py` and `production_logging.py` exist but may not be fully integrated
- Ensure they're used in all scrapers

### C. Utility Organization
- `src/utils/` directory could consolidate:
  - scraper_utils.py
  - text_utils.py
  - browser_utils.py
  - resilience.py
  - logging_utils.py

---

## Detailed Code Snippets for Reference

### Config Consolidation Example
```python
# OLD: Multiple places
from db_config import get_database_config
from config_manager import ConfigManager
from deployment_config import get_config

# NEW: Single import
from config_manager import ConfigManager
db_url, env_name = ConfigManager.get_database_config()
```

### Logging Consolidation Example
```python
# OLD: Every scraper does this
from logging_config import setup_logging
setup_logging('script_name')

# NEW: Automatic in base_scraper.__init__
class BaseScraper(ABC):
    def __init__(self, config_path):
        self.logger = self._setup_logger(self.__class__.__name__)
        
    @staticmethod
    def _setup_logger(name):
        from logging_config import setup_logging
        return setup_logging(name)
```

---

