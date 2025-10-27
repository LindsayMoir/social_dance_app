# Logging System - Quick Reference

## Current State Summary

| Module | Lines | Status | Usage |
|--------|-------|--------|-------|
| **logging_config.py** | 98 | ACTIVE | 22 files import it |
| **logging_utils.py** | 72 | ACTIVE | 2 files import it |
| **production_logging.py** | 448 | UNUSED | 0 files import it |

## What Each Module Does

### logging_config.py (RECOMMENDED FOR CURRENT USE)
```python
from logging_config import setup_logging, get_logger

# Call once per script/module at startup
setup_logging('script_name')

# Get logger instance for use
logger = get_logger(__name__)
logger.info("Your message here")
```
- Detects RENDER environment variable
- Logs to file locally, stdout on Render
- Simple, focused, actively used

### logging_utils.py (USED BY 2 FILES)
```python
from logging_utils import log_extracted_text

# Smart logging that prevents log file bloat
log_extracted_text('function_name', 'url', long_text_content, logger)
```
- Logs first 100 + last 100 chars of extracted text
- Shows total character count
- Prevents massive log files

### production_logging.py (UNUSED - NOT RECOMMENDED)
- Comprehensive structured logging
- JSON output, sensitive data masking, correlation IDs
- Performance metrics, context management
- **Not integrated** - don't use for new code

## Files That Need Updates

### Import Changes Required (2 files):

**1. src/fb_v2.py (line 47)**
```python
# BEFORE
from logging_utils import log_extracted_text

# AFTER
from logging_config import log_extracted_text
```

**2. src/clean_up.py (line 15)**
```python
# BEFORE
from logging_utils import log_extracted_text

# AFTER
from logging_config import log_extracted_text
```

### Files Using logging_config (22 - NO CHANGES NEEDED):
- gs.py, gen_scraper.py, pipeline.py, app.py, llm.py, clean_up.py
- irrelevant_rows.py, images.py, emails.py, main.py, upload_auth_to_db.py
- ebs_v2.py, db.py, dedup_llm.py, credentials.py, read_pdfs_v2.py
- Plus 5 test files

## Implementation Plan

### Phase 1: Consolidate (1-2 days)
Merge logging_config.py + logging_utils.py + useful parts of production_logging.py

### Phase 2: Update Imports (1 hour)
Change 2 files: fb_v2.py and clean_up.py

### Phase 3: Documentation (2-6 hours)
Add deprecation notices, update docs

### Phase 4: Cleanup (Future)
Can remove production_logging.py and old logging_utils.py after deployment

## Key Benefits

- **Single source of truth** - One logging module
- **Backward compatible** - All existing code works unchanged
- **Opt-in features** - Advanced features available when needed
- **Zero downtime** - Gradual adoption possible

## Future Capabilities (Built-In)

Once consolidated, easily add:
- Structured JSON logging
- Sensitive data masking (passwords, tokens)
- Performance metrics tracking
- Correlation IDs for distributed tracing
- Log rotation and archival

## Testing Before Deploy

```bash
# Verify backward compatibility
python -m pytest tests/

# Check all imports still work
grep -r "from logging_config" src/
grep -r "from logging_utils" src/

# Verify log files created correctly
python src/main.py  # Should create logs/ directory
```

## Important Notes

- logging_config uses `force=True` to override root logger
- All setup should happen at module load time, not in functions
- log_extracted_text prevents huge log files from text content
- RENDER environment variable controls behavior (file vs stdout)

## Questions?

See full analysis: LOGGING_CONSOLIDATION_REPORT.md
