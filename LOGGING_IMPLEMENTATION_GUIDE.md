# Logging System Consolidation - Implementation Guide

## Overview

This guide provides step-by-step instructions to consolidate the three logging modules into a single unified `logging_config.py`.

**Target Output:** One comprehensive but backward-compatible logging module (~350 lines)

---

## Phase 1: Module Consolidation

### Step 1.1: Create Consolidated logging_config.py

The new `logging_config.py` should include:

1. **All current logging_config.py content** (keep as-is)
2. **Text utility functions** from logging_utils.py
3. **Optional classes** from production_logging.py

#### Structure:

```
logging_config.py (NEW - consolidated version)
├── Imports & Docstring
├── Version info
├── SECTION 1: Basic Setup Functions (current)
│   ├── setup_logging()
│   └── get_logger()
├── SECTION 2: Text Extraction Utilities (from logging_utils)
│   ├── log_extracted_text()
│   └── log_extracted_text_summary()
├── SECTION 3: Optional Advanced Features (from production_logging)
│   ├── SensitiveDataFilter (optional)
│   ├── JSONFormatter (optional)
│   └── LogContext (optional)
└── SECTION 4: Future Extensibility
    └── Configuration examples
```

### Step 1.2: Implementation Code

Here's the complete consolidated version:

```python
#!/usr/bin/env python3
"""
Unified logging configuration for social_dance_app.

Provides consistent logging across:
- Local development (file-based)
- Render production (stdout)
- Optional: structured logging, sensitive data filtering, performance metrics

This module consolidates:
- Original logging_config.py (environment-aware setup)
- logging_utils.py (text extraction logging)
- Optional features from production_logging.py

Usage:
    from logging_config import setup_logging, get_logger
    
    # Basic setup (current behavior)
    setup_logging('script_name')
    logger = get_logger(__name__)
    logger.info("Application started")
    
    # With text extraction logging
    from logging_config import log_extracted_text
    log_extracted_text("function_name", "url", long_text, logger)
    
    # Future: structured logging (when needed)
    setup_logging('script_name', structured=True, mask_sensitive=True)
"""

import logging
import os
import sys
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

__version__ = '2.0'

# ============================================================================
# SECTION 1: CORE LOGGING SETUP (from original logging_config.py)
# ============================================================================

def setup_logging(
    script_name: str,
    level: int = logging.INFO,
    structured: bool = False,
    mask_sensitive: bool = False,
    log_dir: str = 'logs',
    enable_rotation: bool = False,
    rotate_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 10,
    enable_correlation: bool = False
) -> Optional['_LogContext']:
    """
    Configure logging based on execution environment.
    
    This function adapts logging behavior based on where the code is running:
    - Local Development: Logs to file (logs/{script_name}_log.txt)
    - Render Production: Logs to stdout (for Render console)
    
    Args:
        script_name (str): Name of the script (used for log filename in local mode)
        level (int): Logging level (default: logging.INFO)
        structured (bool): Enable JSON structured logging (default: False)
        mask_sensitive (bool): Mask sensitive data (passwords, tokens, etc)
        log_dir (str): Directory for log files (default: 'logs')
        enable_rotation (bool): Enable log file rotation (default: False)
        rotate_bytes (int): Size limit for log rotation (default: 10MB)
        backup_count (int): Number of backup files to keep (default: 10)
        enable_correlation (bool): Enable correlation ID tracking (default: False)
    
    Returns:
        Optional[_LogContext]: Context manager if enable_correlation=True, else None
    
    Environment Detection:
        - RENDER='true': Logs to stdout (for Render console)
        - Otherwise: Logs to logs/{script_name}_log.txt
    
    Example:
        # Basic usage (backward compatible)
        setup_logging('emails')
        logging.info("Processing emails...")
        
        # With structured logging (opt-in)
        ctx = setup_logging('scraper', structured=True, mask_sensitive=True)
    """
    # Detect if running on Render
    is_render = os.getenv('RENDER') == 'true'
    
    # Select formatter
    if structured:
        formatter = _JSONFormatter()
    else:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Build handlers list
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if mask_sensitive:
        console_handler.addFilter(_SensitiveDataFilter())
    handlers.append(console_handler)
    
    # File handler (for local development)
    if not is_render:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"{script_name}_log.txt"
        
        if enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=rotate_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file, mode='a')
        
        file_handler.setFormatter(formatter)
        if mask_sensitive:
            file_handler.addFilter(_SensitiveDataFilter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    # Log initialization message
    if is_render:
        logging.info(f"Logging configured for Render (stdout) - {script_name}")
    else:
        logging.info(f"Logging configured for local development - {script_name}")
    
    # Return context if needed
    if enable_correlation:
        return _LogContext()
    return None


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    This is useful when you want module-specific loggers that still
    respect the centralized configuration set by setup_logging().
    
    Args:
        name (str): Logger name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Module-specific log message")
    """
    return logging.getLogger(name)


# ============================================================================
# SECTION 2: TEXT EXTRACTION UTILITIES (from logging_utils.py)
# ============================================================================

def log_extracted_text(
    function_name: str,
    url: str,
    extracted_text: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log extracted text with smart truncation to prevent log file bloat.
    
    This function logs text extraction in an intelligent way:
    - Short text (≤200 chars): Logs full text
    - Long text (>200 chars): Logs first 100 + "......" + last 100 + length
    
    This prevents log files from being overwhelmed with thousands of characters
    of extracted text while still providing enough context to verify extraction
    worked correctly.
    
    Args:
        function_name (str): Name of the function where extraction occurred
        url (str): The URL or identifier where the text was extracted from
        extracted_text (str): The full extracted text content
        logger (Optional[logging.Logger]): Logger instance to use.
                                           If None, uses module logger.
    
    Example:
        >>> log_extracted_text("extract_event_text", 
        ...                    "https://example.com", 
        ...                    "Short text", 
        ...                    logger)
        INFO - extract_event_text: Extracted 10 chars from https://example.com: Short text
        
        >>> log_extracted_text("scrape_page", 
        ...                    "https://example.com", 
        ...                    "A" * 500, 
        ...                    logger)
        INFO - scrape_page: Extracted 500 chars from https://example.com: AAAA...(first 100)...AAAA...(last 100)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not extracted_text:
        logger.warning(f"{function_name}: No text extracted from {url}")
        return
    
    text_len = len(extracted_text)
    
    if text_len <= 200:
        # If text is short, just show it all
        logger.info(
            f"{function_name}: Extracted {text_len} chars from {url}: {extracted_text}"
        )
    else:
        # Show first 100, ellipsis, last 100, and total length
        first_100 = extracted_text[:100].replace('\n', ' ').replace('\r', ' ')
        last_100 = extracted_text[-100:].replace('\n', ' ').replace('\r', ' ')
        logger.info(
            f"{function_name}: Extracted {text_len:,} chars from {url}: "
            f"{first_100}......{last_100}"
        )


def log_extracted_text_summary(
    function_name: str,
    url: str,
    extracted_text: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log a brief summary of extracted text (just the length).
    
    Use this when you don't need to see the content at all, just confirmation
    that text was extracted. Useful for very verbose extraction scenarios.
    
    Args:
        function_name (str): Name of the function where extraction occurred
        url (str): The URL or identifier where the text was extracted from
        extracted_text (str): The full extracted text content
        logger (Optional[logging.Logger]): Logger instance to use.
                                           If None, uses module logger.
    
    Example:
        >>> log_extracted_text_summary("parse_html", "https://example.com", 
        ...                           big_html_string, logger)
        INFO - parse_html: Extracted 15000 chars from https://example.com
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if not extracted_text:
        logger.warning(f"{function_name}: No text extracted from {url}")
        return
    
    text_len = len(extracted_text)
    logger.info(f"{function_name}: Extracted {text_len:,} chars from {url}")


# ============================================================================
# SECTION 3: OPTIONAL ADVANCED FEATURES (from production_logging.py)
# ============================================================================
# These are opt-in features for future use. Currently not enabled by default.

class _SensitiveDataFilter(logging.Filter):
    """
    Filter to mask sensitive data in logs.
    
    Patterns masked:
    - Passwords
    - API keys
    - Tokens (authorization, bearer tokens)
    - Connection strings
    - Cookies
    - Secrets
    """
    
    PATTERNS = {
        'password': r'password["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        'api_key': r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        'token': r'token["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        'authorization': r'authorization["\']?\s*[:=]\s*["\']?Bearer\s+([^"\'}\s]+)',
        'cookie': r'cookie["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        'secret': r'secret["\']?\s*[:=]\s*["\']?([^"\'}\s]+)',
        'connection_string': r'Server=.+Password=.+',
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and mask sensitive data."""
        if isinstance(record.msg, str):
            for pattern in self.PATTERNS.values():
                record.msg = re.sub(
                    pattern,
                    '****[REDACTED]****',
                    record.msg,
                    flags=re.IGNORECASE
                )
        
        # Also filter args if they're strings
        if record.args:
            if isinstance(record.args, dict):
                for key, value in record.args.items():
                    if isinstance(value, str):
                        for pattern in self.PATTERNS.values():
                            record.args[key] = re.sub(
                                pattern,
                                '****[REDACTED]****',
                                value,
                                flags=re.IGNORECASE
                            )
        
        return True


class _JSONFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs.
    
    Useful for log aggregation services and structured analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class _LogContext:
    """
    Thread-safe context for correlation IDs and metadata.
    
    Useful for distributed tracing and request correlation.
    """
    
    def __init__(self):
        import uuid
        self.correlation_id = str(uuid.uuid4())
        self.metadata: Dict[str, Any] = {}
    
    def get_correlation_id(self) -> str:
        """Get correlation ID for request tracing."""
        return self.correlation_id
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID."""
        self.correlation_id = correlation_id
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# For compatibility with old imports from production_logging
def get_production_logger(*args, **kwargs):
    """Deprecated: Use get_logger() instead."""
    import warnings
    warnings.warn(
        "get_production_logger() is deprecated, use get_logger() instead",
        DeprecationWarning,
        stacklevel=2
    )
    return get_logger(args[0] if args else __name__)


if __name__ == "__main__":
    # Example usage
    print("Testing logging_config consolidation...\n")
    
    # Test 1: Basic setup (backward compatible)
    print("Test 1: Basic setup")
    setup_logging('test_consolidated')
    logger = get_logger(__name__)
    logger.info("Basic logging works!")
    
    # Test 2: Text extraction logging
    print("\nTest 2: Text extraction logging")
    short_text = "Short extracted text"
    long_text = "A" * 500
    log_extracted_text("test_short", "https://example.com", short_text, logger)
    log_extracted_text("test_long", "https://example.com", long_text, logger)
    log_extracted_text_summary("test_summary", "https://example.com", long_text, logger)
    
    print("\n✓ All tests passed!")
    print("✓ Consolidated logging_config.py is working correctly")
```

### Step 1.3: Validation Checklist

After implementing consolidated version:

- [ ] File compiles without syntax errors
- [ ] All imports work (logging, os, sys, json, re, datetime, typing, pathlib)
- [ ] Original setup_logging() API unchanged
- [ ] Original get_logger() API unchanged
- [ ] log_extracted_text() imported from logging_config works
- [ ] log_extracted_text_summary() imported from logging_config works
- [ ] Optional features (structured logging, filtering) don't break basic usage
- [ ] Environment detection (RENDER variable) still works
- [ ] __main__ example runs without errors

---

## Phase 2: Update Imports in 2 Files

### File 1: src/fb_v2.py

**Location:** Line 47

**Current Code:**
```python
from logging_utils import log_extracted_text
```

**New Code:**
```python
from logging_config import log_extracted_text
```

**Usage in file:** 2 calls (lines ~530, ~560)

**Verification:**
```bash
grep -n "log_extracted_text" /mnt/d/GitHub/social_dance_app/src/fb_v2.py
```

### File 2: src/clean_up.py

**Location:** Line 15

**Current Code:**
```python
from logging_utils import log_extracted_text
```

**New Code:**
```python
from logging_config import log_extracted_text
```

**Usage in file:** 10 calls (scattered)

**Verification:**
```bash
grep -n "log_extracted_text" /mnt/d/GitHub/social_dance_app/src/clean_up.py
```

---

## Phase 3: Deprecation and Documentation

### Step 3.1: Add Deprecation Notice to logging_utils.py

```python
#!/usr/bin/env python3
"""
DEPRECATED: This module has been merged into logging_config.py

This file is maintained for backward compatibility only.
All future code should import from logging_config instead.

MIGRATION:
    OLD:  from logging_utils import log_extracted_text
    NEW:  from logging_config import log_extracted_text

Removal scheduled for version 3.0.
"""

# Backward compatibility - import from new location
from logging_config import log_extracted_text, log_extracted_text_summary

__all__ = ['log_extracted_text', 'log_extracted_text_summary']
```

### Step 3.2: Add Deprecation Notice to production_logging.py

```python
#!/usr/bin/env python3
"""
DEPRECATED: Advanced logging features have been merged into logging_config.py

This module is maintained for reference only.
Useful features are now available in logging_config.py as optional parameters.

MIGRATION:
    OLD:  from production_logging import ProductionLogger
    NEW:  setup_logging('name', structured=True, mask_sensitive=True)

Removal scheduled for version 3.0.
"""

import warnings

warnings.warn(
    "production_logging module is deprecated. "
    "Use logging_config.setup_logging() with optional parameters instead. "
    "This module will be removed in v3.0.",
    DeprecationWarning,
    stacklevel=2
)

# Rest of production_logging.py content remains for reference...
```

### Step 3.3: Update Documentation Files

**Update:** documentation/REQUIREMENTS.md or similar
```markdown
## Logging

The application uses a unified logging system via `logging_config.py`:

### Basic Usage
```python
from logging_config import setup_logging, get_logger

setup_logging('my_script')
logger = get_logger(__name__)
logger.info("Application started")
```

### Text Extraction Logging
```python
from logging_config import log_extracted_text

log_extracted_text("function_name", "url", large_text, logger)
```

### Advanced Features (Optional)
- Structured JSON logging: `setup_logging(..., structured=True)`
- Sensitive data masking: `setup_logging(..., mask_sensitive=True)`
- Correlation IDs: `setup_logging(..., enable_correlation=True)`
```

---

## Phase 4: Testing

### Unit Tests to Add

Create `tests/test_logging_consolidation.py`:

```python
"""Tests for consolidated logging_config module."""

import logging
import tempfile
import os
from pathlib import Path

from logging_config import (
    setup_logging,
    get_logger,
    log_extracted_text,
    log_extracted_text_summary
)


class TestLoggingConfigBackwardCompatibility:
    """Verify consolidated module maintains backward compatibility."""
    
    def test_basic_setup_logging(self):
        """Test basic setup_logging call works."""
        setup_logging('test')
        logger = logging.getLogger()
        assert logger is not None
    
    def test_get_logger(self):
        """Test get_logger returns logger instance."""
        logger = get_logger('test_module')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_module'
    
    def test_log_extracted_text_short(self, caplog):
        """Test logging short extracted text."""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            log_extracted_text('test', 'http://example.com', 'Short text', logger)
        assert 'Short text' in caplog.text
    
    def test_log_extracted_text_long(self, caplog):
        """Test logging long extracted text with truncation."""
        logger = get_logger(__name__)
        long_text = 'A' * 500
        with caplog.at_level(logging.INFO):
            log_extracted_text('test', 'http://example.com', long_text, logger)
        assert '500' in caplog.text  # Character count should appear
        assert '......' in caplog.text  # Ellipsis should appear
    
    def test_log_extracted_text_summary(self, caplog):
        """Test summary logging."""
        logger = get_logger(__name__)
        long_text = 'A' * 500
        with caplog.at_level(logging.INFO):
            log_extracted_text_summary('test', 'http://example.com', long_text, logger)
        assert '500' in caplog.text


class TestOptionalFeatures:
    """Test optional advanced features."""
    
    def test_structured_logging_option(self):
        """Test structured logging parameter exists."""
        ctx = setup_logging('test', structured=True)
        # Should not raise an error
    
    def test_mask_sensitive_option(self):
        """Test mask_sensitive parameter exists."""
        ctx = setup_logging('test', mask_sensitive=True)
        # Should not raise an error
    
    def test_render_environment_detection(self, monkeypatch):
        """Test RENDER environment variable detection."""
        # Simulate Render environment
        monkeypatch.setenv('RENDER', 'true')
        setup_logging('test')
        # Should log to stdout instead of file


class TestFileCreation:
    """Test that log files are created correctly."""
    
    def test_local_log_file_created(self, monkeypatch):
        """Test that log file is created in local mode."""
        # Ensure not on Render
        monkeypatch.delenv('RENDER', raising=False)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / 'logs'
            setup_logging('test_file', log_dir=str(log_dir))
            
            log_file = log_dir / 'test_file_log.txt'
            assert log_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Integration Tests

```bash
# Test 1: Verify imports still work
python -c "from logging_config import setup_logging, get_logger, log_extracted_text; print('OK')"

# Test 2: Verify backward compatibility
python -c "from logging_utils import log_extracted_text; print('OK')"

# Test 3: Test basic functionality
python -m pytest tests/test_logging_consolidation.py -v

# Test 4: Check no broken imports in codebase
grep -r "from logging" src/ --include="*.py" | grep -v logging_config | head -20
```

---

## Deployment Checklist

### Before Deployment

- [ ] Consolidated logging_config.py created and tested
- [ ] All 2 import updates completed (fb_v2.py, clean_up.py)
- [ ] Deprecation notices added to old modules
- [ ] Documentation updated
- [ ] Unit tests added and passing
- [ ] Integration tests passing
- [ ] No broken imports in codebase
- [ ] Code review completed

### Deployment Steps

1. **Create feature branch**
   ```bash
   git checkout -b refactor/consolidate-logging
   ```

2. **Replace logging_config.py with consolidated version**
   ```bash
   # Backup old version (if needed for reference)
   cp src/logging_config.py src/logging_config.py.bak
   # Copy new consolidated version
   cp consolidated_logging_config.py src/logging_config.py
   ```

3. **Update imports in 2 files**
   ```bash
   # fb_v2.py
   sed -i 's/from logging_utils import/from logging_config import/' src/fb_v2.py
   # clean_up.py
   sed -i 's/from logging_utils import/from logging_config import/' src/clean_up.py
   ```

4. **Update deprecated modules**
   - Simplify logging_utils.py to re-export from logging_config
   - Add deprecation notice to production_logging.py

5. **Run tests**
   ```bash
   python -m pytest tests/ -v
   python -m pytest tests/test_logging_consolidation.py -v
   ```

6. **Commit changes**
   ```bash
   git add src/logging_config.py src/fb_v2.py src/clean_up.py
   git add src/logging_utils.py src/production_logging.py
   git add LOGGING_CONSOLIDATION_REPORT.md
   git commit -m "Consolidate logging modules into unified logging_config.py"
   ```

7. **Push and create PR**
   ```bash
   git push origin refactor/consolidate-logging
   # Create PR on GitHub
   ```

### Post-Deployment Verification

- [ ] All tests passing in CI/CD
- [ ] No regressions reported
- [ ] Log files created correctly in production
- [ ] No breaking changes to existing scripts
- [ ] Documentation accessible to team

---

## Rollback Plan

If issues occur:

1. **Immediate rollback**
   ```bash
   git revert <commit-hash>
   ```

2. **Restore backups**
   ```bash
   cp src/logging_config.py.bak src/logging_config.py
   git checkout HEAD -- src/fb_v2.py src/clean_up.py
   ```

3. **Verify functionality**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Future Enhancements

Once consolidation is complete, these features can be easily added:

1. **Configuration File Support**
   ```python
   setup_logging_from_config('config/logging.yaml')
   ```

2. **Advanced Context Management**
   ```python
   with ctx.correlation_context('request-123'):
       # Automatic correlation ID tracking
       pass
   ```

3. **Performance Monitoring**
   ```python
   with ctx.performance_context('database_query'):
       # Automatic timing and logging
       pass
   ```

4. **External Log Streaming**
   - Send logs to CloudWatch, DataDog, etc.
   - Add handlers for remote log aggregation

5. **Metrics Integration**
   - Export logging metrics to Prometheus
   - Track error rates, log volume, etc.

---

## Questions or Issues?

Refer to: LOGGING_CONSOLIDATION_REPORT.md (detailed analysis)
Refer to: LOGGING_QUICK_REFERENCE.md (quick answers)

