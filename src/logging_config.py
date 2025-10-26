"""
Centralized logging configuration for multi-environment support.

Consolidation of three logging modules (logging_config, logging_utils, production_logging)
into a single, unified logging system with support for both simple and advanced scenarios.

Environment Behavior:
    - Local Development: Logs written to files in logs/ directory
    - Render Production: Logs written to stdout for visibility in Render console

Simple Usage:
    from logging_config import setup_logging
    setup_logging('script_name')
    logging.info("This will go to the right place automatically")

Advanced Usage (Production-Grade Structured Logging):
    from logging_config import get_production_logger
    logger = get_production_logger('app_name')
    logger.info("Message with context", user_id=123, action="login")

Utility Functions:
    from logging_config import log_extracted_text, log_extracted_text_summary
    log_extracted_text("function_name", "https://example.com", extracted_content, logger)

Features:
    - Automatic environment detection (RENDER environment variable)
    - Consistent log format across all scripts
    - Creates log directory if needed (local only)
    - Thread-safe and works with multiprocessing
    - JSON structured logging for log aggregation
    - Log rotation and retention policies
    - Sensitive data filtering/masking
    - Performance metrics logging
    - Correlation IDs for request tracing
"""

import logging
import logging.handlers
import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from functools import wraps
import time
import uuid
from contextlib import contextmanager


# ============================================================================
# SIMPLE LOGGING (from original logging_config.py)
# ============================================================================

def setup_logging(script_name: str, level=logging.INFO):
    """
    Configure logging based on execution environment.

    Args:
        script_name (str): Name of the script (used for log filename in local mode)
        level (int): Logging level (default: logging.INFO)

    Environment Detection:
        - RENDER='true': Logs to stdout (for Render console)
        - Otherwise: Logs to logs/{script_name}_log.txt

    Example:
        setup_logging('emails')
        logging.info("Processing emails...")  # Goes to right destination automatically
    """
    # Check if running on Render
    is_render = os.getenv('RENDER') == 'true'

    # Common format for all logs
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'

    if is_render:
        # RENDER MODE: Log to stdout so it appears in Render console
        logging.basicConfig(
            level=level,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True  # Override any existing configuration
        )
        logging.info(f"Logging configured for Render (stdout) - {script_name}")
    else:
        # LOCAL MODE: Log to file
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)

        log_file = f"{log_dir}/{script_name}_log.txt"

        logging.basicConfig(
            filename=log_file,
            filemode='a',
            level=level,
            format=log_format,
            datefmt=date_format,
            force=True  # Override any existing configuration
        )
        logging.info(f"Logging configured for local development (file: {log_file})")


def get_logger(name: str):
    """
    Get a logger instance with the given name.

    This is useful when you want module-specific loggers that still
    respect the centralized configuration.

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
# LOGGING UTILITIES (from logging_utils.py)
# ============================================================================

def log_extracted_text(function_name: str, url: str, extracted_text: str, logger: logging.Logger = None) -> None:
    """
    Log extracted text with first 100 and last 100 characters plus length.

    This prevents log files from being overwhelmed with thousands of characters
    of extracted text while still providing enough context to verify the extraction
    worked correctly.

    Args:
        function_name: Name of the function where extraction occurred (for debugging)
        url: The URL or identifier where the text was extracted from
        extracted_text: The full extracted text content
        logger: Logger instance to use. If None, uses root logger.

    Examples:
        >>> log_extracted_text("extract_event_text", "https://example.com", "Short text", logger)
        INFO - extract_event_text: Extracted 10 chars from https://example.com: Short text

        >>> log_extracted_text("scrape_page", "https://example.com", "A"*500, logger)
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
        logger.info(f"{function_name}: Extracted {text_len} chars from {url}: {extracted_text}")
    else:
        # Show first 100, ellipsis, last 100, and total length
        first_100 = extracted_text[:100].replace('\n', ' ').replace('\r', ' ')
        last_100 = extracted_text[-100:].replace('\n', ' ').replace('\r', ' ')
        logger.info(f"{function_name}: Extracted {text_len:,} chars from {url}: {first_100}......{last_100}")


def log_extracted_text_summary(function_name: str, url: str, extracted_text: str, logger: logging.Logger = None) -> None:
    """
    Log a brief summary of extracted text (just the length).

    Use this when you don't need to see the content at all, just confirmation
    that text was extracted.

    Args:
        function_name: Name of the function where extraction occurred (for debugging)
        url: The URL or identifier where the text was extracted from
        extracted_text: The full extracted text content
        logger: Logger instance to use. If None, uses root logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not extracted_text:
        logger.warning(f"{function_name}: No text extracted from {url}")
        return

    text_len = len(extracted_text)
    logger.info(f"{function_name}: Extracted {text_len:,} chars from {url}")


# ============================================================================
# PRODUCTION LOGGING (from production_logging.py)
# ============================================================================

class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs."""

    # Patterns for sensitive data
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
                record.msg = re.sub(pattern, '****[REDACTED]****', record.msg, flags=re.IGNORECASE)

        # Also filter args if they're strings
        if record.args:
            if isinstance(record.args, dict):
                for key, value in record.args.items():
                    if isinstance(value, str):
                        for pattern in self.PATTERNS.values():
                            record.args[key] = re.sub(pattern, '****[REDACTED]****', value, flags=re.IGNORECASE)
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    re.sub(pattern, '****[REDACTED]****', str(arg), flags=re.IGNORECASE)
                    if isinstance(arg, str)
                    else arg
                    for arg in record.args
                    for pattern in self.PATTERNS.values()
                )

        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

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

        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id

        # Add custom fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and isinstance(record.exc_info, tuple):
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        return json.dumps(log_data, default=str)


class PerformanceFormatter(logging.Formatter):
    """Formatter for performance metrics logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format performance metrics."""
        perf_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': 'performance_metric',
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Extract performance metrics
        if hasattr(record, 'extra_fields'):
            metrics = record.extra_fields
            perf_data.update({
                'duration_ms': metrics.get('duration_ms'),
                'operation': metrics.get('operation'),
                'success': metrics.get('success', True),
                'error': metrics.get('error'),
                'items_processed': metrics.get('items_processed'),
                'rate_per_second': metrics.get('rate_per_second'),
            })

        return json.dumps(perf_data, default=str)


class LogContext:
    """Thread-safe context for correlation IDs and metadata."""

    def __init__(self):
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

    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata."""
        return self.metadata.copy()


class ProductionLogger:
    """Main logger interface for production logging."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.config = config or {}
        self.context = LogContext()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup log handlers based on configuration."""
        log_level = self.config.get('log_level', 'INFO')
        log_dir = self.config.get('log_dir', 'logs')
        enable_console = self.config.get('enable_console', True)
        enable_file = self.config.get('enable_file', True)
        enable_json = self.config.get('enable_json', True)

        # Set log level
        self.logger.setLevel(getattr(logging, log_level))

        # Add sensitive data filter to all handlers
        sensitive_filter = SensitiveDataFilter()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))
            console_handler.addFilter(sensitive_filter)

            if enable_json:
                formatter = JSONFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )

            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler with rotation
        if enable_file:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # Main log file
            log_file = os.path.join(log_dir, f'{self.name}.log')
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10  # Keep 10 backups
            )
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.addFilter(sensitive_filter)

            formatter = JSONFormatter() if enable_json else logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Error log file
            error_file = os.path.join(log_dir, f'{self.name}_error.log')
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=10
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.addFilter(sensitive_filter)
            error_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(error_handler)

    def debug(self, message: str, **extra_fields) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_fields)

    def info(self, message: str, **extra_fields) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra_fields)

    def warning(self, message: str, **extra_fields) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, extra_fields)

    def error(self, message: str, exc_info: bool = False, **extra_fields) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, extra_fields, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False, **extra_fields) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra_fields, exc_info=exc_info)

    def _log(self, level: int, message: str, extra_fields: Dict[str, Any], exc_info: bool = False) -> None:
        """Internal logging method with context."""
        # Prepare extra fields
        fields = {
            'correlation_id': self.context.get_correlation_id(),
            **self.context.get_metadata(),
            **extra_fields
        }

        # Get exception info if requested
        exc_tuple = None
        if exc_info:
            exc_tuple = sys.exc_info()

        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            'logging_config.py',
            0,
            message,
            (),
            exc_tuple
        )
        record.extra_fields = fields
        record.correlation_id = self.context.get_correlation_id()

        self.logger.handle(record)

    def log_performance(self, operation: str, duration_ms: float, success: bool = True,
                        items_processed: int = 0, error: str = None) -> None:
        """Log performance metrics."""
        rate_per_second = (items_processed / (duration_ms / 1000)) if duration_ms > 0 and items_processed else 0

        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            caller_frame := None,
            lineno := 0,
            f'Performance: {operation} completed in {duration_ms:.2f}ms',
            (),
            None
        )

        record.extra_fields = {
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            'success': success,
            'items_processed': items_processed,
            'rate_per_second': round(rate_per_second, 2),
            'error': error,
            'correlation_id': self.context.get_correlation_id(),
        }
        record.correlation_id = self.context.get_correlation_id()

        self.logger.handle(record)

    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for setting correlation ID."""
        old_id = self.context.get_correlation_id()
        if correlation_id:
            self.context.set_correlation_id(correlation_id)
        try:
            yield self.context
        finally:
            self.context.set_correlation_id(old_id)

    @contextmanager
    def performance_context(self, operation: str):
        """Context manager for measuring performance."""
        start_time = time.time()
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self.log_performance(operation, duration_ms, success=True)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log_performance(operation, duration_ms, success=False, error=str(e))
            raise


def log_method_call(operation_name: Optional[str] = None):
    """Decorator to log method calls with performance metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = None

            # Try to get logger from self
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = logging.getLogger(func.__module__)

            op_name = operation_name or f'{func.__module__}.{func.__name__}'
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                if isinstance(logger, ProductionLogger):
                    logger.log_performance(op_name, duration_ms, success=True)
                else:
                    logger.info(f'{op_name} completed in {duration_ms:.2f}ms')

                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                if isinstance(logger, ProductionLogger):
                    logger.log_performance(op_name, duration_ms, success=False, error=str(e))
                else:
                    logger.error(f'{op_name} failed: {str(e)}', exc_info=True)

                raise

        return wrapper
    return decorator


# Global logger instance
_global_logger: Optional[ProductionLogger] = None


def get_production_logger(name: str = __name__, config: Optional[Dict[str, Any]] = None) -> ProductionLogger:
    """Get or create global production logger."""
    global _global_logger

    if _global_logger is None:
        _global_logger = ProductionLogger(name, config)

    return _global_logger


def configure_production_logging(config: Dict[str, Any]) -> ProductionLogger:
    """Configure production logging with custom settings."""
    global _global_logger
    _global_logger = ProductionLogger('scraper', config)
    return _global_logger
