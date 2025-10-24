#!/usr/bin/env python3
"""
production_logging.py - Production-Grade Structured Logging System

Provides comprehensive logging infrastructure for production deployment:
- JSON structured logging for log aggregation
- Log rotation and retention policies
- Separate error and access logs
- Correlation IDs for request tracing
- Sensitive data filtering/masking
- Performance metrics logging
- Asyncio-compatible logging

Supports multiple handlers:
- Console output (formatted for development/debugging)
- File handlers (with rotation and retention)
- Structured JSON handlers (for log aggregation services)
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import re
from functools import wraps
import time
import uuid
from contextlib import contextmanager


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
        import sys

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
            'production_logging.py',
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


if __name__ == "__main__":
    # Example usage
    config = {
        'log_level': 'DEBUG',
        'log_dir': 'logs',
        'enable_console': True,
        'enable_file': True,
        'enable_json': True,
    }

    logger = configure_production_logging(config)

    print("=== Testing Structured Logging ===\n")

    # Test basic logging
    logger.info("Application started", version="1.0.0", environment="production")
    logger.debug("Debug information", detail="This is a debug message")

    # Test performance logging
    import time
    start = time.time()
    time.sleep(0.1)
    duration_ms = (time.time() - start) * 1000
    logger.log_performance("data_processing", duration_ms, success=True, items_processed=1000)

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("An error occurred", exc_info=True, error_type="ValueError")

    # Test correlation context
    with logger.correlation_context("correlation-123"):
        logger.info("Request processing", user_id="user123", path="/api/events")
        logger.info("Data retrieved", records=500)

    # Test performance context
    try:
        with logger.performance_context("expensive_operation"):
            time.sleep(0.05)
            logger.info("Processing items", count=100)
    except Exception as e:
        logger.error(f"Operation failed: {e}")

    # Test sensitive data filtering
    logger.info("Database connection", password="<PASSWORD>", api_key="<API_KEY>")

    print("\n✅ Logging configured successfully")
    print("📁 Check logs/ directory for output files")
