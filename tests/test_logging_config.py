"""Tests for centralized logging sanitization."""

import logging
import sys
from io import StringIO

sys.path.insert(0, "src")
from logging_config import (
    SensitiveQueryParameterFilter,
    _install_sensitive_query_parameter_filter,
    redact_sensitive_query_parameters,
)


def test_redact_sensitive_query_parameters_masks_key_variants() -> None:
    message = (
        "GET https://example.test/events?key=calendar-secret&api_key=api-secret&"
        "apikey=legacy-secret&safe=value"
    )

    sanitized = redact_sensitive_query_parameters(message)

    assert "calendar-secret" not in sanitized
    assert "api-secret" not in sanitized
    assert "legacy-secret" not in sanitized
    assert "key=<redacted>" in sanitized
    assert "api_key=<redacted>" in sanitized
    assert "apikey=<redacted>" in sanitized
    assert "safe=value" in sanitized


def test_sensitive_query_parameter_filter_redacts_format_arguments() -> None:
    record = logging.LogRecord(
        name="urllib3.connectionpool",
        level=logging.DEBUG,
        pathname=__file__,
        lineno=1,
        msg="request URL: %s",
        args=("https://example.test/calendar?key=calendar-secret",),
        exc_info=None,
    )

    result = SensitiveQueryParameterFilter().filter(record)

    assert result is True
    assert "calendar-secret" not in record.getMessage()
    assert "key=<redacted>" in record.getMessage()


def test_install_sensitive_query_parameter_filter_sanitizes_handler_output() -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    try:
        _install_sensitive_query_parameter_filter()
        root_logger.info("request URL: https://example.test/calendar?key=calendar-secret")
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)

    assert "calendar-secret" not in stream.getvalue()
    assert "key=<redacted>" in stream.getvalue()
