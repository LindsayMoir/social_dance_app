"""
Resilience utilities for error handling and retry logic in web scrapers.

This module consolidates retry patterns, exponential backoff, and error handling
logic used across multiple scrapers (fb.py, scraper.py, rd_ext.py, images.py, ebs.py).

Classes:
    RetryManager: Centralized retry and exponential backoff management

Key responsibilities:
    - Retry logic with configurable attempts and delays
    - Exponential backoff with jitter
    - Error classification and handling
    - Circuit breaker pattern support
"""

import logging
import random
import time
from typing import Optional, Callable, Any, TypeVar, List
from functools import wraps
from enum import Enum


class RetryStrategy(Enum):
    """Retry strategy enumeration."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"


T = TypeVar('T')


class RetryManager:
    """
    Centralized retry and resilience management for web scrapers.

    Consolidates retry logic, exponential backoff, and error handling
    patterns that are repeated across all scrapers.
    """

    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0  # seconds
    DEFAULT_MAX_DELAY = 60.0  # seconds
    DEFAULT_STRATEGY = RetryStrategy.EXPONENTIAL_WITH_JITTER

    # Retriable error types
    RETRIABLE_ERRORS = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError,
    )

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 max_delay: float = DEFAULT_MAX_DELAY,
                 strategy: RetryStrategy = DEFAULT_STRATEGY,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize RetryManager.

        Args:
            max_retries (int): Maximum number of retry attempts
            base_delay (float): Base delay in seconds
            max_delay (float): Maximum delay in seconds
            strategy (RetryStrategy): Retry strategy to use
            logger (logging.Logger, optional): Logger instance
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.logger = logger or logging.getLogger(__name__)

    def calculate_delay(self, attempt: int, custom_jitter: bool = True) -> float:
        """
        Calculate delay for retry attempt based on strategy.

        Args:
            attempt (int): Current attempt number (0-indexed)
            custom_jitter (bool): Whether to add jitter to the delay

        Returns:
            float: Delay in seconds
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay

        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)

        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)

        elif self.strategy == RetryStrategy.EXPONENTIAL_WITH_JITTER:
            delay = self.base_delay * (2 ** attempt)
            if custom_jitter:
                # Add ±10% jitter to avoid thundering herd
                jitter = delay * 0.1 * random.uniform(-1, 1)
                delay += jitter

        else:
            delay = self.base_delay

        # Cap at max delay
        return min(delay, self.max_delay)

    def retry_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Retry a synchronous function with configured strategy.

        Args:
            func (Callable): Function to retry
            *args: Positional arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            T: Return value from function
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except self.RETRIABLE_ERRORS as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts failed for {func.__name__}: {e}"
                    )

            except Exception as e:
                # Non-retriable error - fail immediately
                self.logger.error(f"Non-retriable error in {func.__name__}: {e}")
                raise

        raise last_exception or RuntimeError(f"Failed to execute {func.__name__} after {self.max_retries} attempts")

    async def retry_async(self, async_func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Retry an asynchronous function with configured strategy.

        Args:
            async_func (Callable): Async function to retry
            *args: Positional arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Any: Return value from function
        """
        import asyncio

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await async_func(*args, **kwargs)

            except self.RETRIABLE_ERRORS as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.max_retries} attempts failed for {async_func.__name__}: {e}"
                    )

            except Exception as e:
                # Non-retriable error - fail immediately
                self.logger.error(f"Non-retriable error in {async_func.__name__}: {e}")
                raise

        raise last_exception or RuntimeError(f"Failed to execute {async_func.__name__} after {self.max_retries} attempts")

    def retry_with_backoff(self, max_retries: int, base_delay: float = 1.0,
                          max_delay: float = 60.0, strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_WITH_JITTER):
        """
        Decorator for retrying functions with backoff.

        Args:
            max_retries (int): Maximum number of retry attempts
            base_delay (float): Base delay in seconds
            max_delay (float): Maximum delay in seconds
            strategy (RetryStrategy): Retry strategy to use

        Returns:
            Callable: Decorated function with retry logic
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                manager = RetryManager(max_retries, base_delay, max_delay, strategy, self.logger)
                return manager.retry_sync(func, *args, **kwargs)
            return wrapper
        return decorator

    def retry_with_exponential_backoff(self, initial_delay: float = 1.0, max_delay: float = 60.0):
        """
        Decorator for retrying with exponential backoff.

        Args:
            initial_delay (float): Initial delay in seconds
            max_delay (float): Maximum delay in seconds

        Returns:
            Callable: Decorated function with retry logic
        """
        return self.retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=initial_delay,
            max_delay=max_delay,
            strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER
        )

    def handle_error(self, error: Exception, context: Optional[str] = None, retriable: bool = False) -> bool:
        """
        Handle and classify an error.

        Args:
            error (Exception): Error to handle
            context (str, optional): Context description
            retriable (bool): Whether error is retriable

        Returns:
            bool: True if error is retriable, False otherwise
        """
        error_name = error.__class__.__name__
        is_retriable = isinstance(error, self.RETRIABLE_ERRORS) or retriable

        context_msg = f" in {context}" if context else ""
        status_msg = "retriable" if is_retriable else "non-retriable"

        if is_retriable:
            self.logger.warning(f"{status_msg} error{context_msg}: {error_name}: {error}")
        else:
            self.logger.error(f"{status_msg} error{context_msg}: {error_name}: {error}")

        return is_retriable

    def get_retriable_errors(self) -> tuple:
        """Get tuple of retriable error types."""
        return self.RETRIABLE_ERRORS

    def classify_http_status(self, status_code: int) -> bool:
        """
        Classify HTTP status code as retriable.

        Args:
            status_code (int): HTTP status code

        Returns:
            bool: True if retriable, False otherwise
        """
        # Retriable status codes
        retriable_statuses = {408, 429, 500, 502, 503, 504}
        return status_code in retriable_statuses

    def add_retriable_error(self, error_type: type) -> None:
        """
        Add additional retriable error type.

        Args:
            error_type (type): Error class to add to retriable list
        """
        if not isinstance(self.RETRIABLE_ERRORS, list):
            self.RETRIABLE_ERRORS = list(self.RETRIABLE_ERRORS)
        if error_type not in self.RETRIABLE_ERRORS:
            self.RETRIABLE_ERRORS.append(error_type)


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures.

    Tracks failure rate and can "break" to prevent further attempts
    during known failure periods.
    """

    def __init__(self, failure_threshold: float = 0.5, timeout: int = 60, logger: Optional[logging.Logger] = None):
        """
        Initialize CircuitBreaker.

        Args:
            failure_threshold (float): Failure rate threshold (0-1) to trigger break
            timeout (int): Timeout in seconds before attempting to close
            logger (logging.Logger, optional): Logger instance
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self) -> None:
        """Record a successful call."""
        self.successes += 1
        if self.state == "half-open":
            self.reset()

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        total = self.failures + self.successes
        failure_rate = self.failures / total if total > 0 else 0

        if failure_rate >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(f"Circuit breaker opened (failure rate: {failure_rate:.2%})")

    def can_execute(self) -> bool:
        """
        Check if call can execute.

        Returns:
            bool: True if call can proceed, False if circuit is open
        """
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker attempting half-open")
                return True
            return False

        # half-open
        return True

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failures = 0
        self.successes = 0
        self.last_failure_time = None
        self.state = "closed"
        self.logger.info("Circuit breaker reset to closed")
