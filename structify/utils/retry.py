"""Retry logic with exponential backoff."""

import time
import functools
from dataclasses import dataclass
from typing import Callable, TypeVar, Any
from collections.abc import Sequence

from structify.core.exceptions import RateLimitError, TimeoutError, ProviderError

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 5
    base_delay: int = 60
    max_delay: int = 300
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (RateLimitError, TimeoutError, ConnectionError)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for the given attempt using exponential backoff.

    Args:
        attempt: Current attempt number (1-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    import random

    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add up to 10% jitter
        jitter_amount = delay * 0.1 * random.random()
        delay += jitter_amount

    return delay


def retry_with_backoff(
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration
        on_retry: Callback called on each retry with (attempt, exception, delay)

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)

                        if on_retry:
                            on_retry(attempt, e, delay)

                        time.sleep(delay)
                    else:
                        raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_call(
    func: Callable[..., T],
    args: Sequence = (),
    kwargs: dict[str, Any] | None = None,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """
    Call a function with retry logic.

    Args:
        func: Function to call
        args: Positional arguments
        kwargs: Keyword arguments
        config: Retry configuration
        on_retry: Callback called on each retry

    Returns:
        Function result
    """
    if kwargs is None:
        kwargs = {}
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(1, config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(attempt, config)

                if on_retry:
                    on_retry(attempt, e, delay)

                time.sleep(delay)
            else:
                raise

    if last_exception:
        raise last_exception


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is retryable
    """
    error_str = str(error).lower()

    # Rate limit errors
    if "429" in str(error) or "quota" in error_str or "resource" in error_str:
        return True

    # Timeout errors
    if "504" in str(error) or "deadline" in error_str or "timeout" in error_str:
        return True

    # Connection errors
    if "connection" in error_str or "network" in error_str:
        return True

    return False


def classify_error(error: Exception) -> type[ProviderError]:
    """
    Classify an error into a specific exception type.

    Args:
        error: The exception to classify

    Returns:
        The appropriate exception class
    """
    error_str = str(error).lower()

    if "429" in str(error) or "quota" in error_str or "resource" in error_str:
        return RateLimitError

    if "504" in str(error) or "deadline" in error_str or "timeout" in error_str:
        return TimeoutError

    return ProviderError
