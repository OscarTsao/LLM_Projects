"""Exponential backoff retry utility for handling transient failures.

This module provides retry logic with exponential backoff for operations that may
fail transiently (e.g., network requests, file I/O).

Implements FR-005: Model loading retry with exponential backoff (1s, 2s, 4s, 8s, 16s).
"""

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def exponential_backoff_retry(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """Decorator for retrying a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: 5)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 16.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry
                 Signature: on_retry(exception, attempt_number)

    Returns:
        Decorated function that retries on failure

    Example:
        @exponential_backoff_retry(max_attempts=5, base_delay=1.0)
        def load_model(model_id: str):
            return AutoModel.from_pretrained(model_id)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            func_name = getattr(func, "__name__", repr(func))

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"Function {func_name} failed after {max_attempts} attempts. "
                            f"Last error: {str(e)}"
                        )
                        raise

                    # Calculate delay: min(base_delay * 2^(attempt-1), max_delay)
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

                    logger.warning(
                        f"Function {func_name} failed on attempt {attempt}/{max_attempts}. "
                        f"Retrying in {delay}s. Error: {str(e)}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # This should never be reached, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Unexpected error in retry logic for {func_name}")

        return wrapper

    return decorator


class RetryableOperation:
    """Context manager for retryable operations with exponential backoff.

    Example:
        with RetryableOperation(max_attempts=5) as retry:
            result = retry.execute(lambda: load_model("bert-base"))
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, operation: Callable[[], T]) -> T:
        """Execute an operation with retry logic.

        Args:
            operation: Callable that performs the operation

        Returns:
            Result of the operation

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except self.exceptions as e:
                last_exception = e

                if attempt == self.max_attempts:
                    logger.error(
                        f"Operation failed after {self.max_attempts} attempts. "
                        f"Last error: {str(e)}"
                    )
                    raise

                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)

                logger.warning(
                    f"Operation failed on attempt {attempt}/{self.max_attempts}. "
                    f"Retrying in {delay}s. Error: {str(e)}"
                )

                time.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected error in retry logic")
