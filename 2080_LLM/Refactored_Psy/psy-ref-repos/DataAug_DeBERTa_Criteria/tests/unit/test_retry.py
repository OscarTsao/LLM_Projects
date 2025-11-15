"""Unit tests for exponential backoff retry utility."""

from unittest.mock import Mock, patch

import pytest

from src.dataaug_multi_both.utils.retry import RetryableOperation, exponential_backoff_retry


class TestExponentialBackoffRetry:
    """Test suite for exponential_backoff_retry decorator."""

    def test_success_on_first_attempt(self):
        """Test that function succeeds on first attempt without retry."""
        mock_func = Mock(return_value="success")
        decorated = exponential_backoff_retry()(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self):
        """Test that function succeeds after some retries."""
        mock_func = Mock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])
        decorated = exponential_backoff_retry(max_attempts=5)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_failure_after_max_attempts(self):
        """Test that function raises exception after max attempts."""
        mock_func = Mock(side_effect=ValueError("persistent failure"))
        decorated = exponential_backoff_retry(max_attempts=3)(mock_func)

        with pytest.raises(ValueError, match="persistent failure"):
            decorated()

        assert mock_func.call_count == 3

    @patch('time.sleep')
    def test_exponential_backoff_delays(self, mock_sleep):
        """Test that delays follow exponential backoff pattern."""
        mock_func = Mock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            ValueError("fail 3"),
            "success"
        ])
        decorated = exponential_backoff_retry(
            max_attempts=5,
            base_delay=1.0,
            max_delay=16.0
        )(mock_func)

        result = decorated()

        assert result == "success"
        # Verify delays: 1s, 2s, 4s
        assert mock_sleep.call_count == 3
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0, 4.0]

    @patch('time.sleep')
    def test_max_delay_cap(self, mock_sleep):
        """Test that delay is capped at max_delay."""
        mock_func = Mock(side_effect=[
            ValueError("fail") for _ in range(10)
        ] + ["success"])
        decorated = exponential_backoff_retry(
            max_attempts=11,
            base_delay=1.0,
            max_delay=8.0
        )(mock_func)

        result = decorated()

        assert result == "success"
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        # Delays: 1, 2, 4, 8, 8, 8, 8, 8, 8, 8 (capped at 8)
        assert all(delay <= 8.0 for delay in delays)
        assert delays[3:] == [8.0] * 7  # All subsequent delays capped

    def test_specific_exceptions_only(self):
        """Test that only specified exceptions are retried."""
        mock_func = Mock(side_effect=RuntimeError("not retryable"))
        decorated = exponential_backoff_retry(
            max_attempts=3,
            exceptions=(ValueError,)
        )(mock_func)

        with pytest.raises(RuntimeError, match="not retryable"):
            decorated()

        assert mock_func.call_count == 1  # No retries

    def test_on_retry_callback(self):
        """Test that on_retry callback is called on each retry."""
        callback = Mock()
        mock_func = Mock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])
        decorated = exponential_backoff_retry(
            max_attempts=5,
            on_retry=callback
        )(mock_func)

        result = decorated()

        assert result == "success"
        assert callback.call_count == 2
        # Verify callback arguments
        assert isinstance(callback.call_args_list[0][0][0], ValueError)
        assert callback.call_args_list[0][0][1] == 1  # First retry
        assert callback.call_args_list[1][0][1] == 2  # Second retry


class TestRetryableOperation:
    """Test suite for RetryableOperation context manager."""

    def test_success_on_first_attempt(self):
        """Test that operation succeeds on first attempt."""
        operation = Mock(return_value="success")

        with RetryableOperation(max_attempts=5) as retry:
            result = retry.execute(operation)

        assert result == "success"
        assert operation.call_count == 1

    def test_success_after_retries(self):
        """Test that operation succeeds after retries."""
        operation = Mock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])

        with RetryableOperation(max_attempts=5) as retry:
            result = retry.execute(operation)

        assert result == "success"
        assert operation.call_count == 3

    def test_failure_after_max_attempts(self):
        """Test that operation raises exception after max attempts."""
        operation = Mock(side_effect=ValueError("persistent failure"))

        with pytest.raises(ValueError, match="persistent failure"):
            with RetryableOperation(max_attempts=3) as retry:
                retry.execute(operation)

        assert operation.call_count == 3

    @patch('time.sleep')
    def test_exponential_backoff_delays(self, mock_sleep):
        """Test that delays follow exponential backoff pattern."""
        operation = Mock(side_effect=[
            ValueError("fail 1"),
            ValueError("fail 2"),
            "success"
        ])

        with RetryableOperation(max_attempts=5, base_delay=1.0) as retry:
            result = retry.execute(operation)

        assert result == "success"
        assert mock_sleep.call_count == 2
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == [1.0, 2.0]

