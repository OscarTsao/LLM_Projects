"""Metrics buffering for MLflow outages.

This module provides disk-backed buffering of metrics during MLflow outages,
with automatic replay when the backend becomes available.

Implements FR-017: Metrics buffering during MLflow outages
Implements Principle IV: Graceful degradation
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Single metric entry for buffering."""

    key: str
    value: float
    step: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricEntry":
        """Create from dictionary."""
        return cls(**data)


class MetricsBuffer:
    """Disk-backed buffer for metrics during MLflow outages.

    Implements FR-017: Buffer metrics to disk during outages, replay on recovery.
    """

    def __init__(
        self, buffer_file: Path, max_buffer_size_mb: float = 100.0, replay_batch_size: int = 100
    ):
        """Initialize metrics buffer.

        Args:
            buffer_file: Path to JSONL buffer file
            max_buffer_size_mb: Warn when buffer exceeds this size (no hard limit)
            replay_batch_size: Number of metrics to replay in one batch
        """
        self.buffer_file = Path(buffer_file)
        self.max_buffer_size_mb = max_buffer_size_mb
        self.replay_batch_size = replay_batch_size

        # Create buffer directory
        self.buffer_file.parent.mkdir(parents=True, exist_ok=True)

        # Track buffer state
        self._buffer_count = 0
        self._last_size_check = 0

        logger.info(
            f"Initialized MetricsBuffer: "
            f"file={buffer_file}, "
            f"max_size={max_buffer_size_mb}MB"
        )

    def buffer_metric(self, key: str, value: float, step: int, timestamp: float | None = None):
        """Buffer a single metric to disk.

        Args:
            key: Metric name
            value: Metric value
            step: Training step
            timestamp: Timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        entry = MetricEntry(key=key, value=value, step=step, timestamp=timestamp)

        # Append to JSONL file
        with open(self.buffer_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        self._buffer_count += 1

        # Check buffer size periodically
        if self._buffer_count % 100 == 0:
            self._check_buffer_size()

    def buffer_metrics(self, metrics: dict[str, float], step: int):
        """Buffer multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
        """
        timestamp = time.time()

        for key, value in metrics.items():
            self.buffer_metric(key, value, step, timestamp)

    def get_buffered_count(self) -> int:
        """Get number of buffered metrics.

        Returns:
            Number of metrics in buffer
        """
        if not self.buffer_file.exists():
            return 0

        count = 0
        with open(self.buffer_file) as f:
            for _ in f:
                count += 1

        return count

    def replay_buffer(
        self, log_fn: callable, max_retries: int = 3, base_delay: float = 1.0
    ) -> bool:
        """Replay buffered metrics with exponential backoff.

        Implements FR-017: Automatic replay with exponential backoff.

        Args:
            log_fn: Function to log metrics (e.g., mlflow.log_metric)
            max_retries: Maximum retry attempts per batch
            base_delay: Base delay for exponential backoff

        Returns:
            True if all metrics replayed successfully, False otherwise
        """
        if not self.buffer_file.exists():
            return True

        # Read all buffered metrics
        buffered_metrics = []
        try:
            with open(self.buffer_file) as f:
                for line in f:
                    if line.strip():
                        entry_dict = json.loads(line)
                        buffered_metrics.append(MetricEntry.from_dict(entry_dict))
        except Exception as e:
            logger.error(f"Failed to read buffer file: {e}")
            return False

        if not buffered_metrics:
            # Empty buffer, remove file
            self.buffer_file.unlink()
            return True

        logger.info(f"Replaying {len(buffered_metrics)} buffered metrics...")

        # Replay in batches
        success = True
        for i in range(0, len(buffered_metrics), self.replay_batch_size):
            batch = buffered_metrics[i : i + self.replay_batch_size]

            # Try to log batch with exponential backoff
            for attempt in range(max_retries):
                try:
                    for entry in batch:
                        log_fn(
                            key=entry.key,
                            value=entry.value,
                            step=entry.step,
                            timestamp=int(entry.timestamp * 1000),  # MLflow uses milliseconds
                        )

                    logger.debug(f"Replayed batch {i // self.replay_batch_size + 1}")
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            f"Failed to replay batch (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to replay batch after {max_retries} attempts: {e}")
                        success = False

        # If successful, clear buffer
        if success:
            self.buffer_file.unlink()
            self._buffer_count = 0
            logger.info("Successfully replayed all buffered metrics")

        return success

    def clear_buffer(self):
        """Clear the buffer file."""
        if self.buffer_file.exists():
            self.buffer_file.unlink()
            self._buffer_count = 0
            logger.info("Buffer cleared")

    def _check_buffer_size(self):
        """Check buffer size and warn if exceeds threshold."""
        if not self.buffer_file.exists():
            return

        size_mb = self.buffer_file.stat().st_size / (1024**2)

        if size_mb > self.max_buffer_size_mb:
            logger.warning(
                f"Metrics buffer size ({size_mb:.2f} MB) exceeds threshold "
                f"({self.max_buffer_size_mb} MB). Consider checking MLflow connectivity."
            )


class MLflowMetricsLogger:
    """MLflow metrics logger with automatic buffering on outages.

    Implements FR-017: Graceful degradation during MLflow outages.
    """

    def __init__(
        self, buffer_file: Path, mlflow_client: Any | None = None, enable_buffering: bool = True
    ):
        """Initialize MLflow metrics logger.

        Args:
            buffer_file: Path to buffer file
            mlflow_client: MLflow client (optional)
            enable_buffering: Whether to enable buffering
        """
        self.mlflow_client = mlflow_client
        self.enable_buffering = enable_buffering

        if enable_buffering:
            self.buffer = MetricsBuffer(buffer_file)
        else:
            self.buffer = None

        logger.info(
            f"Initialized MLflowMetricsLogger (buffering={'enabled' if enable_buffering else 'disabled'})"
        )

    def log_metric(self, key: str, value: float, step: int):
        """Log a single metric with automatic buffering on failure.

        Args:
            key: Metric name
            value: Metric value
            step: Training step
        """
        try:
            if self.mlflow_client:
                self.mlflow_client.log_metric(key, value, step)
            else:
                # Simulate logging (for testing)
                logger.debug(f"Logged metric: {key}={value} (step={step})")
        except Exception as e:
            logger.warning(f"Failed to log metric to MLflow: {e}")

            if self.enable_buffering and self.buffer:
                logger.info(f"Buffering metric: {key}={value}")
                self.buffer.buffer_metric(key, value, step)
            else:
                raise

    def log_metrics(self, metrics: dict[str, float], step: int):
        """Log multiple metrics with automatic buffering on failure.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
        """
        try:
            if self.mlflow_client:
                for key, value in metrics.items():
                    self.mlflow_client.log_metric(key, value, step)
            else:
                logger.debug(f"Logged metrics: {metrics} (step={step})")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

            if self.enable_buffering and self.buffer:
                logger.info(f"Buffering {len(metrics)} metrics")
                self.buffer.buffer_metrics(metrics, step)
            else:
                raise

    def replay_buffered_metrics(self) -> bool:
        """Replay buffered metrics.

        Returns:
            True if replay successful, False otherwise
        """
        if not self.enable_buffering or not self.buffer:
            return True

        if not self.mlflow_client:
            logger.warning("Cannot replay metrics: no MLflow client configured")
            return False

        return self.buffer.replay_buffer(self.mlflow_client.log_metric)
