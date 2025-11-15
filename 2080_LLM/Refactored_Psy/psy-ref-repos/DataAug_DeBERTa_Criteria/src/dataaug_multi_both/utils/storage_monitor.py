"""Background storage monitoring thread for proactive disk space management.

This module provides a background thread that monitors disk space and sets
a critical flag when available space drops below a threshold.

Implements FR-018: Proactive pruning when disk < 10%
Implements Principle II: Storage-Optimized Artifact Management
"""

import logging
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageMonitor:
    """Background thread for monitoring disk space.

    Monitors the filesystem containing the experiments directory and sets
    a critical flag when available space drops below 10%.

    Implements FR-018: Proactive storage monitoring.
    """

    def __init__(
        self,
        monitor_path: Path,
        check_interval: int = 60,
        critical_threshold_percent: float = 10.0,
        warning_threshold_percent: float = 20.0,
    ):
        """Initialize storage monitor.

        Args:
            monitor_path: Path to monitor (e.g., experiments/ directory)
            check_interval: Seconds between checks (default: 60)
            critical_threshold_percent: Critical threshold percentage (default: 10%)
            warning_threshold_percent: Warning threshold percentage (default: 20%)
        """
        self.monitor_path = Path(monitor_path)
        self.check_interval = check_interval
        self.critical_threshold = critical_threshold_percent
        self.warning_threshold = warning_threshold_percent

        # Thread-safe communication
        self.storage_critical = threading.Event()
        self.storage_warning = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Logging throttling
        self._last_info_log = datetime.min
        self._last_warning_log = datetime.min
        self._info_log_interval = timedelta(minutes=10)

    def start(self):
        """Start the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Storage monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started storage monitor for {self.monitor_path}")

    def stop(self):
        """Stop the monitoring thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info("Stopped storage monitor")

    def is_critical(self) -> bool:
        """Check if storage is in critical state.

        Returns:
            True if available space < critical threshold
        """
        return self.storage_critical.is_set()

    def is_warning(self) -> bool:
        """Check if storage is in warning state.

        Returns:
            True if available space < warning threshold
        """
        return self.storage_warning.is_set()

    def get_disk_usage(self) -> dict:
        """Get current disk usage statistics.

        Returns:
            Dictionary with total, used, free bytes and percent used
        """
        try:
            usage = shutil.disk_usage(self.monitor_path)
            percent_used = (usage.used / usage.total) * 100
            percent_free = 100 - percent_used

            return {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "percent_used": percent_used,
                "percent_free": percent_free,
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
            }
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {}

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        while not self._stop_event.is_set():
            try:
                usage = self.get_disk_usage()

                if not usage:
                    time.sleep(self.check_interval)
                    continue

                percent_free = usage["percent_free"]

                # Update flags
                if percent_free < self.critical_threshold:
                    if not self.storage_critical.is_set():
                        self.storage_critical.set()
                        logger.error(
                            f"CRITICAL: Disk space below {self.critical_threshold}%! "
                            f"Free: {usage['free_gb']:.2f} GB ({percent_free:.1f}%)"
                        )
                else:
                    if self.storage_critical.is_set():
                        self.storage_critical.clear()
                        logger.info(
                            f"Disk space recovered above {self.critical_threshold}%. "
                            f"Free: {usage['free_gb']:.2f} GB ({percent_free:.1f}%)"
                        )

                if percent_free < self.warning_threshold:
                    if not self.storage_warning.is_set():
                        self.storage_warning.set()

                    # Log warning (throttled)
                    now = datetime.now()
                    if now - self._last_warning_log > timedelta(minutes=5):
                        logger.warning(
                            f"WARNING: Disk space below {self.warning_threshold}%. "
                            f"Free: {usage['free_gb']:.2f} GB ({percent_free:.1f}%)"
                        )
                        self._last_warning_log = now
                else:
                    if self.storage_warning.is_set():
                        self.storage_warning.clear()

                # Periodic info logging (every 10 minutes)
                now = datetime.now()
                if now - self._last_info_log > self._info_log_interval:
                    logger.info(
                        f"Disk usage: {usage['used_gb']:.2f} GB / {usage['total_gb']:.2f} GB "
                        f"({percent_free:.1f}% free)"
                    )
                    self._last_info_log = now

            except Exception as e:
                logger.error(f"Error in storage monitor loop: {e}")

            # Sleep until next check
            time.sleep(self.check_interval)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Global storage monitor instance (singleton pattern)
_global_monitor: StorageMonitor | None = None
_monitor_lock = threading.Lock()


def get_storage_monitor(monitor_path: Path = Path("experiments"), **kwargs) -> StorageMonitor:
    """Get or create the global storage monitor instance.

    Args:
        monitor_path: Path to monitor
        **kwargs: Additional arguments for StorageMonitor

    Returns:
        Global StorageMonitor instance
    """
    global _global_monitor

    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = StorageMonitor(monitor_path, **kwargs)
        return _global_monitor


def is_storage_critical() -> bool:
    """Check if storage is in critical state (convenience function).

    Returns:
        True if storage is critical, False otherwise
    """
    global _global_monitor
    if _global_monitor is None:
        return False
    return _global_monitor.is_critical()
