# File: src/utils/logging.py
"""Logging utilities with progress tracking and timing."""

import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path
import sys


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        level: Logging level
        log_file: Optional file to write logs to
        format_str: Custom format string

    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = ColoredFormatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class Timer:
    """Simple timer for measuring elapsed time."""

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.elapsed_time = time.time() - self.start_time
        self.start_time = None
        return self.elapsed_time

    def elapsed(self) -> float:
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return self.elapsed_time
        return time.time() - self.start_time


class ProgressTracker:
    """Progress tracker with ETA estimation."""

    def __init__(self, total: int, description: str = "Progress") -> None:
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """Update progress by increment."""
        self.current = min(self.current + increment, self.total)

    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        return {
            "current": self.current,
            "total": self.total,
            "percentage": (self.current / self.total) * 100,
            "elapsed": elapsed,
            "rate": rate,
            "eta": eta,
        }

    def log_progress(self, logger: logging.Logger) -> None:
        """Log current progress."""
        stats = self.get_stats()
        logger.info(
            f"{self.description}: {stats['current']}/{stats['total']} "
            f"({stats['percentage']:.1f}%) - "
            f"Rate: {stats['rate']:.2f}/s - "
            f"ETA: {stats['eta']:.0f}s"
        )