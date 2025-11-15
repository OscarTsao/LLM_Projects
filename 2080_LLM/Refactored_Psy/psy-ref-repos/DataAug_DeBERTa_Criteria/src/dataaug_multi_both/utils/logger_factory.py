"""Centralized logger factory for the application.

This module provides a factory for creating loggers with consistent configuration
across the application, integrating the dual logging system.

Implements FR-020: Dual logging (JSON + stdout)
Implements FR-032: Log sanitization
"""

import logging
from pathlib import Path


def create_logger(
    name: str,
    log_dir: Path | None = None,
    level: str = "INFO",
    enable_json: bool = True,
    enable_sanitization: bool = True,
) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name (typically module name)
        log_dir: Directory for JSON log files (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to enable JSON file logging
        enable_sanitization: Whether to enable sensitive data sanitization

    Returns:
        Configured logger instance

    Example:
        logger = create_logger(__name__, log_dir=Path("experiments/logs"))
        logger.info("Training started")
    """
    # Import here to avoid circular dependencies
    from src.dataaug_multi_both.utils.logging import build_logger

    # Use build_logger from existing logging module
    if log_dir and enable_json:
        logger = build_logger(name, log_dir=str(log_dir))
    else:
        # Create basic logger with stdout only
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger


def get_default_log_dir() -> Path:
    """Get the default log directory.

    Returns:
        Path to default log directory (experiments/logs)
    """
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def create_run_logger(
    run_id: str, log_dir: Path | None = None, level: str = "INFO"
) -> logging.Logger:
    """Create a logger for a specific training run.

    Args:
        run_id: Unique identifier for the run
        log_dir: Directory for log files (defaults to experiments/logs)
        level: Logging level

    Returns:
        Configured logger for the run

    Example:
        logger = create_run_logger("trial_001")
        logger.info("Trial started")
    """
    if log_dir is None:
        log_dir = get_default_log_dir()

    logger_name = f"run.{run_id}"
    return create_logger(logger_name, log_dir=log_dir, level=level)


def create_component_logger(component: str, level: str = "INFO") -> logging.Logger:
    """Create a logger for a specific component.

    Args:
        component: Component name (e.g., "trainer", "checkpoint_manager")
        level: Logging level

    Returns:
        Configured logger for the component

    Example:
        logger = create_component_logger("trainer")
        logger.info("Trainer initialized")
    """
    logger_name = f"dataaug_multi_both.{component}"
    return create_logger(logger_name, enable_json=False, level=level)


# Module-level logger for this module
logger = create_component_logger("logger_factory")
