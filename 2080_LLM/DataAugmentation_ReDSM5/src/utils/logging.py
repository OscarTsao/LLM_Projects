"""
Logging utilities for experiment tracking and debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "redsm5",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_combo_info(logger: logging.Logger, combo: list, combo_hash: str) -> None:
    """
    Log information about an augmentation combo.
    
    Args:
        logger: Logger instance
        combo: List of augmenter names
        combo_hash: Combo hash
    """
    logger.info(f"Combo: {' -> '.join(combo)}")
    logger.info(f"Hash: {combo_hash}")
    logger.info(f"Length: {len(combo)}")


def log_hpo_progress(
    logger: logging.Logger,
    trial_number: int,
    params: dict,
    value: float,
) -> None:
    """
    Log HPO trial progress.
    
    Args:
        logger: Logger instance
        trial_number: Trial number
        params: Trial parameters
        value: Trial value (metric)
    """
    logger.info(f"Trial {trial_number}: value={value:.4f}")
    logger.info(f"Parameters: {params}")


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    """
    Log evaluation metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{prefix}{name}: {value:.4f}")
        else:
            logger.info(f"{prefix}{name}: {value}")
