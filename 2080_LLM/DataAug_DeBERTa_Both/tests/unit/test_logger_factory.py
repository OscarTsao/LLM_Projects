"""Unit tests for logger factory."""

import pytest
import tempfile
import logging
from pathlib import Path
from src.dataaug_multi_both.utils.logger_factory import (
    create_logger,
    create_run_logger,
    create_component_logger,
    get_default_log_dir
)


class TestLoggerFactory:
    """Test suite for logger factory."""
    
    def test_create_logger_basic(self):
        """Test basic logger creation."""
        logger = create_logger("test_logger", enable_json=False)
        assert logger is not None
        assert logger.name == "test_logger"
        assert isinstance(logger, logging.Logger)
    
    def test_create_logger_with_log_dir(self):
        """Test logger creation with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_logger(
                "test_logger_dir",
                log_dir=Path(tmpdir),
                enable_json=True
            )
            assert logger is not None
            assert len(logger.handlers) >= 1
    
    def test_create_logger_respects_level(self):
        """Test that logger respects log level."""
        logger = create_logger("test_logger_level", level="DEBUG", enable_json=False)
        assert logger.level == logging.DEBUG
        
        logger = create_logger("test_logger_level2", level="ERROR", enable_json=False)
        assert logger.level == logging.ERROR
    
    def test_create_run_logger(self):
        """Test run logger creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = create_run_logger("trial_001", log_dir=Path(tmpdir))
            assert logger is not None
            assert "trial_001" in logger.name
    
    def test_create_component_logger(self):
        """Test component logger creation."""
        logger = create_component_logger("trainer")
        assert logger is not None
        assert "trainer" in logger.name
        assert "dataaug_multi_both" in logger.name
    
    def test_get_default_log_dir(self):
        """Test default log directory creation."""
        log_dir = get_default_log_dir()
        assert log_dir.exists()
        assert log_dir.name == "logs"
        assert log_dir.parent.name == "experiments"
    
    def test_multiple_loggers_independent(self):
        """Test that multiple loggers are independent."""
        logger1 = create_logger("logger1", enable_json=False, level="DEBUG")
        logger2 = create_logger("logger2", enable_json=False, level="ERROR")
        
        assert logger1.name != logger2.name
        assert logger1.level != logger2.level

