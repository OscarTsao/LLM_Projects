"""Unit tests for dual logging system."""

import pytest
import logging
import tempfile
from pathlib import Path
from src.dataaug_multi_both.utils.logging import (
    sanitize_text,
    build_logger,
    format_storage_exhaustion_error
)


class TestSanitization:
    """Test suite for log sanitization."""
    
    def test_sanitize_hf_token(self):
        """Test that Hugging Face tokens are sanitized."""
        text = "Loading model with token hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890"
        sanitized = sanitize_text(text)
        assert "hf_***" in sanitized
        assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890" not in sanitized
    
    def test_sanitize_bearer_token(self):
        """Test that Bearer tokens are sanitized."""
        text = "Authorization: Bearer abc123def456ghi789"
        sanitized = sanitize_text(text)
        assert "Bearer ***" in sanitized
        assert "abc123def456ghi789" not in sanitized
    
    def test_sanitize_email(self):
        """Test that email addresses are sanitized."""
        text = "Contact: user@example.com for support"
        sanitized = sanitize_text(text)
        assert "[redacted-email]" in sanitized
        assert "user@example.com" not in sanitized
    
    def test_sanitize_api_key(self):
        """Test that API keys are sanitized."""
        text = "api_key=sk_test_1234567890abcdef"
        sanitized = sanitize_text(text)
        # The pattern sanitizes the value after api_key=
        assert "api_key=***" in sanitized
        assert "sk_test_1234567890abcdef" not in sanitized
    
    def test_sanitize_password(self):
        """Test that passwords are sanitized."""
        text = "password=mysecretpassword123"
        sanitized = sanitize_text(text)
        assert "password=***" in sanitized
        assert "mysecretpassword123" not in sanitized


class TestDualLogger:
    """Test suite for dual logging system."""
    
    def test_logger_creation(self):
        """Test that logger can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = build_logger("test_logger", log_dir=tmpdir)
            assert logger is not None
            assert logger.name == "test_logger"
    
    def test_logger_has_handlers(self):
        """Test that logger has both stdout and file handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = build_logger("test_logger", log_dir=tmpdir)
            assert len(logger.handlers) == 2  # stdout + file
    
    def test_logger_writes_to_file(self):
        """Test that logger writes to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = build_logger("test_logger_file", log_dir=tmpdir)
            logger.info("Test message")

            # Flush and close handlers to ensure file is written
            for handler in logger.handlers:
                handler.flush()
                if hasattr(handler, 'close'):
                    handler.close()

            # Check if any .jsonl file was created
            log_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(log_files) > 0, f"No log files found in {tmpdir}"

            content = log_files[0].read_text()
            assert "Test message" in content
            assert "INFO" in content
    
    def test_logger_sanitizes_sensitive_data(self):
        """Test that logger sanitizes sensitive data in logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = build_logger("test_logger_sanitize", log_dir=tmpdir)
            logger.info("Token: hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890")

            # Flush and close handlers to ensure file is written
            for handler in logger.handlers:
                handler.flush()
                if hasattr(handler, 'close'):
                    handler.close()

            # Check if any .jsonl file was created
            log_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(log_files) > 0, f"No log files found in {tmpdir}"

            content = log_files[0].read_text()

            # The sanitization replaces the entire token with ***
            assert "***" in content
            assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890" not in content


class TestStorageExhaustionFormatter:
    """Test suite for storage exhaustion error formatter."""
    
    def test_format_basic_error(self):
        """Test basic error formatting."""
        error_msg = format_storage_exhaustion_error(
            current_usage_gb=50.0,
            available_gb=5.0,
            next_checkpoint_size_gb=10.0,
            trial_dir=Path("/experiments/trial_001")
        )
        
        assert "Storage exhaustion detected" in error_msg
        assert "50.00 GiB" in error_msg
        assert "5.00 GiB" in error_msg
        assert "10.00 GiB" in error_msg
        assert "/experiments/trial_001" in error_msg
    
    def test_format_with_artifacts(self):
        """Test error formatting with artifact enumeration."""
        artifacts = [
            (Path("checkpoint_1.pt"), 2.5),
            (Path("checkpoint_2.pt"), 2.5),
            (Path("checkpoint_3.pt"), 2.0),
        ]
        
        error_msg = format_storage_exhaustion_error(
            current_usage_gb=50.0,
            available_gb=5.0,
            next_checkpoint_size_gb=10.0,
            trial_dir=Path("/experiments/trial_001"),
            artifact_sizes=artifacts
        )
        
        assert "Largest artifacts by size" in error_msg
        assert "checkpoint_1.pt" in error_msg
        assert "2.50 GiB" in error_msg
    
    def test_format_with_retention_hint(self):
        """Test error formatting with retention policy hint."""
        error_msg = format_storage_exhaustion_error(
            current_usage_gb=50.0,
            available_gb=5.0,
            next_checkpoint_size_gb=10.0,
            trial_dir=Path("/experiments/trial_001"),
            retention_policy_hint="keep_last_n=1,keep_best_k=1"
        )
        
        assert "Adjust retention policy" in error_msg
        assert "keep_last_n=1,keep_best_k=1" in error_msg

