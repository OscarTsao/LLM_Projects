import io
import json
import logging
from pathlib import Path

import pytest

from src.dataaug_multi_both.utils.logging import (
    SizeAndTimeRotatingFileHandler,
    build_logger,
    format_storage_exhaustion_error,
    sanitize_log_payload,
)


@pytest.mark.unit
def test_sanitize_log_payload_masks_sensitive_data() -> None:
    payload = {
        "token": "Bearer secret-123",
        "openai": "sk-verylongsecretvalue",
        "hf": "hf_abcdefgh12345678",
        "email": "user@example.com",
        "nested": {"password": "hunter2"},
    }

    sanitized = sanitize_log_payload(payload)

    assert sanitized["token"] == "Bearer ***"
    assert sanitized["openai"] == "sk-***"
    assert sanitized["hf"] == "hf_***"
    assert sanitized["email"] == "[redacted-email]"
    assert sanitized["nested"]["password"] == "***"


@pytest.mark.unit
def test_json_logger_masks_values(workspace_tmp_path: Path) -> None:
    logger_name = "test_logger"
    logger = build_logger(logger_name, log_dir=workspace_tmp_path)

    # Redirect stream handler to avoid polluting stderr during tests.
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(io.StringIO())

    file_handler = next(
        handler for handler in logger.handlers if isinstance(handler, SizeAndTimeRotatingFileHandler)
    )

    logger.info(
        "User email %s token=%s",
        "tester@example.com",
        "sk-secret-1234567890",
        extra={"api_key": "hf_secret_token", "component": "trainer"},
    )
    for handler in logger.handlers:
        handler.flush()

    log_path = Path(file_handler.baseFilename)
    with open(log_path, "r", encoding="utf-8") as f:
        line = f.readline()

    payload = json.loads(line)
    assert payload["message"] == "User email [redacted-email] token=sk-***"
    assert payload["component"] == "trainer"
    assert payload["extra"]["api_key"] == "hf_***"
    assert "tester@example.com" not in line
    assert file_handler.backupCount == 14

    # Cleanup handlers for other tests.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.__dict__.pop("_is_dataaug_configured", None)


@pytest.mark.unit
def test_format_storage_exhaustion_error_lists_artifacts(workspace_tmp_path: Path) -> None:
    message = format_storage_exhaustion_error(
        current_usage_gb=512.0,
        available_gb=10.5,
        next_checkpoint_size_gb=12.0,
        trial_dir=workspace_tmp_path,
        artifact_sizes=[
            (workspace_tmp_path / "checkpoint_epoch10.pt", 8.2),
            (workspace_tmp_path / "checkpoint_epoch09.pt", 7.8),
        ],
        retention_policy_hint="2",
    )

    assert "512.00 GiB" in message
    assert "10.50 GiB" in message
    assert "12.00 GiB" in message
    assert "checkpoint_epoch10.pt" in message
    assert "retention policy" in message
    assert "hydra override 'trainer.checkpoint.retention=2'" in message
