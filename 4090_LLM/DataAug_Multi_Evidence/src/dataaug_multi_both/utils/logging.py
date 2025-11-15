from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

SANITIZATION_PATTERNS: tuple[tuple[re.Pattern[str], str | Callable[[Any], str]], ...] = (
    # Specific API key patterns first (they take priority)
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer ***"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}\b"), "sk-***"),
    (re.compile(r"\bhf_[A-Za-z0-9_\-]{8,}\b"), "hf_***"),
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[redacted-email]",
    ),
    # Generic pattern last (only applies if specific patterns haven't already matched)
    # Skip if value looks like an API key we've already sanitized (sk-***, hf_***, Bearer ***)
    (
        re.compile(
            r"(?i)\b(api[-_]?key|token|secret|password)\b\s*([:=])\s*(['\"]?)(?!sk-\*\*\*|hf_\*\*\*|Bearer\s+\*\*\*)([^'\"\s]+)(\3)"
        ),
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}***{m.group(3)}",
    ),
)

LOGGING_RESERVED_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "stacklevel",
}


@dataclass
class LogEvent:
    timestamp: str
    level: str
    message: str
    component: str | None = None
    trial_id: str | None = None
    extra: dict[str, Any] | None = None


def sanitize_text(value: str) -> str:
    sanitized = value
    for pattern, replacement in SANITIZATION_PATTERNS:
        if callable(replacement):
            sanitized = pattern.sub(replacement, sanitized)
        else:
            sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_log_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, str):
        return sanitize_text(payload)
    if isinstance(payload, Mapping):
        sanitized_dict: dict[str, Any] = {}
        for k, v in payload.items():
            key_str = str(k)
            sanitized_value = sanitize_log_payload(v)
            # Only apply key-based sanitization if value is a string and wasn't already sanitized
            if re.search(r"(?i)(api[-_]?key|token|secret|password)", key_str):
                if isinstance(v, str) and v == sanitized_value:
                    # Value wasn't sanitized by patterns, apply key-based sanitization
                    sanitized_value = "***"
            sanitized_dict[key_str] = sanitized_value
        return sanitized_dict
    if isinstance(payload, list | tuple | set | frozenset):
        sanitized_items = [sanitize_log_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(sanitized_items)
        if isinstance(payload, set):
            return set(sanitized_items)
        if isinstance(payload, frozenset):
            return frozenset(sanitized_items)
        return sanitized_items
    if isinstance(payload, int | float | bool):
        return payload
    return str(payload)


class SanitizingFormatter(logging.Formatter):
    """Base formatter that masks sensitive tokens before rendering."""

    def get_sanitized_message(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            message = str(record.msg)
        return sanitize_text(message)

    def get_sanitized_extra(self, record: logging.LogRecord) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in LOGGING_RESERVED_ATTRS:
                continue
            extra[key] = sanitize_log_payload(value)
        return extra

    def formatException(self, ei):  # noqa: N802
        formatted = super().formatException(ei)
        return sanitize_text(formatted)


class JsonLogFormatter(SanitizingFormatter):
    """Formatter that emits structured JSON lines and masks sensitive fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = LogEvent(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            message=self.get_sanitized_message(record),
            component=self._safe_get(record, "component"),
            trial_id=self._safe_get(record, "trial_id"),
            extra=self._build_extra(record),
        )
        payload = {k: v for k, v in asdict(event).items() if v is not None}
        return json.dumps(payload, default=str)

    def _safe_get(self, record: logging.LogRecord, attr: str) -> str | None:
        value = getattr(record, attr, None)
        if value is None:
            return None
        return str(sanitize_log_payload(value))

    def _build_extra(self, record: logging.LogRecord) -> dict[str, Any] | None:
        extra = self.get_sanitized_extra(record)
        if extra:
            # component/trial_id already captured separately
            extra.pop("component", None)
            extra.pop("trial_id", None)
        return extra or None


class HumanReadableFormatter(SanitizingFormatter):
    """Formatter for stdout logging with concise sanitized output."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        sanitized_message = self.get_sanitized_message(record)
        original_msg, original_args = record.msg, record.args
        try:
            record.msg = sanitized_message
            record.args = ()
            formatted = super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args

        extra = self.get_sanitized_extra(record)
        extra.pop("component", None)
        extra.pop("trial_id", None)
        if extra:
            kv_pairs = ", ".join(f"{key}={value}" for key, value in extra.items())
            formatted = f"{formatted} | {kv_pairs}"

        if record.exc_info:
            formatted = f"{formatted}\n{self.formatException(record.exc_info)}"
        return formatted


class SizeAndTimeRotatingFileHandler(TimedRotatingFileHandler):
    """Rotate log files daily or when exceeding ``max_bytes``."""

    def __init__(
        self,
        filename: str,
        max_bytes: int = 1_000_000_000,
        backup_count: int = 14,
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(filename, when="midnight", backupCount=backup_count, encoding=encoding)
        self.max_bytes = max_bytes

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802
        if super().shouldRollover(record):
            return True

        if self.max_bytes <= 0:
            return False

        if self.stream is None:
            self.stream = self._open()

        current_size = self.stream.tell()
        message = self.format(record)
        anticipated_size = current_size + len(message.encode(self.encoding or "utf-8")) + 1
        return anticipated_size >= self.max_bytes


def build_logger(
    name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or return) a logger configured with sanitized stdout and JSONL handlers."""
    logger = logging.getLogger(name)
    if getattr(logger, "_is_dataaug_configured", False):
        return logger

    logger.setLevel(level)
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    json_path = log_directory / f"{name}.jsonl"
    json_handler = SizeAndTimeRotatingFileHandler(str(json_path))
    json_handler.setFormatter(JsonLogFormatter())

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(HumanReadableFormatter())

    logger.addHandler(stream_handler)
    logger.addHandler(json_handler)
    logger.propagate = False
    logger._is_dataaug_configured = True  # type: ignore[attr-defined]
    return logger


def format_storage_exhaustion_error(
    current_usage_gb: float,
    available_gb: float,
    next_checkpoint_size_gb: float,
    trial_dir: Path,
    artifact_sizes: Sequence[tuple[Path, float]] | None = None,
    retention_policy_hint: str | None = None,
) -> str:
    """
    Create an actionable error message when checkpoint retention cannot be satisfied.

    Parameters
    ----------
    current_usage_gb:
        Current disk usage for the trials directory.
    available_gb:
        Remaining free space on the volume.
    next_checkpoint_size_gb:
        Estimated size for the next checkpoint to be written.
    trial_dir:
        Directory that holds the trial artifacts.
    artifact_sizes:
        Optional sequence of ``(path, size_gb)`` pairs ordered from largest to smallest.
    retention_policy_hint:
        Optional text describing how to adjust the retention policy.
    """

    lines = [
        "Storage exhaustion detected: unable to satisfy checkpoint retention guarantees.",
        f"- Current usage: {current_usage_gb:.2f} GiB",
        f"- Available space: {available_gb:.2f} GiB",
        f"- Next checkpoint needs: {next_checkpoint_size_gb:.2f} GiB",
        f"- Trial directory: {trial_dir}",
    ]

    artifacts = list(artifact_sizes or [])
    if artifacts:
        lines.append("- Largest artifacts by size:")
        for path, size_gb in artifacts[:10]:
            lines.append(f"    • {path}: {size_gb:.2f} GiB")

    lines.append("- Recommended actions:")
    lines.append(f"    1. Inspect artifacts: du -h {trial_dir} | sort -hr | head")
    lines.append(
        f"    2. Remove stale checkpoints: find {trial_dir} -name 'checkpoint_*' -mtime +3 -delete"
    )
    if retention_policy_hint:
        lines.append(
            f"    3. Adjust retention policy: hydra override 'trainer.checkpoint.retention={retention_policy_hint}'"
        )
    else:
        lines.append(
            "    3. Reduce retention in Hydra config (trainer.checkpoint.retention=<smaller value>)"
        )
    lines.append(
        "    4. Re-run the trial after cleanup: poetry run python -m dataaug_multi_both.cli.resume"
    )

    return "\n".join(lines)
