from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Optional

import mlflow

logger = logging.getLogger(__name__)


@dataclass
class BufferSettings:
    directory: Path
    backoff_initial: float = 1.0
    backoff_max: float = 60.0
    max_records_per_replay: int = 200


class MlflowBufferedLogger:
    """Disk-backed MLflow logging buffer with best-effort replay."""

    def __init__(self, settings: BufferSettings) -> None:
        self.settings = settings
        self.settings.directory.mkdir(parents=True, exist_ok=True)
        self._buffer_path = self.settings.directory / "pending.jsonl"
        self._artifact_stash = self.settings.directory / "stash"
        self._artifact_stash.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._backoff = self.settings.backoff_initial
        self._next_retry_ts = 0.0

    # ------------------------------------------------------------------ utils
    def _write_event(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            with self._buffer_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.debug("Queued MLflow event for retry: type=%s", payload.get("type"))

    def _pop_events(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._buffer_path.exists():
                return []
            lines = self._buffer_path.read_text(encoding="utf-8").splitlines()
            self._buffer_path.unlink()
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupted MLflow buffer line: %s", line)
        return events

    def _stash_artifact(self, source: Path) -> Path:
        target = self._artifact_stash / f"{uuid.uuid4()}_{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    def _current_run_id(self) -> str:
        run = mlflow.active_run()
        return run.info.run_id if run else "no-active-run"

    # ---------------------------------------------------------------- actions
    def _attempt_replay(self, force: bool = False) -> None:
        now = time.time()
        if not force and now < self._next_retry_ts:
            return

        events = self._pop_events()
        if not events:
            self._backoff = self.settings.backoff_initial
            return

        to_retry: list[dict[str, Any]] = []
        processed = 0
        for payload in events:
            if processed >= self.settings.max_records_per_replay:
                to_retry.append(payload)
                continue
            processed += 1
            try:
                self._handle_event(payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Replay failed for %s: %s", payload.get("type"), exc)
                to_retry.append(payload)
                break  # Avoid tight retry loop

        if to_retry:
            with self._lock:
                with self._buffer_path.open("a", encoding="utf-8") as f:
                    for payload in to_retry:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._backoff = min(self._backoff * 2, self.settings.backoff_max)
            self._next_retry_ts = time.time() + self._backoff
        else:
            self._backoff = self.settings.backoff_initial
            self._next_retry_ts = 0.0

    def _handle_event(self, payload: dict[str, Any]) -> None:
        event_type = payload["type"]
        if event_type == "metrics":
            mlflow.log_metrics(payload["metrics"], step=payload.get("step"))
        elif event_type == "params":
            mlflow.log_params(payload["params"])
        elif event_type == "artifact":
            path = Path(payload["path"])
            artifact_path = payload.get("artifact_path")
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            # Clean up cached artifact after successful upload
            try:
                path.unlink()
            except FileNotFoundError:  # pragma: no cover - best effort
                pass
        else:  # pragma: no cover - unexpected type
            logger.warning("Unknown MLflow buffer event type: %s", event_type)

    # ---------------------------------------------------------------- logging
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]) -> None:
        try:
            mlflow.log_metrics(metrics, step=step)
            self._attempt_replay()
        except Exception as exc:  # pragma: no cover - depends on MLflow backend
            run_id = self._current_run_id()
            logger.warning(
                "Failed to log metrics to MLflow (run_id=%s). Buffering. Error: %s", run_id, exc
            )
            payload = {"type": "metrics", "metrics": metrics, "step": step, "run_id": run_id}
            self._write_event(payload)

    def log_params(self, params: Dict[str, Any]) -> None:
        try:
            mlflow.log_params(params)
            self._attempt_replay()
        except Exception as exc:  # pragma: no cover
            run_id = self._current_run_id()
            logger.warning(
                "Failed to log params to MLflow (run_id=%s). Buffering. Error: %s", run_id, exc
            )
            payload = {"type": "params", "params": params, "run_id": run_id}
            self._write_event(payload)

    def log_artifact(self, path: Path, artifact_path: Optional[str]) -> None:
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
            self._attempt_replay()
        except Exception as exc:  # pragma: no cover
            run_id = self._current_run_id()
            logger.warning(
                "Failed to log artifact to MLflow (run_id=%s). Buffering. Error: %s", run_id, exc
            )
            cached = self._stash_artifact(path)
            payload = {
                "type": "artifact",
                "path": str(cached),
                "artifact_path": artifact_path,
                "run_id": run_id,
            }
            self._write_event(payload)

    # -------------------------------------------------------------- lifecycle
    def on_run_start(self) -> None:
        self._attempt_replay(force=True)

    def on_run_end(self) -> None:
        self._attempt_replay(force=True)


_BUFFER: Optional[MlflowBufferedLogger] = None


def configure_buffer(cfg: Mapping[str, Any]) -> None:
    global _BUFFER
    if _BUFFER is not None:
        return

    buffer_cfg = cfg.get("mlflow", {}).get("buffer", {})
    directory = Path(buffer_cfg.get("dir", "./artifacts/mlflow_buffer"))
    settings = BufferSettings(
        directory=directory,
        backoff_initial=float(buffer_cfg.get("backoff_initial", 1.0)),
        backoff_max=float(buffer_cfg.get("backoff_max", 60.0)),
        max_records_per_replay=int(buffer_cfg.get("max_records_per_replay", 200)),
    )
    _BUFFER = MlflowBufferedLogger(settings)
    logger.debug("Configured MLflow buffered logger at %s", directory)


def _ensure_buffer() -> MlflowBufferedLogger:
    if _BUFFER is None:
        raise RuntimeError("MLflow buffer not configured. Call configure_buffer first.")
    return _BUFFER


def on_run_start() -> None:
    _ensure_buffer().on_run_start()


def on_run_end() -> None:
    _ensure_buffer().on_run_end()


def log_metric_safe(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    _ensure_buffer().log_metrics(metrics, step)


def log_params_safe(params: Dict[str, Any]) -> None:
    _ensure_buffer().log_params(params)


def log_artifact_safe(path: Path, artifact_path: Optional[str] = None) -> None:
    _ensure_buffer().log_artifact(path, artifact_path)
