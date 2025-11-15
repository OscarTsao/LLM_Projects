"""Disk-backed MLflow buffering for resilient logging."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import mlflow

logger = logging.getLogger(__name__)


class MlflowBuffer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _record(self, payload: Dict[str, Any]) -> None:
        ts = int(time.time() * 1000)
        shard = self.root / f"mlflow_buffer_{ts}.jsonl"
        with shard.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def log_param(self, key: str, value: Any) -> None:
        try:
            mlflow.log_param(key, value)
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("MLflow param logging failed; buffering: %s", exc)
            self._record({"type": "param", "key": key, "value": value})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        try:
            if step is None:
                mlflow.log_metric(key, value)
            else:
                mlflow.log_metric(key, value, step=step)
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow metric logging failed; buffering: %s", exc)
            self._record({"type": "metric", "key": key, "value": value, "step": step})

    def log_tags(self, tags: Dict[str, Any]) -> None:
        try:
            mlflow.set_tags(tags)
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow tag logging failed; buffering: %s", exc)
            self._record({"type": "tags", "tags": tags})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        try:
            mlflow.log_artifact(str(path), artifact_path)
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow artifact logging failed; buffering pointer: %s", exc)
            self._record(
                {
                    "type": "artifact",
                    "path": str(path),
                    "artifact_path": artifact_path,
                }
            )

    def replay(self) -> None:
        shards = sorted(self.root.glob("mlflow_buffer_*.jsonl"))
        for shard in shards:
            try:
                with shard.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        payload = json.loads(line)
                        kind = payload.get("type")
                        if kind == "param":
                            mlflow.log_param(payload["key"], payload["value"])
                        elif kind == "metric":
                            mlflow.log_metric(payload["key"], payload["value"], step=payload.get("step"))
                        elif kind == "tags":
                            mlflow.set_tags(payload["tags"])
                        elif kind == "artifact":
                            mlflow.log_artifact(payload["path"], artifact_path=payload.get("artifact_path"))
                shard.unlink()
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to replay MLflow buffer %s: %s", shard, exc)


__all__ = ["MlflowBuffer"]

