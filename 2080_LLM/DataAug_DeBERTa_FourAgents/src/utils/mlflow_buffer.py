from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class MlflowBufferedLogger:
    """A tiny MLflow wrapper with disk-backed buffering.

    - On log failures, appends records to JSONL buffer files under
      `artifacts/mlflow_buffer/<run_id>/`.
    - `replay()` attempts to flush buffered records back to MLflow
      with exponential backoff and jitter. Failures are ignored.
    - If MLflow is unavailable, calls are no-ops except buffering.
    """

    def __init__(self, run_id: str, base_dir: str | Path = "artifacts/mlflow_buffer") -> None:
        self.run_id = str(run_id)
        self.base_dir = Path(base_dir) / self.run_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    # --------------- internal helpers ---------------
    def _append_buffer(self, kind: str, payload: Dict[str, Any]) -> None:
        path = self.base_dir / f"{kind}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def _read_buffer(self, kind: str) -> List[Dict[str, Any]]:
        path = self.base_dir / f"{kind}.jsonl"
        if not path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def _clear_buffer(self, kind: str) -> None:
        path = self.base_dir / f"{kind}.jsonl"
        try:
            path.unlink()
        except Exception:
            pass

    # --------------- public API ---------------
    def replay(self) -> None:
        if self._mlflow is None:
            return
        kinds = ["metrics", "params", "tags"]
        for kind in kinds:
            records = self._read_buffer(kind)
            if not records:
                continue
            delay = 0.25
            attempts = 0
            while records and attempts < 6:
                try:
                    if kind == "metrics":
                        for rec in records:
                            metrics = rec.get("metrics", {})
                            step = rec.get("step")
                            timestamp = rec.get("timestamp")
                            self._mlflow.log_metrics(metrics, step=step)
                    elif kind == "params":
                        for rec in records:
                            params = rec.get("params", {})
                            self._mlflow.log_params(params)
                    elif kind == "tags":
                        for rec in records:
                            tags = rec.get("tags", {})
                            self._mlflow.set_tags(tags)
                    # success
                    self._clear_buffer(kind)
                    break
                except Exception:
                    attempts += 1
                    time.sleep(delay + random.random() * 0.1)
                    delay = min(delay * 2, 4.0)

    def log_metrics(self, dict_or_list: Dict[str, float] | Iterable[Dict[str, Any]], step: Optional[int] = None, timestamp: Optional[int] = None) -> None:
        try:
            if self._mlflow is not None:
                if isinstance(dict_or_list, dict):
                    self._mlflow.log_metrics(dict_or_list, step=step)
                else:
                    for d in dict_or_list:
                        self._mlflow.log_metrics(d, step=d.get("step", step))
            else:
                raise RuntimeError("mlflow unavailable")
        except Exception:
            payload = {"metrics": dict_or_list if isinstance(dict_or_list, dict) else list(dict_or_list), "step": step, "timestamp": timestamp}
            self._append_buffer("metrics", payload)

    def log_params(self, params: Dict[str, Any]) -> None:
        try:
            if self._mlflow is not None:
                self._mlflow.log_params(params)
            else:
                raise RuntimeError("mlflow unavailable")
        except Exception:
            self._append_buffer("params", {"params": params})

    def set_tags(self, tags: Dict[str, Any]) -> None:
        try:
            if self._mlflow is not None:
                self._mlflow.set_tags(tags)
            else:
                raise RuntimeError("mlflow unavailable")
        except Exception:
            self._append_buffer("tags", {"tags": tags})

    # Artifacts are attempted but not buffered
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        try:
            if self._mlflow is not None:
                self._mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception:
            pass

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        try:
            if self._mlflow is not None:
                self._mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        except Exception:
            pass

