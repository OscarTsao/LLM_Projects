from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import mlflow


def init_mlflow(experiments_dir: str, experiment_name: str) -> str:
    exp_dir = Path(experiments_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    backend = (exp_dir / "mlflow.db").resolve()
    artifacts = (exp_dir / "artifacts").resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{backend.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def log_system_fingerprint() -> None:
    try:
        import torch
        import transformers

        mlflow.log_param("lib.torch", torch.__version__)
        mlflow.log_param("lib.transformers", transformers.__version__)
        if torch.cuda.is_available():
            mlflow.log_param("cuda.device_count", torch.cuda.device_count())
            mlflow.log_param("cuda.name.0", torch.cuda.get_device_name(0))
    except Exception:
        pass
    try:
        mlflow.log_param("sys.python_version", platform.python_version())
        mlflow.log_param("sys.platform", platform.platform())
    except Exception:
        pass
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.log_param("git.sha", git_sha)
    except Exception:
        pass


class BufferedMLflowLogger:
    """Buffers MLflow writes to disk if MLflow is temporarily unavailable."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.buf = self.run_dir / "mlflow_buffer.jsonl"
        self.buf.parent.mkdir(parents=True, exist_ok=True)

    def _write_buf(self, rec: dict) -> None:
        with self.buf.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _replay(self, fn, payload: dict, backoff: float = 0.25, tries: int = 5) -> bool:
        for attempt in range(tries):
            try:
                fn(**payload)
                return True
            except Exception:
                time.sleep(backoff * (2**attempt))
        self._write_buf({"fn": fn.__name__, "payload": payload})
        return False

    def replay_buffer(self) -> None:
        if not self.buf.exists():
            return
        lines = self.buf.read_text(encoding="utf-8").splitlines()
        self.buf.unlink(missing_ok=True)
        for line in lines:
            try:
                record = json.loads(line)
                name = record.get("fn")
                payload = record.get("payload", {})
                if name == "log_params":
                    mlflow.log_params(payload["params"])
                elif name == "log_metrics":
                    mlflow.log_metrics(payload["metrics"], step=payload.get("step"))
                elif name == "set_tags":
                    mlflow.set_tags(payload["tags"])
                elif name == "log_artifact":
                    mlflow.log_artifact(payload["local_path"], artifact_path=payload.get("artifact_path"))
            except Exception:
                continue

    def log_params(self, params: dict) -> None:
        self._replay(mlflow.log_params, {"params": params})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._replay(mlflow.log_metrics, {"metrics": metrics, "step": step})

    def set_tags(self, tags: dict) -> None:
        self._replay(mlflow.set_tags, {"tags": tags})

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self._replay(mlflow.log_artifact, {"local_path": local_path, "artifact_path": artifact_path})


__all__ = ["init_mlflow", "log_system_fingerprint", "BufferedMLflowLogger"]
