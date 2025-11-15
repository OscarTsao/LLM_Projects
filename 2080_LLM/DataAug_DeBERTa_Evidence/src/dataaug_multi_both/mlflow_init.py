from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import mlflow
import torch

from dataaug_multi_both.config import flatten_dict
from dataaug_multi_both.mlflow_buffer import (
    configure_buffer,
    log_artifact_safe,
    log_metric_safe,
    log_params_safe,
    on_run_end,
    on_run_start,
)

logger = logging.getLogger(__name__)


def configure_mlflow(cfg: Mapping[str, Any]) -> None:
    mlflow_cfg = cfg["mlflow"]
    tracking_uri = mlflow_cfg["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    configure_buffer(cfg)


def _safe_git_info() -> dict[str, Any]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return {"git_commit": commit, "git_dirty": bool(dirty)}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": True}


def _hardware_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_memory"] = torch.cuda.get_device_properties(0).total_memory
    else:
        info["gpu_name"] = "cpu"
        info["gpu_total_memory"] = 0
    return info


def _chunk_dict_items(items: Iterable[tuple[str, Any]], chunk_size: int = 100) -> Iterable[dict[str, Any]]:
    bucket: dict[str, Any] = {}
    for key, value in items:
        bucket[key] = value
        if len(bucket) >= chunk_size:
            yield bucket
            bucket = {}
    if bucket:
        yield bucket


def log_run_metadata(cfg: Mapping[str, Any], dataset_meta: Mapping[str, Any]) -> None:
    params: dict[str, Any] = {}
    for key, value in flatten_dict(cfg).items():
        if isinstance(value, (dict, list)):
            value = str(value)
        params[f"cfg.{key}"] = value

    data_params = {f"data.{key}": value for key, value in dataset_meta.items()}
    git_params = _safe_git_info()
    hardware_params = {f"hardware.{key}": value for key, value in _hardware_info().items()}

    for chunk in _chunk_dict_items(params.items()):
        log_params_safe(chunk)
    log_params_safe(data_params)
    log_params_safe(git_params)
    log_params_safe(hardware_params)


def log_metrics(metrics: Mapping[str, float], step: int) -> None:
    log_metric_safe(dict(metrics), step=step)


def log_evaluation_artifact(report: Mapping[str, Any], output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evaluation_report.json"
    with path.open("w", encoding="utf-8") as f:
        import json

        json.dump(report, f, indent=2)
    log_artifact_safe(path, artifact_path=f"experiments/{run_id}")
    return path


def register_run_start() -> None:
    on_run_start()


def register_run_end() -> None:
    on_run_end()
