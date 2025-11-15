"""MLflow tracking utilities for experiment management."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is available and configured."""
    if not MLFLOW_AVAILABLE:
        return False
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    return tracking_uri is not None and tracking_uri != ""


def setup_mlflow(
    experiment_name: str = "redsm5-classification",
    tracking_uri: str | None = None,
) -> None:
    """Initialize MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (defaults to env var)
    """
    if not MLFLOW_AVAILABLE:
        return

    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    if not is_mlflow_enabled():
        return
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to MLflow."""
    if not is_mlflow_enabled():
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: Path | str) -> None:
    """Log an artifact file to MLflow."""
    if not is_mlflow_enabled():
        return
    mlflow.log_artifact(str(local_path))


def log_model(model_path: Path | str, artifact_path: str = "model") -> None:
    """Log a model to MLflow."""
    if not is_mlflow_enabled():
        return
    mlflow.log_artifact(str(model_path), artifact_path=artifact_path)


def start_run(run_name: str | None = None, nested: bool = False) -> Any:
    """Start an MLflow run.

    Args:
        run_name: Name for the run
        nested: Whether this is a nested run

    Returns:
        MLflow run context manager
    """
    if not is_mlflow_enabled():
        return _NoOpContext()
    return mlflow.start_run(run_name=run_name, nested=nested)


def end_run() -> None:
    """End the active MLflow run."""
    if not is_mlflow_enabled():
        return
    mlflow.end_run()


def set_tag(key: str, value: Any) -> None:
    """Set a tag on the active run."""
    if not is_mlflow_enabled():
        return
    mlflow.set_tag(key, value)


def set_tags(tags: dict[str, Any]) -> None:
    """Set multiple tags on the active run."""
    if not is_mlflow_enabled():
        return
    mlflow.set_tags(tags)


class _NoOpContext:
    """No-op context manager when MLflow is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
