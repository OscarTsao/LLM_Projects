"""Helpers for configuring MLflow within the project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
from omegaconf import DictConfig, OmegaConf


def _resolve_relative_path(path_str: str, project_root: Path) -> Path:
    """Resolve a filesystem path relative to the project root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _normalize_sqlite_uri(uri: str, project_root: Path) -> str:
    """Ensure SQLite URIs point inside the project and create parent directories."""
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri

    path_str = uri[len(prefix) :]
    # Detect relative paths that appear with a single leading slash
    is_windows_drive = len(path_str) >= 3 and path_str[1] == ":" and path_str[2] == "/"
    if path_str.startswith("/") and not path_str.startswith("//") and not is_windows_drive:
        # Drop the artificial leading slash introduced by the URI form
        path_str = path_str[1:]

    path = _resolve_relative_path(path_str, project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{path.as_posix()}"


def _normalize_file_uri(uri: str, project_root: Path) -> str:
    """Ensure file URIs are anchored in the project directory."""
    prefix = "file:"
    if not uri.startswith(prefix):
        return uri

    path_str = uri[len(prefix) :]
    # Support both file:relative/path and file:///absolute/path
    if path_str.startswith("//"):
        path = Path("//" + path_str.lstrip("/")).expanduser().resolve()
    else:
        if path_str.startswith("/"):
            candidate = Path(path_str).expanduser()
        else:
            candidate = _resolve_relative_path(path_str, project_root)
        path = candidate.resolve()

    path.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{path.as_posix()}"


def resolve_mlflow_uris(
    cfg: DictConfig,
    project_root: Optional[Path] = None,
) -> tuple[str, Optional[str]]:
    """Resolve the MLflow tracking and artifact URIs to absolute locations.

    Args:
        cfg: MLflow configuration block.
        project_root: Explicit project root. Defaults to Hydra's original cwd.

    Returns:
        Tuple containing the normalized tracking URI and optional artifact URI.
    """
    if project_root is None:
        hydra_runtime_root = OmegaConf.select(cfg, "hydra.runtime.cwd", default=os.getcwd())
        project_root = Path(hydra_runtime_root).resolve()

    tracking_uri: str = cfg.mlflow.tracking_uri
    artifact_uri: Optional[str] = getattr(cfg.mlflow, "artifact_uri", None)

    if tracking_uri.startswith("sqlite:///"):
        tracking_uri = _normalize_sqlite_uri(tracking_uri, project_root)
    elif tracking_uri.startswith("file:"):
        tracking_uri = _normalize_file_uri(tracking_uri, project_root)

    if artifact_uri:
        if artifact_uri.startswith("sqlite:///"):
            artifact_uri = _normalize_sqlite_uri(artifact_uri, project_root)
        elif artifact_uri.startswith("file:"):
            artifact_uri = _normalize_file_uri(artifact_uri, project_root)
        else:
            resolved = _resolve_relative_path(artifact_uri, project_root)
            resolved.mkdir(parents=True, exist_ok=True)
            artifact_uri = resolved.as_posix()

    return tracking_uri, artifact_uri


def configure_mlflow(cfg: DictConfig) -> tuple[str, Optional[str]]:
    """Configure MLflow to use the local SQLite-backed tracking store.

    Args:
        cfg: Full Hydra configuration containing the ``mlflow`` block.

    Returns:
        The normalized tracking URI and artifact URI.
    """
    default_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    hydra_root = OmegaConf.select(cfg, "hydra.runtime.cwd", default=default_root)
    project_root = Path(hydra_root).resolve()
    tracking_uri, artifact_uri = resolve_mlflow_uris(cfg, project_root)

    mlflow.set_tracking_uri(tracking_uri)
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)

    experiment_name = cfg.mlflow.experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_uri)
    elif artifact_uri and experiment.artifact_location != artifact_uri:
        # Existing experiments keep their artifact URI; surface a helpful hint.
        print(
            f"MLflow experiment '{experiment_name}' already exists with "
            f"artifact location {experiment.artifact_location}. "
            f"Configured artifact URI {artifact_uri} will be kept for new experiments."
        )

    mlflow.set_experiment(experiment_name)
    if artifact_uri:
        os.environ.setdefault("MLFLOW_ARTIFACT_URI", artifact_uri)

    return tracking_uri, artifact_uri


__all__ = ["configure_mlflow", "resolve_mlflow_uris"]
