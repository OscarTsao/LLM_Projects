"""MLflow tracking utilities for experiment management."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore

logger = logging.getLogger(__name__)

MLFLOW_ACTIVE = True
DEFAULT_EXPERIMENT = "redsm5-classification"
DEFAULT_FALLBACK_URI = "sqlite:///mlflow.db"


def _is_remote_tracking_uri(uri: str | None) -> bool:
    if not uri:
        return False
    return uri.startswith("http://") or uri.startswith("https://")


def _normalize_file_uri(uri: str) -> str:
    """Convert relative file URIs into absolute URIs understood by MLflow."""
    if uri.startswith("file:"):
        path_part = uri[5:]
        path = Path(path_part)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return f"file:{path.as_posix()}"
    if "://" not in uri:
        path = Path(uri)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return f"file:{path.as_posix()}"
    return uri


def _tracking_uri_reachable(uri: str) -> bool:
    """Check whether an MLflow tracking server can be reached."""
    if not _is_remote_tracking_uri(uri):
        return True
    if not MLFLOW_AVAILABLE:
        return False
    try:
        client = MlflowClient(tracking_uri=uri)
        if hasattr(client, "search_experiments"):
            client.search_experiments(max_results=1)
        else:  # pragma: no cover - legacy fallback
            client.list_experiments()
        return True
    except Exception:  # pragma: no cover - depends on external connectivity
        return False


def _dict_from_cfg(node: DictConfig | dict[str, Any] | None) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, DictConfig):
        return dict(OmegaConf.to_container(node, resolve=True))  # type: ignore[arg-type]
    return dict(node)


def _extract_mlflow_settings(config: Any) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        section = config.get("mlflow", None)
        return _dict_from_cfg(section)
    if isinstance(config, dict):
        if "mlflow" in config and isinstance(config["mlflow"], (DictConfig, dict)):
            return _dict_from_cfg(config["mlflow"])
        if any(key in config for key in ("enabled", "tracking_uri", "experiment_name")):
            return _dict_from_cfg(config)
    return {}


def is_mlflow_enabled() -> bool:
    """Check if MLflow tracking is available and configured."""
    if not (MLFLOW_AVAILABLE and MLFLOW_ACTIVE):
        return False
    tracking_uri = mlflow.get_tracking_uri()
    return tracking_uri is not None and tracking_uri != ""


def setup_mlflow(
    config_or_experiment: DictConfig | dict[str, Any] | str | None = None,
    *,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    enabled: bool | None = None,
    fallback_tracking_uri: str | None = None,
) -> None:
    """Initialize MLflow tracking with container-aware defaults and graceful fallback."""
    if not MLFLOW_AVAILABLE:
        return

    global MLFLOW_ACTIVE

    cfg = _extract_mlflow_settings(config_or_experiment)

    if isinstance(config_or_experiment, str):
        experiment_name = experiment_name or config_or_experiment

    experiments_cfg = cfg.get("experiments")
    if not isinstance(experiments_cfg, dict):
        experiments_cfg = {}

    enabled = cfg.get("enabled", enabled if enabled is not None else True)
    if not enabled:
        MLFLOW_ACTIVE = False
        logger.info("MLflow logging disabled via configuration.")
        return

    experiment_name = (
        experiment_name
        or experiments_cfg.get("training")
        or cfg.get("experiment_name")
        or DEFAULT_EXPERIMENT
    )
    tracking_uri = tracking_uri or cfg.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI")
    fallback_tracking_uri = (
        fallback_tracking_uri
        or cfg.get("fallback_tracking_uri")
        or os.getenv("MLFLOW_FALLBACK_TRACKING_URI")
        or DEFAULT_FALLBACK_URI
    )

    if tracking_uri:
        tracking_uri = _normalize_file_uri(tracking_uri)
    if fallback_tracking_uri:
        fallback_tracking_uri = _normalize_file_uri(fallback_tracking_uri)

    if tracking_uri and not _tracking_uri_reachable(tracking_uri):
        logger.warning(
            "MLflow tracking URI %s is unreachable; switching to fallback %s.",
            tracking_uri,
            fallback_tracking_uri,
        )
        tracking_uri = fallback_tracking_uri

    if not tracking_uri:
        tracking_uri = fallback_tracking_uri

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        MLFLOW_ACTIVE = True
        logger.debug("MLflow tracking initialised at %s (experiment=%s).", tracking_uri, experiment_name)
        return
    except Exception as exc:  # pragma: no cover - relies on external service
        logger.warning(
            "Failed to initialise MLflow at %s (%s). Attempting fallback tracking URI %s.",
            tracking_uri,
            exc,
            fallback_tracking_uri,
        )

    try:
        mlflow.set_tracking_uri(fallback_tracking_uri)
        mlflow.set_experiment(experiment_name)
        MLFLOW_ACTIVE = True
        logger.info(
            "MLflow logging fallback activated using %s (experiment=%s).",
            fallback_tracking_uri,
            experiment_name,
        )
    except Exception as exc:  # pragma: no cover - relies on external service
        MLFLOW_ACTIVE = False
        logger.error("Fallback MLflow initialisation failed (%s). MLflow logging disabled.", exc)


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
    """Start an MLflow run."""
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
