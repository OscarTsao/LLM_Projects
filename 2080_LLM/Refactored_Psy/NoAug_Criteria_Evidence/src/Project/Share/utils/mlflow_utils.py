from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TRACKING_URI = f"sqlite:///{(_ROOT / 'mlflow.db').resolve()}"
_DEFAULT_ARTIFACT_URI = (_ROOT / "mlruns").resolve()
_DEFAULT_ARTIFACT_URI.mkdir(parents=True, exist_ok=True)
_DEFAULT_ARTIFACT_URI_STR = _DEFAULT_ARTIFACT_URI.as_uri()


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    artifact_location: Optional[str] = None,
) -> None:
    """Configure MLflow tracking URI, experiment, and default tags.

    Defaults target the local SQLite backend store (`mlflow.db`) and artifact root
    (`mlruns/`) to keep training/HPO runs self-contained inside the repository.
    """
    import mlflow

    resolved_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    resolved_artifact_uri = artifact_location or os.environ.get(
        "MLFLOW_ARTIFACT_URI", _DEFAULT_ARTIFACT_URI_STR
    )

    os.environ.setdefault("MLFLOW_TRACKING_URI", resolved_tracking_uri)
    os.environ.setdefault("MLFLOW_ARTIFACT_URI", resolved_artifact_uri)

    mlflow.set_tracking_uri(resolved_tracking_uri)

    if experiment:
        try:
            existing_experiment = mlflow.get_experiment_by_name(experiment)
            if existing_experiment is None:
                mlflow.create_experiment(experiment, artifact_location=resolved_artifact_uri)
        except Exception:
            # Fall back to MLflow's internal handling if anything fails
            pass
        mlflow.set_experiment(experiment)

    if tags:
        try:
            mlflow.set_tags(tags)
        except Exception:
            # set_tags requires an active run; ignore if none is active
            pass


def enable_autologging(enable: bool = True) -> None:
    """Enable or disable MLflow autologging with sensible defaults.

    Attempts framework-specific autologging when available and falls back
    to generic `mlflow.autolog`.
    """
    import mlflow

    if not enable:
        mlflow.autolog(disable=True)
        return

    # Prefer generic autolog (works for many frameworks in MLflow>=2)
    try:
        mlflow.autolog()
        return
    except Exception:
        pass

    # Fall back to common framework-specific autologging if present
    for mod_name in ("mlflow.pytorch", "mlflow.sklearn", "mlflow.xgboost", "mlflow.lightgbm"):
        try:
            mod = __import__(mod_name, fromlist=["autolog"])  # type: ignore
            getattr(mod, "autolog")()
        except Exception:
            continue


@contextlib.contextmanager
def mlflow_run(
    name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Context manager that starts and ends an MLflow run.

    Usage:
        with mlflow_run("demo", tags={"stage": "dev"}):
            ... your training loop ...
    """
    import mlflow

    with mlflow.start_run(run_name=name, nested=nested) as run:
        if tags:
            try:
                mlflow.set_tags(tags)
            except Exception:
                pass
        if params:
            try:
                mlflow.log_params(params)
            except Exception:
                pass
        yield run
