from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Iterator


@contextlib.contextmanager
def mlflow_run(run_name: str, params: Dict[str, str]):
    """Best-effort MLflow run wrapper.

    Tries to use the real mlflow package, otherwise falls back to a
    no-op context manager while still ensuring the mlruns directory exists.
    """

    tracking_uri = "file:./mlruns"
    Path("mlruns").mkdir(exist_ok=True)

    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("four-agent-pipeline")
        with mlflow.start_run(run_name=run_name):
            for key, value in params.items():
                mlflow.log_param(key, value)
            yield mlflow
    except Exception:
        yield None

