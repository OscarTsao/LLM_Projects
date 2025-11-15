"""MLflow initialisation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlflow

from dataaug_multi_both.mlflow_buffer import MlflowBuffer

logger = logging.getLogger(__name__)


def init_mlflow(
    tracking_uri: str | None,
    experiment_name: str,
    buffer_dir: Path,
) -> MlflowBuffer:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    buffer = MlflowBuffer(buffer_dir)

    try:
        buffer.replay()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to replay MLflow buffer on init: %s", exc)

    return buffer


__all__ = ["init_mlflow"]

