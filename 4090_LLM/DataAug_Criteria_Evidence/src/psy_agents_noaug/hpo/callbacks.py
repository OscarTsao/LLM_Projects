"""Callback helpers for Optuna trials."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from typing import Any

import optuna

try:  # Optional dependency in test environments
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional import
    MLFLOW_AVAILABLE = False

from .utils import TopKStore, TrialSummary


@dataclass
class MlflowConfig:
    tracking_uri: str | None
    experiment: str
    tags: dict[str, Any] | None = None


class MlflowCallback:
    """Lightweight MLflow integration for Optuna trials."""

    def __init__(self, config: MlflowConfig | None) -> None:
        self.config = config
        self._active_run = None

    def start(self, *, trial: optuna.Trial, agent: str, study: str) -> None:
        if not (self.config and MLFLOW_AVAILABLE):
            return
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment)
        run_name = f"trial-{trial.number}"
        tags = {
            "agent": agent,
            "study": study,
            "optuna_trial": trial.number,
        }
        if self.config.tags:
            tags.update(self.config.tags)
        self._active_run = mlflow.start_run(run_name=run_name, nested=True, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._active_run and MLFLOW_AVAILABLE:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._active_run and MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics)

    def log_json(self, name: str, payload: dict[str, Any]) -> None:
        if self._active_run and MLFLOW_AVAILABLE:
            mlflow.log_dict(payload, name)

    def finish(self, status: str = "FINISHED") -> str | None:
        if self._active_run and MLFLOW_AVAILABLE:
            run_id = self._active_run.info.run_id  # type: ignore[assignment]
            mlflow.set_tag("status", status)
            mlflow.end_run(status=status)
            self._active_run = None
            return run_id
        return None


class TrialCallbackManager:
    """Bundle MLflow logging and Top-K bookkeeping."""

    def __init__(
        self,
        *,
        mlflow_cfg: MlflowConfig | None,
        topk_store: TopKStore,
    ) -> None:
        self.mlflow = MlflowCallback(mlflow_cfg)
        self.topk_store = topk_store

    @contextlib.contextmanager
    def context(self, *, trial: optuna.Trial, agent: str, study: str):
        self.mlflow.start(trial=trial, agent=agent, study=study)
        try:
            yield
        except Exception:
            self.mlflow.finish(status="FAILED")
            raise

    def log_trial(
        self,
        trial: optuna.Trial,
        *,
        params: dict[str, Any],
        metrics: dict[str, float],
        seeds: list[int],
        started_at: float | None,
    ) -> None:
        if params:
            flat_params = {
                key: value
                for key, value in params.items()
                if isinstance(value, (str, int, float, bool))
            }
            self.mlflow.log_params(flat_params)
            self.mlflow.log_json("config.json", params)

        if metrics:
            self.mlflow.log_metrics(metrics)

        artifact_uri = self.mlflow.finish(status="FINISHED")

        summary = TrialSummary(
            trial_number=trial.number,
            f1_macro=float(metrics.get("f1_macro", 0.0)),
            ece=metrics.get("ece"),
            logloss=metrics.get("logloss"),
            runtime_s=metrics.get("runtime_s"),
            seed_info=seeds,
            params=params,
            artifact_uri=artifact_uri,
            started_at=started_at,
        )
        self.topk_store.record(summary)


def encode_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sanitise params for logging (Optuna handles nested dictionaries poorly)."""

    return {
        key: (json.dumps(value) if isinstance(value, dict) else value)
        for key, value in params.items()
    }
