"""Optuna objective builder for PSY Agents HPO."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import optuna
import torch

from .callbacks import MlflowConfig, TrialCallbackManager, encode_params
from .evaluation import run_experiment
from .spaces import SearchSpace, SpaceConstraints
from .utils import TopKStore, TrialTimer, ensure_directory


@dataclass
class ObjectiveSettings:
    """Configuration bundle for building the optimisation objective."""

    agent: str
    study: str
    outdir: Path
    epochs: int
    seeds: Sequence[int]
    patience: int
    max_samples: int | None
    multi_objective: bool
    topk: int
    mlflow_uri: str | None
    mlflow_experiment: str


class ObjectiveBuilder:
    """Callable Optuna objective that orchestrates training and logging."""

    def __init__(
        self,
        space: SearchSpace,
        settings: ObjectiveSettings,
        constraints: SpaceConstraints | None = None,
    ) -> None:
        self.space = space
        self.settings = settings
        self.constraints = constraints or SpaceConstraints()
        ensure_directory(settings.outdir)

        self.topk_store = TopKStore(
            outdir=settings.outdir,
            agent=settings.agent,
            study=settings.study,
            k=settings.topk,
        )

        mlflow_cfg = None
        if settings.mlflow_uri:
            mlflow_cfg = MlflowConfig(
                tracking_uri=settings.mlflow_uri,
                experiment=settings.mlflow_experiment,
            )

        self.callbacks = TrialCallbackManager(
            mlflow_cfg=mlflow_cfg,
            topk_store=self.topk_store,
        )

    def __call__(self, trial: optuna.Trial) -> float | tuple[float, float]:
        seeds = [int(s) for s in self.settings.seeds]
        params = self.space.sample(trial, self.constraints)
        trial.set_user_attr("config_json", json.dumps(params))
        trial.set_user_attr("seeds", seeds)

        timer = TrialTimer()

        with self.callbacks.context(
            trial=trial,
            agent=self.settings.agent,
            study=self.settings.study,
        ):
            with timer:
                try:
                    metrics = run_experiment(
                        self.settings.agent,
                        params,
                        epochs=self.settings.epochs,
                        seeds=seeds,
                        patience=self.settings.patience,
                        max_samples=self.settings.max_samples,
                    )
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if ("out of memory" in message) or (
                        "cuda" in message and "memory" in message
                    ):
                        trial.set_user_attr("oom", True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise optuna.TrialPruned("CUDA OOM") from exc
                    raise

        metrics["runtime_s"] = metrics.get("runtime_s") or timer.duration
        trial.set_user_attr("primary", metrics["f1_macro"])
        trial.set_user_attr("ece", metrics["ece"])
        trial.set_user_attr("logloss", metrics["logloss"])
        trial.set_user_attr("runtime_s", metrics["runtime_s"])
        trial.set_user_attr("started_at", timer.start)

        loggable = {
            "f1_macro": metrics["f1_macro"],
            "ece": metrics["ece"],
            "logloss": metrics["logloss"],
            "runtime_s": metrics["runtime_s"],
        }
        self.callbacks.log_trial(
            trial,
            params=encode_params(params),
            metrics=loggable,
            seeds=list(seeds),
            started_at=timer.start,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.settings.multi_objective:
            return (metrics["f1_macro"], metrics["ece"])
        return float(metrics["f1_macro"])
