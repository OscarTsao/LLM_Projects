from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna

from ..utils.mlflow_setup import BufferedMLflowLogger
from .samplers_pruners import build_pruner, build_sampler


@dataclass(slots=True)
class DatasetConfig:
    dataset_id: str
    revision: str | None = None
    cache_dir: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"


@dataclass(slots=True)
class HardwareConfig:
    max_length: int = 512
    per_device_batch_size: int = 8
    grad_accumulation_steps: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    amp_dtype: str = "none"
    gradient_checkpointing: bool = False

    def validate(self) -> None:
        if self.max_length > 512:
            self.max_length = 512
        if self.grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1")


@dataclass(slots=True)
class StageConfig:
    stage: str
    max_epochs: int
    patience: int = 20
    seed: int = 42
    deterministic: bool = False


@dataclass(slots=True)
class TrialResult:
    metric: float
    threshold: float
    run_id: str
    params: dict[str, Any]
    best_epoch: int
    epochs_trained: int
    duration_seconds: float


def _create_run_dir(experiments_dir: str, stage: str, trial_number: int) -> Path:
    run_dir = Path(experiments_dir) / "runs" / f"stage_{stage}" / f"trial_{trial_number:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _log_trial_tags(
    logger: BufferedMLflowLogger,
    stage: str,
    trial: optuna.Trial,
    metric_value: float | None,
    elapsed: float,
) -> None:
    study = trial.study
    total_trials = len(study.trials)
    trial_index = trial.number + 1
    completion_rate = trial_index / max(1, total_trials)
    eta_seconds = (elapsed / completion_rate) - elapsed if completion_rate > 0 else 0.0

    try:
        best_trial = study.best_trial  # type: ignore[assignment]
        best_value = float(best_trial.value) if best_trial.value is not None else float("nan")
        best_id = best_trial.number
    except Exception:
        best_value = float(metric_value) if metric_value is not None else float("nan")
        best_id = trial.number

    tags = {
        "stage": stage,
        "trial_index": str(trial_index),
        "trial_total": str(total_trials),
        "completion_rate": f"{completion_rate:.4f}",
        "elapsed_seconds": f"{elapsed:.1f}",
        "eta_seconds": f"{eta_seconds:.1f}",
        "best_metric": f"{best_value:.6f}",
        "best_trial_id": str(best_id),
    }
    if metric_value is not None:
        tags["metric.last"] = f"{metric_value:.6f}"
    logger.set_tags(tags)


def objective_factory(
    stage: str,
    max_epochs: int,
    trainer_fn: Callable[..., Mapping[str, Any]],
    eval_metric_name: str = "macro_f1",
    experiments_dir: str = "experiments",
    base_seed: int = 42,
    trainer_kwargs: Mapping[str, Any] | None = None,
) -> Callable[[optuna.Trial], float]:
    trainer_kwargs = dict(trainer_kwargs or {})

    def objective(trial: optuna.Trial) -> float:
        start_time = time.time()
        run_dir = _create_run_dir(experiments_dir, stage, trial.number)
        logger = BufferedMLflowLogger(run_dir)

        seed = base_seed + trial.number
        logger.set_tags({"seed.realized": str(seed)})

        try:
            result = trainer_fn(
                trial=trial,
                stage=stage,
                max_epochs=max_epochs,
                seed=seed,
                mlflow_logger=logger,
                run_dir=run_dir,
                **trainer_kwargs,
            )
            metric_value = float(result.get(eval_metric_name, float("nan")))
        except optuna.TrialPruned:
            _log_trial_tags(logger, stage, trial, None, time.time() - start_time)
            raise
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                logger.set_tags({"trial_pruned": "1", "prune_reason": "CUDA OOM"})
                _log_trial_tags(logger, stage, trial, None, time.time() - start_time)
                raise optuna.TrialPruned("CUDA OOM") from exc
            raise

        elapsed = time.time() - start_time
        logger.log_metrics({eval_metric_name: metric_value}, step=max_epochs)
        _log_trial_tags(logger, stage, trial, metric_value, elapsed)
        return metric_value

    return objective


def run_study(
    stage: str,
    study_name: str,
    storage_url: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    timeout_seconds: int,
    max_epochs: int,
    callbacks: list[Callable[[optuna.Study, optuna.FrozenTrial], None]],
) -> optuna.Study:
    start_time = time.time()

    def _console_progress_callback(study: optuna.Study, frozen_trial: optuna.FrozenTrial) -> None:
        try:
            completed = sum(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
            total = max(n_trials, completed)
            best_val = float(study.best_value) if study.best_value is not None else float("nan")
        except Exception:
            completed = len(study.trials)
            total = max(n_trials, completed)
            best_val = float("nan")

        elapsed = time.time() - start_time
        rate = completed / max(1, total)
        eta = (elapsed / rate - elapsed) if rate > 0 else 0.0
        state_name = str(frozen_trial.state).split(".")[-1]
        val_str = "NA" if frozen_trial.value is None else f"{float(frozen_trial.value):.4f}"
        print(
            f"[HPO {stage}] Trial {frozen_trial.number} {state_name} value={val_str} "
            f"| Completed {completed}/{total} | Best={best_val:.4f} | Elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
        )

    sampler = build_sampler(stage)
    pruner = build_pruner(stage, max_epochs)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    # Always include console progress along with any provided callbacks
    all_callbacks = list(callbacks) + [_console_progress_callback]
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=all_callbacks,
        n_jobs=1,
        gc_after_trial=True,
    )
    return study


__all__ = [
    "DatasetConfig",
    "HardwareConfig",
    "StageConfig",
    "TrialResult",
    "objective_factory",
    "run_study",
]
