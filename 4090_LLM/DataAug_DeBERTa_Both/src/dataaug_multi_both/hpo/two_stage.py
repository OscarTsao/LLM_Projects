from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import optuna
from optuna import Study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, Trial

from .search_space_v2 import (
    HardwareCapabilities,
    NarrowedSpace,
    Stage,
    detect_hardware_capabilities,
    narrow_stage2_space,
    sample_parameters,
)
from ..training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
    StorageStats,
)
from ..utils.thresholding import tune_threshold

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageConfig:
    stage: Stage
    study_name: str
    sampler_seed: int
    data_fraction: float
    max_epochs: int
    max_steps: Optional[int]
    max_seq_length_cap: int
    description: str


STAGE1_CONFIG = StageConfig(
    stage=Stage.STAGE1,
    study_name="stage1_broad",
    sampler_seed=1337,
    data_fraction=0.5,
    max_epochs=2,
    max_steps=2048,
    max_seq_length_cap=384,
    description="Broad search on subset with aggressive pruning.",
)

STAGE2_CONFIG = StageConfig(
    stage=Stage.STAGE2,
    study_name="stage2_exploit",
    sampler_seed=4242,
    data_fraction=1.0,
    max_epochs=8,
    max_steps=6144,
    max_seq_length_cap=512,
    description="Narrow search on full data with gentler pruning.",
)

EXPERIMENTS_DIR = Path("experiments")
TOP_K_STAGE1 = 50


def run_stage1(
    storage: str,
    n_trials: int,
    n_jobs: int = 1,
    *,
    experiments_dir: str | Path = EXPERIMENTS_DIR,
    study_name: Optional[str] = None,
) -> Study:
    """Execute Stage-1 hyperparameter search."""

    experiments_root = Path(experiments_dir)
    experiments_root.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(
        multivariate=True,
        group=True,
        consider_prior=True,
        seed=STAGE1_CONFIG.sampler_seed,
    )
    pruner = SuccessiveHalvingPruner(
        min_resource=400,
        reduction_factor=3,
        min_early_stopping_rate=1,
    )
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or STAGE1_CONFIG.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    LOGGER.info("Starting Stage-1 study '%s' (%s)", study.study_name, STAGE1_CONFIG.description)
    objective = objective_factory(
        config=STAGE1_CONFIG,
        experiments_dir=experiments_root,
        narrowing=None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)
    return study


def run_stage2(
    storage: str,
    stage1_study: str | Study,
    n_trials: int,
    n_jobs: int = 1,
    *,
    experiments_dir: str | Path = EXPERIMENTS_DIR,
    study_name: Optional[str] = None,
    top_k: int = TOP_K_STAGE1,
) -> Study:
    """Execute Stage-2 hyperparameter search using Stage-1 results for narrowing."""

    experiments_root = Path(experiments_dir)
    experiments_root.mkdir(parents=True, exist_ok=True)

    if isinstance(stage1_study, Study):
        base_study = stage1_study
    else:
        base_study = optuna.load_study(
            study_name=stage1_study,
            storage=storage,
        )

    LOGGER.info(
        "Loaded Stage-1 study '%s' with %d trials. Narrowing Stage-2 search space.",
        base_study.study_name,
        len(base_study.trials),
    )
    narrowing = narrow_stage2_space(
        base_study.get_trials(deepcopy=False),
        top_k=top_k,
    )

    sampler = TPESampler(
        multivariate=True,
        group=True,
        consider_prior=True,
        seed=STAGE2_CONFIG.sampler_seed,
    )
    pruner = MedianPruner(n_startup_trials=15)

    stage2_study = optuna.create_study(
        direction="maximize",
        study_name=study_name or STAGE2_CONFIG.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    LOGGER.info("Starting Stage-2 study '%s' (%s)", stage2_study.study_name, STAGE2_CONFIG.description)
    objective = objective_factory(
        config=STAGE2_CONFIG,
        experiments_dir=experiments_root,
        narrowing=narrowing,
    )
    stage2_study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)
    return stage2_study


@dataclass(frozen=True)
class ObjectiveContext:
    config: StageConfig
    experiments_dir: Path
    narrowing: Optional[NarrowedSpace]
    hardware: HardwareCapabilities = field(default_factory=detect_hardware_capabilities)


def objective_factory(
    config: StageConfig,
    *,
    experiments_dir: Path,
    narrowing: Optional[NarrowedSpace],
) -> Callable[[Trial], float]:
    context = ObjectiveContext(config=config, experiments_dir=experiments_dir, narrowing=narrowing)

    def _objective(trial: Trial) -> float:
        start_time = time.monotonic()
        params = sample_parameters(
            trial,
            config.stage,
            narrowing=context.narrowing,
            hardware=context.hardware,
        )
        params["max_seq_length"] = min(params["max_seq_length"], config.max_seq_length_cap)
        effective_lr = params["learning_rate"] * (params["effective_batch_size"] / 32.0)
        trial.set_user_attr("stage", config.stage.value)
        trial.set_user_attr("effective_learning_rate", effective_lr)

        try:
            outcome = _run_simulated_training(trial, params, context)
        except FloatingPointError as exc:
            trial.set_user_attr("failure", "nan")
            LOGGER.warning("Trial %s encountered NaN; pruning.", trial.number)
            raise TrialPruned("NaN encountered during training.") from exc
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "oom" in message:
                trial.set_user_attr("failure", "oom")
                LOGGER.warning("Trial %s ran out of memory; pruning.", trial.number)
                raise TrialPruned("Out of memory.") from exc
            raise

        duration = time.monotonic() - start_time
        trial.set_user_attr("duration_seconds", duration)
        trial.set_user_attr("macro_f1", outcome.score)
        trial.set_user_attr("trial_id", outcome.trial_id)
        trial.set_user_attr("evaluation_report", str(outcome.report_path))
        trial.set_user_attr("checkpoint_path", str(outcome.checkpoint_path))

        LOGGER.info(
            "Trial %s (%s) -> macro_f1=%.4f, eff_lr=%.6f",
            outcome.trial_id,
            config.stage.value,
            outcome.score,
            effective_lr,
        )
        return outcome.score

    return _objective


@dataclass
class SimulationOutcome:
    trial_id: str
    score: float
    report_path: Path
    checkpoint_path: Path


@dataclass
class _SimulationPayload:
    logits: np.ndarray
    labels: np.ndarray
    noise_scale: float
    scaling: float


def _run_simulated_training(
    trial: Trial,
    params: Dict[str, float | int | bool | str | None],
    context: ObjectiveContext,
) -> SimulationOutcome:
    seed = 1337 + trial.number
    payload = _simulate_payload(params, context, seed)
    per_class = bool(
        context.config.stage is Stage.STAGE2 and params.get("decision_thresholds_per_class", False)
    )
    macro_f1, thresholds = tune_threshold(payload.logits, payload.labels, per_class=per_class)
    epoch_metrics = _build_epoch_metrics(macro_f1, context.config.max_epochs)
    for idx, metric_value in enumerate(epoch_metrics, start=1):
        trial.report(metric_value, step=idx)
        if trial.should_prune():
            trial.set_user_attr("pruned_epoch", idx)
            raise TrialPruned(f"Pruned at epoch {idx}")

    outcome = _persist_simulation_artifacts(
        params=params,
        context=context,
        payload=payload,
        seed=seed,
        macro_f1=macro_f1,
        thresholds=thresholds,
        per_class=per_class,
        epoch_metrics=epoch_metrics,
    )
    return outcome


def _simulate_payload(
    params: Dict[str, float | int | bool | str | None],
    context: ObjectiveContext,
    seed: int,
) -> _SimulationPayload:
    rng = np.random.default_rng(seed)
    batch_size = int(params["batch_size"])
    max_seq_length = int(params["max_seq_length"])
    max_tokens = batch_size * max_seq_length
    token_budget = 64000 if context.config.stage is Stage.STAGE1 else 130000
    if max_tokens > token_budget:
        raise RuntimeError("CUDA out of memory: token budget exceeded.")

    if context.config.stage is Stage.STAGE2 and params["learning_rate"] > 2e-3:
        raise FloatingPointError("Unstable learning detected (simulated NaN).")

    data_fraction = context.config.data_fraction
    base_samples = 1200 if context.config.stage is Stage.STAGE1 else 2200
    num_samples = max(200, int(base_samples * data_fraction))
    num_labels = 10
    feature_dim = min(256, max(32, max_seq_length // 2))
    features = rng.normal(size=(num_samples, feature_dim)).astype("float32")
    weights = rng.normal(size=(feature_dim, num_labels)).astype("float32")
    ground_truth_logits = features @ weights / math.sqrt(feature_dim)
    base_probs = 1.0 / (1.0 + np.exp(-ground_truth_logits))
    labels = (base_probs > 0.5).astype("int8")

    noise_scale = _estimate_noise(params, context.config.stage)
    scaling = _estimate_scaling(params)
    simulated_logits = scaling * ground_truth_logits + rng.normal(scale=noise_scale, size=ground_truth_logits.shape)

    return _SimulationPayload(
        logits=simulated_logits,
        labels=labels,
        noise_scale=float(noise_scale),
        scaling=float(scaling),
    )


def _persist_simulation_artifacts(
    params: Dict[str, float | int | bool | str | None],
    context: ObjectiveContext,
    payload: _SimulationPayload,
    *,
    seed: int,
    macro_f1: float,
    thresholds: float | Sequence[float],
    per_class: bool,
    epoch_metrics: Sequence[float],
    trial_id: str | None = None,
) -> SimulationOutcome:
    experiments_root = context.experiments_dir
    experiments_root.mkdir(parents=True, exist_ok=True)
    resolved_trial_id = trial_id or f"trial_{uuid.uuid4().hex[:8]}"
    trial_dir = experiments_root / resolved_trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    report_path = trial_dir / "evaluation_report.json"
    checkpoint_dir = trial_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_manager = CheckpointManager(
        trial_dir=checkpoint_dir,
        policy=CheckpointRetentionPolicy(keep_last_n=1, keep_best_k=1),
        compatibility=CheckpointMetadata(
            code_version="sim-hpo-0.1",
            model_signature="synthetic-linear",
            head_configuration="simulation-head",
        ),
    )
    storage_stats = StorageStats(available_bytes=10 * 1024**3, total_bytes=20 * 1024**3)
    checkpoint_state = {
        "seed": seed,
        "weights": np.mean(payload.logits, axis=0),
        "thresholds": thresholds,
        "macro_f1": macro_f1,
    }
    record = checkpoint_manager.save_checkpoint(
        state=checkpoint_state,
        epoch=context.config.max_epochs,
        metric_value=macro_f1,
        extra_metadata={"stage": context.config.stage.value},
        storage_stats=storage_stats,
    )

    effective_lr = params["learning_rate"] * (params["effective_batch_size"] / 32.0)
    threshold_payload = _serialize_thresholds(thresholds)

    report_payload = {
        "trial_id": resolved_trial_id,
        "stage": context.config.stage.value,
        "status": "completed",
        "seed": seed,
        "description": context.config.description,
        "metrics": {
            "validation_macro_f1": macro_f1,
            "epoch_metrics": epoch_metrics,
        },
        "thresholding": {
            "mode": "per_class" if per_class else "global",
            "thresholds": threshold_payload,
        },
        "hyperparameters": params,
        "derived": {
            "effective_batch_size": params["effective_batch_size"],
            "effective_learning_rate": effective_lr,
            "noise_scale": payload.noise_scale,
            "scaling_factor": payload.scaling,
        },
        "runtime": {
            "data_fraction": context.config.data_fraction,
            "max_epochs": context.config.max_epochs,
            "max_steps": context.config.max_steps,
        },
        "artifacts": {
            "evaluation_report": str(report_path),
            "checkpoint_best": str(record.path),
        },
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, indent=2, sort_keys=True)

    return SimulationOutcome(
        trial_id=resolved_trial_id,
        score=float(macro_f1),
        report_path=report_path,
        checkpoint_path=record.path,
    )


def simulate_configuration(
    params: Dict[str, float | int | bool | str | None],
    *,
    config: StageConfig = STAGE2_CONFIG,
    experiments_dir: Path | str = EXPERIMENTS_DIR,
    seed: int = 1337,
    trial_id: str | None = None,
    per_class: bool | None = None,
) -> SimulationOutcome:
    """Simulate training for a fixed configuration (used for exports)."""

    prepared_params = dict(params)
    if "batch_size" in prepared_params and "gradient_accumulation_steps" in prepared_params:
        prepared_params.setdefault(
            "effective_batch_size",
            int(prepared_params["batch_size"]) * int(prepared_params["gradient_accumulation_steps"]),
        )

    context = ObjectiveContext(
        config=config,
        experiments_dir=Path(experiments_dir),
        narrowing=None,
    )
    payload = _simulate_payload(prepared_params, context, seed)
    per_class_mode = (
        per_class
        if per_class is not None
        else bool(config.stage is Stage.STAGE2 and prepared_params.get("decision_thresholds_per_class", False))
    )
    macro_f1, thresholds = tune_threshold(payload.logits, payload.labels, per_class=per_class_mode)
    epoch_metrics = _build_epoch_metrics(macro_f1, config.max_epochs)
    return _persist_simulation_artifacts(
        params=prepared_params,
        context=context,
        payload=payload,
        seed=seed,
        macro_f1=macro_f1,
        thresholds=thresholds,
        per_class=per_class_mode,
        epoch_metrics=epoch_metrics,
        trial_id=trial_id,
    )


def _serialize_thresholds(thresholds: object) -> float | Sequence[float]:
    if isinstance(thresholds, np.ndarray):
        return [float(x) for x in thresholds.tolist()]
    if isinstance(thresholds, (list, tuple)):
        return [float(x) for x in thresholds]
    return float(thresholds)  # type: ignore[return-value]


def _estimate_scaling(params: Dict[str, float | int | bool | str | None]) -> float:
    base = 1.0
    effective_batch = float(params["effective_batch_size"])
    base += 0.05 * math.log2(effective_batch / 32.0)
    base -= float(params["dropout"]) * 0.1
    base -= float(params["augmentation_prob"]) * 0.05
    if params["optimizer"] == "adamw":
        base += 0.02
    elif params["optimizer"] == "sgd":
        base -= 0.03
    if params["pooling_strategy"] == "mean":
        base += 0.01
    return float(np.clip(base, 0.6, 1.4))


def _estimate_noise(params: Dict[str, float | int | bool | str | None], stage: Stage) -> float:
    base = 0.55 if stage is Stage.STAGE1 else 0.45
    lr = float(params["learning_rate"])
    lr_opt = 3e-4
    base += abs(math.log10(lr) - math.log10(lr_opt)) * 0.08
    base += math.sqrt(float(params["weight_decay"])) * 0.1
    base += float(params["dropout"]) * 0.12
    base -= float(params["augmentation_prob"]) * 0.08
    base -= (float(params["layerwise_lr_decay"]) - 0.9) * 0.1
    if params["loss_type"] in ("adaptive_focal", "hybrid_bce_adaptive_focal", "hybrid_weighted_bce_adaptive_focal"):
        base -= 0.03
    if params["gradient_checkpointing"]:
        base += 0.02
    if params.get("decision_thresholds_per_class", False):
        base -= 0.02
    return float(np.clip(base, 0.2, 0.9))


def _build_epoch_metrics(final_metric: float, max_epochs: int) -> Sequence[float]:
    start_metric = max(0.05, final_metric - 0.12)
    if max_epochs <= 1:
        return [final_metric]
    return list(np.linspace(start_metric, final_metric * 0.99, num=max_epochs, dtype=float))


__all__ = [
    "run_stage1",
    "run_stage2",
    "objective_factory",
    "StageConfig",
    "STAGE1_CONFIG",
    "STAGE2_CONFIG",
    "SimulationOutcome",
    "simulate_configuration",
]
