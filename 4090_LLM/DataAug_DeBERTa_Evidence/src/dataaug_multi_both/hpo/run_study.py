from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import optuna

from dataaug_multi_both.config import load_project_config
from dataaug_multi_both.hpo.space import (
    build_stage_b_narrowing,
    define_search_space,
    extract_structural_params,
)
from dataaug_multi_both.training.train_loop import run_training_job

logger = logging.getLogger(__name__)


def _compute_trial_seed(study_name: str, trial_number: int, base_seed: int) -> int:
    seed = (base_seed * 9973) + (trial_number + 1) * 37
    for char in study_name:
        seed = (seed * 31 + ord(char)) % (2**32)
    return seed % (2**32)


@dataclass
class StageContext:
    label: str
    total_trials: int
    start_time: float = field(default_factory=time.time)
    durations: list[float] = field(default_factory=list)
    completed: int = 0
    best_value: Optional[float] = None
    best_trial_number: Optional[int] = None

    def record_duration(self, duration: float) -> None:
        self.durations.append(duration)
        self.completed += 1

    def maybe_update_best(self, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        value = trial.value
        if value is None:
            return
        if self.best_value is None or value > self.best_value + 1e-12:
            self.best_value = value
            self.best_trial_number = trial.number

    def eta_seconds(self) -> Optional[float]:
        if not self.durations:
            return None
        avg = sum(self.durations) / len(self.durations)
        remaining = max(0, self.total_trials - self.completed)
        return avg * remaining


def build_sampler(stage_cfg: Mapping[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_type = stage_cfg.get("sampler", {}).get("type", "tpe")
    sampler_params = stage_cfg.get("sampler", {}).get("params", {})
    seed = stage_cfg.get("sampler", {}).get("seed", None)
    if sampler_type == "tpe":
        return optuna.samplers.TPESampler(seed=seed, **sampler_params)
    raise ValueError(f"Unsupported sampler type: {sampler_type}")


def build_pruner(stage_cfg: Mapping[str, Any]) -> optuna.pruners.BasePruner:
    pruner_cfg = stage_cfg.get("pruner", {})
    pruner_type = pruner_cfg.get("type", "median")
    if pruner_type == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=int(pruner_cfg.get("min_resource", 1)),
            max_resource=int(pruner_cfg.get("max_resource", 1)),
            reduction_factor=int(pruner_cfg.get("reduction_factor", 3)),
        )
    if pruner_type == "percentile":
        return optuna.pruners.PercentilePruner(
            percentile=float(pruner_cfg.get("percentile", 25.0)),
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 2)),
        )
    if pruner_type == "median":
        return optuna.pruners.MedianPruner()
    raise ValueError(f"Unsupported pruner type: {pruner_type}")


def run_optuna_stage(
    base_cfg: Mapping[str, Any] | None,
    *,
    study_name: str,
    storage: str,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
    n_trials: int,
    timeout: Optional[int],
    stage_label: str,
    stage_epochs: int,
    global_seed: int,
    objective_metric: str,
    frozen_params: Optional[Mapping[str, Any]] = None,
    narrowed_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    structure_pool: Optional[Sequence[tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]]] = None,
    plateau_patience: Optional[int] = None,
    study_direction: str = "maximize",
    enqueue_params: Optional[Iterable[Mapping[str, Any]]] = None,
) -> optuna.study.Study:
    cfg = base_cfg or load_project_config()
    logger.info(
        "Starting Optuna stage '%s' (%d trials, timeout=%s)",
        stage_label,
        n_trials,
        timeout,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=study_direction,
        sampler=sampler,
        pruner=pruner,
    )

    stage_context = StageContext(label=stage_label, total_trials=n_trials)

    for params in enqueue_params or []:
        sanitized = dict(params)
        structure_idx = sanitized.get("structure.id") if structure_pool else None
        if structure_pool is not None and structure_idx is not None and structure_idx < len(structure_pool):
            struct_frozen, struct_narrowed = structure_pool[structure_idx]
            sanitized.update(struct_frozen)
            for key, bounds in struct_narrowed.items():
                if key not in sanitized:
                    continue
                low = bounds.get("low")
                high = bounds.get("high")
                if low is None or high is None:
                    continue
                if isinstance(low, float) or isinstance(high, float):
                    sanitized[key] = float(min(max(float(sanitized[key]), float(low)), float(high)))
                else:
                    sanitized[key] = int(min(max(int(sanitized[key]), int(low)), int(high)))
        study.enqueue_trial(sanitized)

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
        trial_seed = _compute_trial_seed(study.study_name, trial.number, global_seed)

        combined_frozen: Dict[str, Any] = dict(frozen_params or {})
        combined_narrowed: Dict[str, Dict[str, Any]] = {
            **{k: dict(v) for k, v in (narrowed_params or {}).items()}
        }

        if structure_pool:
            structure_choices = list(range(len(structure_pool)))
            structure_idx = trial.suggest_categorical("structure.id", structure_choices)
            struct_frozen, struct_narrowed = structure_pool[structure_idx]
            combined_frozen.update(struct_frozen)
            for key, value in struct_narrowed.items():
                combined_narrowed[key] = dict(value)
        else:
            structure_idx = None

        telemetry = {
            "stage": stage_label,
            "trial_index": trial.number + 1,
            "trial_total": n_trials,
            "completion_rate": (trial.number + 1) / max(1, n_trials),
            "elapsed_seconds": time.time() - stage_context.start_time,
            "eta_seconds": stage_context.eta_seconds() or -1.0,
            "best_metric": stage_context.best_value if stage_context.best_value is not None else float("nan"),
            "best_trial_number": stage_context.best_trial_number if stage_context.best_trial_number is not None else -1,
        }
        trial.set_user_attr("telemetry", telemetry)
        if structure_idx is not None:
            trial.set_user_attr("structure_index", structure_idx)

        trial_cfg = define_search_space(
            trial,
            base_config=cfg,
            frozen=combined_frozen,
            narrowed=combined_narrowed,
        )
        trial_cfg["seed"] = trial_seed
        trial_cfg["train"]["num_epochs"] = stage_epochs
        trial_cfg["objective"]["primary_metric"] = objective_metric

        trial.set_user_attr("config", trial_cfg)

        try:
            result = run_training_job(trial_cfg, trial=trial)
        except optuna.TrialPruned:
            raise
        finally:
            duration = time.time() - trial_start
            trial.set_user_attr("duration_seconds", duration)
            stage_context.record_duration(duration)

        trial.set_user_attr("metrics", result["metrics"])
        trial.set_user_attr("evaluation_report_path", result.get("evaluation_report_path", ""))
        trial.set_user_attr("checkpoint_path", result.get("checkpoint_path", ""))
        trial.set_user_attr("pruned_at_epoch", result.get("pruned_at_epoch", 0))

        return float(result["objective"])

    callbacks: list[optuna.study.StudyDirection] = []

    if plateau_patience is not None:
        patience_counter = {"count": 0, "best_value": None}

        def plateau_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            stage_context.maybe_update_best(trial)
            best_value = study.best_value if study.best_trial else None
            if best_value is None:
                return
            if patience_counter["best_value"] is None or best_value > patience_counter["best_value"] + 1e-12:
                patience_counter["best_value"] = best_value
                patience_counter["count"] = 0
                return
            patience_counter["count"] += 1
            if patience_counter["count"] >= plateau_patience:
                logger.info(
                    "Plateau detected after %d trials. Stopping stage '%s'.",
                    plateau_patience,
                    stage_label,
                )
                study.stop()

        callbacks.append(plateau_callback)
    else:
        callbacks.append(lambda study, trial: stage_context.maybe_update_best(trial))

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
    )

    logger.info(
        "Completed Optuna stage '%s' (best=%.5f, trials=%d)",
        stage_label,
        study.best_value if study.best_trial else float("nan"),
        len(study.trials),
    )
    return study


def prepare_stage_b_controls(trial_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    frozen = extract_structural_params(trial_cfg)
    narrowed = build_stage_b_narrowing(trial_cfg)
    return frozen, narrowed
