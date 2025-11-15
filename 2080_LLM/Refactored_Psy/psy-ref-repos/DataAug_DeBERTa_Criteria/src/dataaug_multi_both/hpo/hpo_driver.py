from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import optuna
import yaml
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_slice

from ..training.runner import StageBManager, TrainerSettings, merge_defaults, trainer_entrypoint
from .samplers_pruners import PlateauStopper, build_pruner, build_sampler
from .search_space import stage_b_space_from_winner
from .trial_executor import DatasetConfig, HardwareConfig, objective_factory, run_study

TOTAL_TRIALS = 500
GLOBAL_TIMEOUT_SECONDS = 604_800


@dataclass(slots=True)
class StageBudget:
    trials: int
    epochs: int
    timeout_seconds: int


@dataclass(slots=True)
class HpoRunConfig:
    experiments_dir: Path
    experiment_name: str
    study_name: str
    storage_url: str
    dataset: DatasetConfig
    hardware: HardwareConfig
    deterministic: bool
    base_seed: int
    stage_a: StageBudget
    stage_b: StageBudget
    total_timeout: int = GLOBAL_TIMEOUT_SECONDS
    top_k: int = 8
    model_name: str = "microsoft/deberta-v3-base"


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True)


def _render_plots(study: optuna.Study, out_dir: Path) -> None:
    if len(study.trials) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_param_importances(study).write_html(str(out_dir / "param_importances.html"))
        plot_parallel_coordinate(study).write_html(str(out_dir / "parallel_coordinate.html"))
        plot_slice(study).write_html(str(out_dir / "slice.html"))
    except Exception:  # pragma: no cover - plotting is best-effort
        pass


def _mirror_directory(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _summarize_stage(stage_dir: Path, stage_name: str, study: optuna.Study) -> optuna.FrozenTrial:
    best_trial = study.best_trial
    if best_trial is None:
        raise RuntimeError(f"No completed trials for {stage_name}; unable to continue.")

    _save_json(stage_dir / "best_params.json", {"params": best_trial.params})
    _save_text(stage_dir / "best_value.txt", f"{best_trial.value:.6f}\n")
    _save_yaml(
        stage_dir / "best_config.yaml",
        {
            "stage": stage_name,
            "metric": float(best_trial.value),
            "threshold": best_trial.user_attrs.get("threshold"),
            "params": best_trial.params,
        },
    )
    _render_plots(study, stage_dir / "plots")
    return best_trial


def run_two_stage_hpo(config: HpoRunConfig) -> optuna.FrozenTrial:
    if config.stage_a.trials + config.stage_b.trials != TOTAL_TRIALS:
        raise ValueError("Stage trials must sum to 500.")
    if config.total_timeout > GLOBAL_TIMEOUT_SECONDS:
        raise ValueError("Total timeout cannot exceed 7 days (604800 seconds).")

    experiments_root = config.experiments_dir / config.experiment_name
    experiments_root.mkdir(parents=True, exist_ok=True)
    artifacts_root = config.experiments_dir / "artifacts" / "hpo" / config.study_name

    hardware = replace(config.hardware)
    hardware.validate()

    trainer_settings = TrainerSettings(
        dataset_cfg=config.dataset,
        hardware_cfg=hardware,
        experiments_dir=str(config.experiments_dir),
        experiment_name=config.experiment_name,
        model_name=config.model_name,
        deterministic=config.deterministic,
    )

    global_start = time.time()

    stage_a_callbacks = [PlateauStopper(120)]
    stage_a_objective = objective_factory(
        stage="A",
        max_epochs=config.stage_a.epochs,
        trainer_fn=trainer_entrypoint,
        experiments_dir=str(config.experiments_dir),
        base_seed=config.base_seed,
        trainer_kwargs={"settings": trainer_settings},
    )

    stage_a_timeout = min(config.stage_a.timeout_seconds, config.total_timeout)
    stage_a_study = run_study(
        stage="A",
        study_name=f"{config.study_name}_stage_A",
        storage_url=config.storage_url,
        objective=stage_a_objective,
        n_trials=config.stage_a.trials,
        timeout_seconds=stage_a_timeout,
        max_epochs=config.stage_a.epochs,
        callbacks=stage_a_callbacks,
    )

    stage_a_artifacts = artifacts_root / "stage_A"
    best_a = _summarize_stage(stage_a_artifacts, "stage_A", stage_a_study)

    completed_trials = [t for t in stage_a_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Stage-A completed no successful trials; aborting Stage-B.")

    top_k = max(1, config.top_k)
    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_k]
    spaces = [stage_b_space_from_winner(t.params) for t in top_trials]
    candidates = [merge_defaults(t.params) for t in top_trials]
    stage_b_manager = StageBManager(spaces=spaces, candidates=candidates)

    elapsed = time.time() - global_start
    remaining = max(1, config.total_timeout - int(elapsed))

    stage_b_callbacks = [PlateauStopper(120)]
    stage_b_objective = objective_factory(
        stage="B",
        max_epochs=config.stage_b.epochs,
        trainer_fn=trainer_entrypoint,
        experiments_dir=str(config.experiments_dir),
        base_seed=config.base_seed,
        trainer_kwargs={"settings": trainer_settings, "stage_b_manager": stage_b_manager},
    )

    study_b_name = f"{config.study_name}_stage_B"
    sampler = build_sampler("B")
    pruner = build_pruner("B", config.stage_b.epochs)
    stage_b_study = optuna.create_study(
        study_name=study_b_name,
        storage=config.storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if len(stage_b_study.trials) == 0:
        for index, trial in enumerate(top_trials):
            stage_b_study.enqueue_trial(params=dict(trial.params), user_attrs={"space_index": index})

    stage_b_start = time.time()

    def _stage_b_console_progress_callback(study: optuna.Study, frozen_trial: optuna.FrozenTrial) -> None:
        try:
            completed = sum(t.state != optuna.trial.TrialState.RUNNING for t in study.trials)
            total = max(config.stage_b.trials, completed)
            best_val = float(study.best_value) if study.best_value is not None else float("nan")
        except Exception:
            completed = len(study.trials)
            total = max(config.stage_b.trials, completed)
            best_val = float("nan")

        elapsed = time.time() - stage_b_start
        rate = completed / max(1, total)
        eta = (elapsed / rate - elapsed) if rate > 0 else 0.0
        state_name = str(frozen_trial.state).split(".")[-1]
        val_str = "NA" if frozen_trial.value is None else f"{float(frozen_trial.value):.4f}"
        print(
            f"[HPO B] Trial {frozen_trial.number} {state_name} value={val_str} "
            f"| Completed {completed}/{total} | Best={best_val:.4f} | Elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
        )

    stage_b_study.optimize(
        stage_b_objective,
        n_trials=config.stage_b.trials,
        timeout=min(config.stage_b.timeout_seconds, remaining),
        callbacks=stage_b_callbacks + [_stage_b_console_progress_callback],
        n_jobs=1,
        gc_after_trial=True,
    )

    stage_b_artifacts = artifacts_root / "stage_B"
    best_b = _summarize_stage(stage_b_artifacts, "stage_B", stage_b_study)

    final_payload = {
        "stage_a": {
            "trial": best_a.number,
            "metric": float(best_a.value),
            "threshold": best_a.user_attrs.get("threshold"),
        },
        "stage_b": {
            "trial": best_b.number,
            "metric": float(best_b.value),
            "threshold": best_b.user_attrs.get("threshold"),
        },
        "dataset": {
            "id": config.dataset.dataset_id,
            "revision": config.dataset.revision,
        },
        "hardware": {
            "max_length": hardware.max_length,
            "per_device_batch_size": hardware.per_device_batch_size,
            "grad_accumulation_steps": hardware.grad_accumulation_steps,
        },
        "deterministic": config.deterministic,
        "total_trials": config.stage_a.trials + config.stage_b.trials,
        "params": best_b.params,
    }

    _save_json(artifacts_root / "best_params.json", {"params": best_b.params})
    _save_text(artifacts_root / "best_value.txt", f"{best_b.value:.6f}\n")
    _save_yaml(artifacts_root / "best_config.yaml", final_payload)
    _render_plots(stage_b_study, artifacts_root / "final_plots")
    _mirror_directory(artifacts_root, experiments_root)

    return best_b


__all__ = ["StageBudget", "HpoRunConfig", "run_two_stage_hpo"]
