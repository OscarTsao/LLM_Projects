from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import hydra
import optuna
from hydra import utils as hydra_utils
from omegaconf import DictConfig, OmegaConf

from .config_utils import resolve_config_paths
from .train import run_training

LOGGER = logging.getLogger(__name__)


def _sample_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    param_type = spec["type"]
    if param_type == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False)
        )
    if param_type == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if param_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported parameter type: {param_type}")


def _assign_target(cfg: DictConfig, target: str, value: Any) -> None:
    keys = target.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def run_hpo(cfg: DictConfig) -> optuna.Study:
    base_output = Path(cfg.training.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler() if cfg.hpo.sampler == "tpe" else None
    pruner = optuna.pruners.MedianPruner() if cfg.hpo.pruner.type == "median" else None

    study = optuna.create_study(
        study_name=cfg.hpo.study_name,
        direction=cfg.hpo.direction,
        storage=cfg.hpo.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for param_name, spec in cfg.hpo.params.items():
            target_path = spec["target"]
            sampled_value = _sample_param(trial, param_name, spec)
            _assign_target(trial_cfg, target_path, sampled_value)

        trial_output_dir = base_output / f"hpo_trial_{trial.number}"
        trial_cfg.training.output_dir = str(trial_output_dir.resolve())
        trial_cfg.logging.tags["trial_number"] = str(trial.number)

        metrics = run_training(trial_cfg)
        metric_key = f"eval_{cfg.hpo.metric_name}"
        score = metrics.get(metric_key)
        if score is None:
            raise ValueError(f"Metric '{metric_key}' not found in evaluation metrics: {metrics}")
        trial.set_user_attr("metrics", metrics)
        return float(score)

    study.optimize(
        objective,
        n_trials=cfg.hpo.n_trials,
        timeout=cfg.hpo.timeout,
        gc_after_trial=True,
    )
    return study


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = Path(hydra_utils.get_original_cwd())
    cfg.paths.project_root = str(project_root)
    cfg = resolve_config_paths(cfg, base_dir=project_root)

    study = run_hpo(cfg)
    LOGGER.info("Best trial: %s", study.best_trial)
    LOGGER.info("Best value: %s", study.best_value)
    LOGGER.info("Best params: %s", study.best_trial.params)


if __name__ == "__main__":
    main()
