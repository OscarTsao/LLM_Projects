from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from .train import run_training
from .utils import ensure_dir, load_yaml, merge_dicts, save_json, save_yaml, setup_logger

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None

try:
    from ray import tune
except ImportError:  # pragma: no cover
    tune = None


def load_search_space(path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    return cfg


def sample_spec_optuna(trial, name: str, spec: Dict[str, Any]) -> Any:
    stype = spec["type"]
    if stype == "categorical":
        return trial.suggest_categorical(name, spec["values"])
    if stype == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"))
    if stype == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if stype == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    raise ValueError(f"Unsupported spec type '{stype}' for {name}")


def apply_conditions(params: Dict[str, Any], conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not conditions:
        return params
    filtered = dict(params)
    for cond in conditions:
        when = cond.get("when", {})
        include = cond.get("include", [])
        exclude = cond.get("exclude", [])
        if all(filtered.get(key) == value for key, value in when.items()):
            for key in exclude:
                filtered.pop(key, None)
            for key in include:
                if key not in filtered and key in params:
                    filtered[key] = params[key]
    return filtered


def build_trial_config(base_cfg: Dict[str, Any], sampled: Dict[str, Any], conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
    filtered = apply_conditions(sampled, conditions)
    cfg = merge_dicts(base_cfg, filtered)
    cfg["use_wandb"] = False
    return cfg


def run_optuna(args: argparse.Namespace, base_cfg: Dict[str, Any], search_cfg: Dict[str, Any]) -> None:
    if optuna is None:
        raise ImportError("optuna is required but not installed")

    logger = setup_logger()
    out_dir = ensure_dir(args.out_dir)
    space = search_cfg.get("search_space", {})
    conditions = search_cfg.get("conditions", [])

    def objective(trial: "optuna.Trial") -> float:
        sampled = {name: sample_spec_optuna(trial, name, spec) for name, spec in space.items()}
        cfg = build_trial_config(base_cfg, sampled, conditions)

        trial_dir = out_dir / f"trial_{trial.number}"
        run_args = SimpleNamespace(
            labels=args.labels,
            data_dir=args.data_dir,
            hf_id=args.hf_id,
            hf_config=args.hf_config,
            out_dir=str(trial_dir),
            use_wandb=False,
        )
        ensure_dir(trial_dir)
        try:
            result = run_training(cfg, run_args)
            metric = result["dev_metrics"].get("macro_f1", float("nan"))
            if math.isnan(metric):
                raise optuna.TrialPruned("Macro-F1 is NaN")
            trial.set_user_attr("metrics", result)
            return metric
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Trial %s failed: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_trial = study.best_trial
    best_cfg = build_trial_config(base_cfg, best_trial.params, conditions)
    save_yaml(best_cfg, out_dir / "best_config.yaml")
    save_json(best_trial.user_attrs.get("metrics", {}), out_dir / "best_metrics.json")
    logger.info("Best macro-F1=%.4f", best_trial.value)


def build_ray_search_space(space: Dict[str, Any]) -> Dict[str, Any]:
    if tune is None:
        raise ImportError("ray[tune] is required for Ray backend")
    ray_space = {}
    for name, spec in space.items():
        stype = spec["type"]
        if stype == "categorical":
            ray_space[name] = tune.choice(spec["values"])
        elif stype == "uniform":
            ray_space[name] = tune.uniform(spec["low"], spec["high"])
        elif stype == "loguniform":
            ray_space[name] = tune.loguniform(spec["low"], spec["high"])
        elif stype == "int":
            ray_space[name] = tune.randint(spec["low"], spec["high"] + 1)
        else:
            raise ValueError(f"Unsupported spec type '{stype}' for Ray")
    return ray_space


def run_ray(args: argparse.Namespace, base_cfg: Dict[str, Any], search_cfg: Dict[str, Any]) -> None:
    if tune is None:
        raise ImportError("ray[tune] is required but not installed")

    logger = setup_logger()
    out_dir = ensure_dir(args.out_dir)
    space = search_cfg.get("search_space", {})
    conditions = search_cfg.get("conditions", [])
    ray_space = build_ray_search_space(space)

    def ray_objective(config: Dict[str, Any]):
        cfg = build_trial_config(base_cfg, config, conditions)
        trial_id = tune.get_trial_id()
        trial_dir = out_dir / f"trial_{trial_id}"
        run_args = SimpleNamespace(
            labels=args.labels,
            data_dir=args.data_dir,
            hf_id=args.hf_id,
            hf_config=args.hf_config,
            out_dir=str(trial_dir),
            use_wandb=False,
        )
        ensure_dir(trial_dir)
        try:
            result = run_training(cfg, run_args)
            metric = result["dev_metrics"].get("macro_f1", 0.0)
        except Exception as exc:  # pragma: no cover
            logger.warning("Ray trial failed: %s", exc)
            metric = 0.0
        tune.report(macro_f1=metric)

    analysis = tune.run(
        ray_objective,
        config=ray_space,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": args.ray_cpus, "gpu": args.ray_gpus},
        local_dir=str(out_dir),
    )
    best_config = analysis.get_best_config(metric="macro_f1", mode="max")
    best_cfg = build_trial_config(base_cfg, best_config, conditions)
    save_yaml(best_cfg, out_dir / "best_config.yaml")
    best_trial = analysis.get_best_trial(metric="macro_f1", mode="max", scope="all")
    if best_trial and hasattr(best_trial, "last_result"):
        save_json(best_trial.last_result, out_dir / "best_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ReDSM5")
    parser.add_argument("--backend", choices=["optuna", "ray"], required=True)
    parser.add_argument("--config", required=True, help="Base YAML config path")
    parser.add_argument("--search_space", required=True, help="Search space YAML path")
    parser.add_argument("--labels", required=True, help="Labels YAML path")
    parser.add_argument("--out_dir", required=True, help="Output directory for HPO runs")
    parser.add_argument("--data_dir", default="", help="Local dataset directory")
    parser.add_argument("--hf_id", default="", help="Hugging Face dataset identifier")
    parser.add_argument("--hf_config", default="", help="Hugging Face dataset config")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--ray_cpus", type=float, default=4)
    parser.add_argument("--ray_gpus", type=float, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    overrides = {}
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        overrides["max_eval_samples"] = args.max_eval_samples
    if args.max_test_samples is not None:
        overrides["max_test_samples"] = args.max_test_samples
    if overrides:
        base_cfg = merge_dicts(base_cfg, overrides)
    search_cfg = load_search_space(args.search_space)

    if args.backend == "optuna":
        run_optuna(args, base_cfg, search_cfg)
    else:
        run_ray(args, base_cfg, search_cfg)


if __name__ == "__main__":
    main()
