from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna

from src.hpo.search_spaces import make_search_space_stage_a, make_search_space_stage_b, STRUCTURAL_KEYS
from src.hpo.optuna_callbacks import PlateauStopper, TelemetryCallback
from src.models.build_model import resolve_model_config
from src.training.train_model import train_model
from src.utils.mlflow_buffer import MlflowBufferedLogger
from src.utils.oom_guard import OOMDuringTraining
from src.utils.seed_utils import choose_trial_seed, set_all_seeds
from src.utils.hydra_mlflow import mlflow_run
from src.visualize.optuna_artifacts import export_study_artifacts
import os


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with Path(path).open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _resolve_configs(model_name: str | None) -> Dict[str, Any]:
    # Minimal config skeleton
    model, max_len, tags = resolve_model_config(model_name, 256)
    # Data config
    data_cfg = _load_yaml(Path("configs/data/redsm5.yaml"))
    data_path = data_cfg.get("dataset", {}).get("path", "data/redsm5_sample.jsonl")
    return {"model": {"name": model, "max_length": max_len}, "tags": tags, "data": {"path": data_path}}


def _objective(
    study: optuna.study.Study,
    trial: optuna.Trial,
    *,
    stage: str,
    model_name: str,
    epochs: int,
    suggest_fn,
    base_from: Optional[Dict[str, Any]] = None,
    metric_key: str = "val_metric",
    base_seed: int = 42,
) -> float:
    # Trial seed and logger
    seed = choose_trial_seed(study.study_name, trial.number, base_seed)
    set_all_seeds(seed)
    run_name = f"{study.study_name}_t{trial.number:04d}"
    buf = MlflowBufferedLogger(run_name)
    buf.replay()
    buf.set_tags({
        "stage": stage,
        "trial_number": str(trial.number),
        "seed": str(seed),
        "study_name": study.study_name,
    })

    # Suggest params according to stage
    params = suggest_fn(trial)

    # Merge into config; cap max_length for model
    resolved = _resolve_configs(model_name)
    model, max_len, tags = resolve_model_config(model_name, params.get("max_length", resolved["model"]["max_length"]))
    params["max_length"] = max_len
    if tags.get("max_length_capped"):
        buf.set_tags({"max_length_capped": "true"})

    # Basic invalid-config correction for aug strengths if aug disabled (set to 0)
    aug_mask = int(params.get("aug_mask", 0))
    for name in ["token_mask", "word_dropout", "span_delete", "rand_swap", "syn_replace"]:
        key = f"aug_strength_{name}"
        if aug_mask == 0 and params.get(key, 0.0) > 0.0:
            params[key] = 0.0

    # Heuristic: batch_size x max_length guard handled inside train_model via oom_guard
    cfg = {"model": {"name": model, "max_length": max_len}, "params": params}

    # Prefer HF backend when available and USE_HF_TRAIN=1
    if os.environ.get("USE_HF_TRAIN", "0") in ("1", "true", "yes"):
        cfg["backend"] = "hf"
    # Run training
    t0 = time.time()
    try:
        out = train_model(cfg, trial=trial, epochs=epochs, logger=buf)
    except OOMDuringTraining:
        buf.set_tags({"oom": "true"})
        raise optuna.TrialPruned("OOM during training")
    finally:
        buf.replay()

    metric = float(getattr(out, "val_metric", 0.0))
    buf.log_metrics({metric_key: metric})
    # Mirror key params and metric
    mirror = {k: v for k, v in params.items() if k in (STRUCTURAL_KEYS | {"learning_rate", "dropout", "weight_decay", "warmup_ratio", "batch_size", "max_length"})}
    buf.log_params(mirror)
    buf.set_tags({"metric_key": metric_key})
    return metric


def _sorted_top_trials(study: optuna.study.Study, k_top: int, direction: str) -> List[optuna.trial.FrozenTrial]:
    trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
    rev = direction == "maximize"
    trials.sort(key=lambda t: float(t.value), reverse=rev)
    return trials[:k_top]


def hpo_two_stage(
    model: str = "microsoft/deberta-v3-base",
    trials_a: int = 400,
    epochs_a: int = 5,
    trials_b: int = 120,
    epochs_b: int = 12,
    k_top: int = 5,
    seed: int = 42,
    study_prefix: Optional[str] = None,
    direction: str = "maximize",
    metric_key: str = "val_metric",
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    study_name_a = f"{study_prefix or 'evidence'}-A"
    study_name_b = f"{study_prefix or 'evidence'}-B"

    # Stage A
    sampler_a = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=60, n_ei_candidates=128, seed=42)
    pruner_a = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=epochs_a, reduction_factor=3)
    study_a = optuna.create_study(direction=direction, sampler=sampler_a, pruner=pruner_a, study_name=study_name_a)

    # Auto-ramp batch upper bound using a quick GPU probe if HF backend is enabled
    ss_cfg: Dict[str, Any] = {}
    if os.environ.get("USE_HF_TRAIN", "0") in ("1", "true", "yes"):
        try:
            import torch  # type: ignore
            from transformers import AutoModel  # type: ignore

            if torch.cuda.is_available():
                device = torch.device("cuda")
                base_model = AutoModel.from_pretrained(model).to(device)
                # Probe worst-case length to be conservative
                max_len_probe = 512
                candidates = [64, 48, 32, 24, 16, 12, 8, 6, 4]
                safe_bs = 4
                for b in candidates:
                    try:
                        with torch.no_grad():
                            input_ids = torch.zeros((b, max_len_probe), dtype=torch.long, device=device)
                            attention_mask = torch.ones_like(input_ids)
                            with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                                _ = base_model(input_ids=input_ids, attention_mask=attention_mask)
                        safe_bs = b
                        torch.cuda.empty_cache()
                        break
                    except Exception as e:  # noqa: BLE001
                        torch.cuda.empty_cache()
                        continue
                ss_cfg["batch_size_max"] = safe_bs
        except Exception:
            pass

    suggest_a = make_search_space_stage_a(ss_cfg)
    stopper_a = PlateauStopper(patience_trials=120, direction=direction)
    telemetry_a = TelemetryCallback(total_trials=trials_a, direction=direction)

    def objective_a(trial: optuna.Trial) -> float:
        return _objective(study_a, trial, stage="A", model_name=model, epochs=epochs_a, suggest_fn=suggest_a, base_from=None, metric_key=metric_key, base_seed=seed)

    study_a.optimize(objective_a, n_trials=trials_a, timeout=timeout_seconds, callbacks=[stopper_a, telemetry_a])

    # Artifacts for Stage A
    artifacts_root = Path("artifacts/hpo")
    out_a = artifacts_root / study_name_a
    best_params_a = study_a.best_trial.params if study_a.best_trial is not None else {}
    resolved_cfg_a = {"model": {"name": model}, "params": best_params_a}
    export_study_artifacts(study_a, out_a, best_params_a, resolved_cfg_a)

    top_trials = _sorted_top_trials(study_a, k_top=k_top, direction=direction)
    if not top_trials:
        return {"stage": "A", "message": "No successful trials in Stage A"}
    winner_a = top_trials[0]

    # Stage B
    sampler_b = optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=30, n_ei_candidates=128, seed=123)
    pruner_b = optuna.pruners.PercentilePruner(25.0, n_startup_trials=10, n_warmup_steps=2)
    study_b = optuna.create_study(direction=direction, sampler=sampler_b, pruner=pruner_b, study_name=study_name_b)

    frozen_struct = {k: v for k, v in winner_a.params.items() if k in STRUCTURAL_KEYS}
    base_cont = {k: v for k, v in winner_a.params.items() if k not in STRUCTURAL_KEYS}
    suggest_b = make_search_space_stage_b(ss_cfg, frozen_struct=frozen_struct, base_continuous=base_cont)
    stopper_b = PlateauStopper(patience_trials=120, direction=direction)
    telemetry_b = TelemetryCallback(total_trials=trials_b, direction=direction)

    # Enqueue exact top-K configs
    for ft in top_trials:
        study_b.enqueue_trial(ft.params)

    def objective_b(trial: optuna.Trial) -> float:
        return _objective(study_b, trial, stage="B", model_name=model, epochs=epochs_b, suggest_fn=suggest_b, base_from=winner_a.params, metric_key=metric_key, base_seed=seed)

    study_b.optimize(objective_b, n_trials=trials_b, timeout=timeout_seconds, callbacks=[stopper_b, telemetry_b])

    out_b = artifacts_root / study_name_b
    best_params_b = study_b.best_trial.params if study_b.best_trial is not None else {}
    resolved_cfg_b = {"model": {"name": model}, "params": best_params_b}
    export_study_artifacts(study_b, out_b, best_params_b, resolved_cfg_b)

    print(json.dumps({"winner": best_params_b, "best_value": study_b.best_value}, indent=2))
    return {
        "stage_a": {"study": study_name_a, "best_value": study_a.best_value, "best_params": best_params_a, "artifacts": str(out_a)},
        "stage_b": {"study": study_name_b, "best_value": study_b.best_value, "best_params": best_params_b, "artifacts": str(out_b)},
    }


def retrain_best(model: str, best_params: Dict[str, Any], seeds: int = 3, epochs: int = 12) -> Dict[str, Any]:
    vals: List[float] = []
    for i in range(seeds):
        seed = 42 + i
        set_all_seeds(seed)
        cfg = {"model": {"name": model, "max_length": int(best_params.get("max_length", 256))}, "params": best_params}
        out = train_model(cfg, trial=None, epochs=epochs, logger=None)
        vals.append(float(out.val_metric))
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
    std = math.sqrt(max(0.0, var))
    return {"seeds": seeds, "values": vals, "mean": mean, "std": std}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage Optuna HPO driver")
    p.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--trials-a", type=int, default=400)
    p.add_argument("--epochs-a", type=int, default=5)
    p.add_argument("--trials-b", type=int, default=120)
    p.add_argument("--epochs-b", type=int, default=12)
    p.add_argument("--k-top", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-prefix", type=str, default=None)
    p.add_argument("--direction", type=str, default="maximize")
    p.add_argument("--metric-key", type=str, default="val_metric")
    p.add_argument("--timeout-seconds", type=int, default=None)
    p.add_argument("--retrain-best", action="store_true", help="Retrain the final best with multiple seeds and report mean±std")
    p.add_argument("--retrain-seeds", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = hpo_two_stage(
        model=args.model,
        trials_a=args.trials_a,
        epochs_a=args.epochs_a,
        trials_b=args.trials_b,
        epochs_b=args.epochs_b,
        k_top=args.k_top,
        seed=args.seed,
        study_prefix=args.study_prefix,
        direction=args.direction,
        metric_key=args.metric_key,
        timeout_seconds=args.timeout_seconds,
    )
    if args.retrain_best and isinstance(info, dict) and "stage_b" in info:
        best_params = info["stage_b"].get("best_params", {})
        r = retrain_best(args.model, best_params, seeds=args.retrain_seeds, epochs=args.epochs_b)
        out_dir = Path(info["stage_b"].get("artifacts", "artifacts/hpo"))
        out = Path(out_dir) / f"retrain_{args.retrain_seeds}_seeds.json"
        out.write_text(json.dumps(r, indent=2))
        print(json.dumps({"retrain": r}, indent=2))


if __name__ == "__main__":
    main()
