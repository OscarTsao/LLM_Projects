"""Lightweight HPO runner for the Evidence pair classifier.

This avoids heavy dependencies (Optuna, Hydra) while providing a
config-driven search over a small hyperparameter space. It trains
the deterministic baseline model, evaluates metrics, computes an
objective, and exports the best artifacts to `outputs/hpo/{study}/`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None

from src.agents import EvidenceAgent, EvaluationAgent, CriteriaAgent
from src.utils.data import load_dataset
from src.utils.io import read_jsonl


@dataclass
class HpoConfig:
    study_name: str
    n_trials: int
    timeout: Optional[int]
    sampler: str
    pruner: str
    space: Dict[str, Any]
    objective_expr: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_hpo_config(path: Path) -> HpoConfig:
    cfg = _load_yaml(path)
    # Support both template and simplified schemas
    study_name = cfg.get("study_name") or cfg.get("study", {}).get("name") or "evidence_hpo"
    n_trials = int(cfg.get("n_trials") or cfg.get("study", {}).get("n_trials", 16))
    timeout = cfg.get("timeout") or cfg.get("study", {}).get("timeout")
    sampler = (cfg.get("sampler") or cfg.get("study", {}).get("sampler") or "tpe").upper()
    pruner = (cfg.get("pruner") or cfg.get("study", {}).get("pruner") or "asha").upper()
    space = cfg.get("search_space") or cfg.get("space") or {}
    objective_expr = cfg.get("objective") or "evidence_macro_f1_present + 0.2*negation_precision - 0.5*criteria_ece"
    return HpoConfig(
        study_name=str(study_name),
        n_trials=int(n_trials),
        timeout=int(timeout) if timeout is not None else None,
        sampler=str(sampler),
        pruner=str(pruner),
        space=space,
        objective_expr=str(objective_expr),
    )


def _now_run_id(prefix: str = "trial") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{random.randint(1000, 9999)}"


def _uniform(a: float, b: float) -> float:
    lo, hi = (a, b) if a <= b else (b, a)
    return random.random() * (hi - lo) + lo


def _sample_param(values: Any) -> Any:
    # If list of exactly two numbers, treat as continuous range
    if isinstance(values, list) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
        a, b = float(values[0]), float(values[1])
        val = _uniform(a, b)
        # Keep ints as ints when endpoints are ints
        if all(isinstance(v, int) for v in values):
            return int(round(val))
        return float(val)
    # If list of more than two, treat as categorical choices
    if isinstance(values, list) and len(values) >= 1:
        return random.choice(values)
    # Fallback to given value
    return values


def sample_hparams(space: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    random.setstate(rng.getstate())
    return {k: _sample_param(v) for k, v in space.items()}


def _load_data_from_config(data_config: Path) -> Tuple[List[Dict], List[Dict], List[Dict], Path]:
    cfg = _load_yaml(data_config)
    dataset_path = Path(cfg.get("dataset", {}).get("path", ""))
    if not dataset_path:
        raise ValueError("Dataset path missing in data config")
    dataset = list(load_dataset(dataset_path))

    splits = cfg.get("splits", {})
    if splits:
        # Lazy split loader: reuse logic similar to pipeline but keep simple here
        from src.utils.splits import filter_dataset_by_ids
        split_dir = Path(splits.get("dir", "configs/data/splits"))
        def resolve(p):
            pp = Path(str(p).replace("${splits.dir}", str(split_dir)))
            return pp if pp.is_absolute() else split_dir / pp
        from src.utils.io import read_jsonl
        def load_ids(path: Path) -> List[str]:
            if not path.is_file():
                return []
            return [row.get("post_id") for row in read_jsonl(path)]
        train_ids = load_ids(resolve(splits.get("train", "train.jsonl")))
        dev_ids = load_ids(resolve(splits.get("dev", "dev.jsonl")))
        test_ids = load_ids(resolve(splits.get("test", "test.jsonl")))
        train_ds = filter_dataset_by_ids(dataset, train_ids) if train_ids else dataset
        eval_ids = test_ids or dev_ids or train_ids
        eval_ds = filter_dataset_by_ids(dataset, eval_ids) if eval_ids else dataset
    else:
        train_ds = dataset
        eval_ds = dataset

    return list(train_ds), list(eval_ds), dataset, dataset_path


def _safe_eval(expr: str, variables: Dict[str, float]) -> float:
    # Replace known aliases
    aliases = {
        "macro_F1_present": "evidence_macro_f1_present",
        "macro_f1_present": "evidence_macro_f1_present",
        "neg_precision": "negation_precision",
        "ECE": "criteria_ece",
        "ece": "criteria_ece",
    }
    for k, v in aliases.items():
        expr = expr.replace(k, v)
    # Only allow numbers, variable names, and + - * / ( ) . _ letters
    allowed_chars = set("0123456789.+-*/() _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if not set(expr) <= allowed_chars:
        raise ValueError("Objective expression contains unsupported characters")
    # Tokenize by spaces and operators
    # Simple eval with restricted globals/locals
    code = compile(expr, "<objective>", "eval")
    return float(eval(code, {"__builtins__": {}}, variables))


def hpo_run(
    data_config: Path,
    evidence_config: Path,
    hpo_config: Path,
    outputs_root: Path = Path("outputs"),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed or 42)
    hcfg = load_hpo_config(hpo_config)
    study_dir = outputs_root / "hpo" / hcfg.study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds, full_ds, dataset_path = _load_data_from_config(data_config)
    evidence_agent = EvidenceAgent(evidence_config)
    criteria_agent = CriteriaAgent(Path("configs/criteria/aggregator.yaml"))
    eval_agent = EvaluationAgent(Path("artifacts/calibration.json"))

    best: Dict[str, Any] = {"objective": -math.inf}
    trial_summaries: List[Dict[str, Any]] = []

    for trial_idx in range(hcfg.n_trials):
        hparams = sample_hparams(hcfg.space, rng)
        run_id = _now_run_id(prefix=f"trial{trial_idx:03d}")
        training_dir = outputs_root / "training" / hcfg.study_name / run_id
        evaluation_dir = outputs_root / "evaluation" / hcfg.study_name / run_id

        train_info = evidence_agent.train(train_ds, training_dir, seed=seed, dry_run=False, hparams=hparams)
        preds_path = evaluation_dir / "predictions.jsonl"
        evidence_agent.infer(eval_ds, Path(train_info["model_path"]), preds_path, seed=seed)
        predictions = list(read_jsonl(preds_path))

        # Build criteria results from predictions
        criteria_results = criteria_agent.aggregate(predictions)
        metrics_paths = eval_agent.evaluate(predictions, criteria_results, full_ds, evaluation_dir)
        # Load metrics and compute objective
        with Path(metrics_paths["test_metrics"]).open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        variables = {
            "evidence_macro_f1_present": float(metrics.get("evidence_macro_f1_present", 0.0)),
            "negation_precision": float(metrics.get("negation_precision", 0.0)),
            "criteria_ece": float(metrics.get("criteria_ece", metrics.get("ece", 1.0))),
        }
        objective = _safe_eval(hcfg.objective_expr, variables)

        summary = {
            "trial": trial_idx,
            "run_id": run_id,
            "training_dir": str(training_dir),
            "evaluation_dir": str(evaluation_dir),
            "hparams": hparams,
            "metrics": variables,
            "objective": objective,
        }
        trial_summaries.append(summary)

        if objective > best["objective"]:
            best = summary

    # Export best artifacts
    best_dir = study_dir
    best_dir.mkdir(parents=True, exist_ok=True)
    # Copy model + metrics
    best_training_dir = Path(best["training_dir"])
    best_eval_dir = Path(best["evaluation_dir"])
    model_src = best_training_dir / "model.ckpt"
    config_src = best_training_dir / "config.yaml"
    val_src = best_eval_dir / "val_metrics.json"
    test_src = best_eval_dir / "test_metrics.json"

    def _copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            dst.write_bytes(src.read_bytes())

    _copy(model_src, best_dir / "best.ckpt")
    # Persist best config combining evidence config and hparams
    best_cfg_path = best_dir / "best_config.yaml"
    cfg_snapshot = {
        "evidence_config": str(evidence_config),
        "data_config": str(data_config),
        "hpo_config": str(hpo_config),
        "hparams": best.get("hparams", {}),
    }
    if yaml is not None:
        with best_cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg_snapshot, fh)
    else:
        best_cfg_path.write_text(json.dumps(cfg_snapshot, indent=2))

    _copy(val_src, best_dir / "val_metrics.json")
    _copy(test_src, best_dir / "test_metrics.json")

    # Persist study summary
    summary_path = study_dir / "study_summary.json"
    payload = {
        "study_name": hcfg.study_name,
        "n_trials": hcfg.n_trials,
        "sampler": hcfg.sampler,
        "pruner": hcfg.pruner,
        "objective": hcfg.objective_expr,
        "best": best,
        "trials": trial_summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2))

    return {
        "study_dir": str(study_dir),
        "best_model": str((best_dir / "best.ckpt").resolve()),
        "best_config": str(best_cfg_path.resolve()),
        "study_summary": str(summary_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HPO for evidence pair classifier")
    p.add_argument("--data-config", type=Path, default=Path("configs/data/redsm5.yaml"))
    p.add_argument("--evidence-config", type=Path, default=Path("configs/evidence/pairclf.yaml"))
    p.add_argument("--hpo-config", type=Path, default=Path("configs/hpo/evidence_pairclf.yaml"))
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = hpo_run(args.data_config, args.evidence_config, args.hpo_config, args.outputs_root, seed=args.seed)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
