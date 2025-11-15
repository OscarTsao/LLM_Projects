"""Retrain best configuration with multiple seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import optuna

from dataaug_multi_both.hpo.driver import DEFAULT_MODEL
from dataaug_multi_both.hpo.objective import ObjectiveConfig, build_objective
from dataaug_multi_both.hpo.run_study import StageSettings
from dataaug_multi_both.mlflow_init import init_mlflow


class _DummyTrial:
    def __init__(self, number: int) -> None:
        self.number = number

    def set_user_attr(self, key: str, value: Any) -> None:
        pass

    def report(self, value: float, step: int | None = None) -> None:  # pragma: no cover - simple stub
        pass

    def should_prune(self) -> bool:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrain best configuration with multiple seeds")
    parser.add_argument("--best-path", required=True, help="Path to best_trial.json emitted by hpo-best")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to retrain with")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per retrain run")
    parser.add_argument("--output-root", default="experiments/retrain_runs", help="Directory for outputs")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base encoder model name")
    parser.add_argument("--dataset-config", default="configs/data/dataset.yaml", help="Dataset YAML path")
    parser.add_argument("--use-real-data", action="store_true", help="Use real dataset instead of synthetic")
    parser.add_argument("--synthetic-train-size", type=int, default=128)
    parser.add_argument("--synthetic-val-size", type=int, default=64)
    parser.add_argument("--synthetic-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, help="Global seed offset")
    return parser


def _load_best_params(best_path: Path) -> Dict[str, Any]:
    with best_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("params", payload)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mlflow_buffer = init_mlflow(
        tracking_uri=f"sqlite:///{output_root / 'mlflow.db'}",
        experiment_name="deberta_v3_evidence_retrain",
        buffer_dir=output_root / "artifacts/mlflow_buffer",
    )

    best_params = _load_best_params(Path(args.best_path))

    objective_cfg = ObjectiveConfig(
        output_root=output_root,
        default_model=args.model,
        dataset_config=Path(args.dataset_config) if args.dataset_config else None,
        objective_metric="val_f1",
        seed=args.seed,
        use_synthetic=not args.use_real_data,
        synthetic_train_size=max(4, args.synthetic_train_size),
        synthetic_val_size=max(2, args.synthetic_val_size),
        synthetic_seq_len=max(8, args.synthetic_seq_len),
        mlflow_buffer=mlflow_buffer,
    )
    objective_fn = build_objective(objective_cfg)

    metrics: List[float] = []
    results_dir = output_root / "artifacts/retrain"
    results_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.seeds):
        seed = args.seed + idx
        settings = StageSettings(
            stage_name=f"retrain_seed_{seed}",
            search_space={},
            sampler=optuna.samplers.RandomSampler(seed=seed),
            pruner=optuna.pruners.NopPruner(),
            n_trials=1,
            timeout=None,
            plateau_patience=999,
            epochs=args.epochs,
        )
        trial = _DummyTrial(number=idx)
        start = time.time()
        metric = objective_fn(trial, dict(best_params), settings)
        duration = time.time() - start
        metrics.append(metric)
        with (results_dir / f"seed_{seed}.json").open("w", encoding="utf-8") as fh:
            json.dump({"seed": seed, "metric": metric, "duration_seconds": duration}, fh, indent=2)
        mlflow_buffer.log_metric("retrain.metric", metric)

    summary = {
        "seeds": args.seeds,
        "metrics": metrics,
        "mean": statistics.mean(metrics) if metrics else 0.0,
        "stdev": statistics.stdev(metrics) if len(metrics) > 1 else 0.0,
    }
    with (results_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    mlflow_buffer.log_metric("retrain.metric_mean", summary["mean"])
    mlflow_buffer.log_metric("retrain.metric_stdev", summary["stdev"])
    mlflow_buffer.log_artifact(results_dir / "summary.json")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
