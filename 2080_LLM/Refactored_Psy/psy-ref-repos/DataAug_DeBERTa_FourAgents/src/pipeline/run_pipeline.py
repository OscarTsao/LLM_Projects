from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from src.agents import CriteriaAgent, EvidenceAgent, EvaluationAgent
from src.utils.data import load_dataset
from src.utils.io import read_jsonl, write_jsonl
from src.utils.hydra_mlflow import mlflow_run
from src.utils.splits import filter_dataset_by_ids, load_split_ids


def load_yaml(path: Path) -> Dict:
    if yaml is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-agent pipeline")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/redsm5.yaml"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline/default.yaml"))
    parser.add_argument("--evidence-config", type=Path, default=Path("configs/evidence/pairclf.yaml"))
    parser.add_argument("--suggest-config", type=Path, default=Path("configs/suggest/voi.yaml"))
    parser.add_argument("--criteria-config", type=Path, default=Path("configs/criteria/aggregator.yaml"))
    parser.add_argument("--calibration-path", type=Path, default=Path("artifacts/calibration.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = load_yaml(args.data_config)
    dataset_path = Path(data_cfg.get("dataset", {}).get("path", ""))
    if not dataset_path:
        raise ValueError("Dataset path missing in data config")
    dataset = list(load_dataset(dataset_path))

    split_cfg = data_cfg.get("splits", {})
    split_dir = Path(split_cfg.get("dir", "configs/data/splits"))

    def resolve_split(name: str, default: str) -> Path:
        value = split_cfg.get(name)
        if value is None:
            return split_dir / default
        value = str(value).replace("${splits.dir}", str(split_dir))
        path = Path(value)
        if not path.is_absolute():
            path = split_dir / path
        return path

    train_ids = load_split_ids(resolve_split("train", "train.jsonl")) if split_cfg else []
    dev_ids = load_split_ids(resolve_split("dev", "dev.jsonl")) if split_cfg else []
    test_ids = load_split_ids(resolve_split("test", "test.jsonl")) if split_cfg else []

    train_dataset = filter_dataset_by_ids(dataset, train_ids) if train_ids else dataset
    eval_ids = test_ids or dev_ids or train_ids
    eval_dataset = filter_dataset_by_ids(dataset, eval_ids) if eval_ids else dataset

    pipeline_cfg = load_yaml(args.pipeline_config)
    outputs_cfg = pipeline_cfg.get("outputs", {})
    root = Path(outputs_cfg.get("root", "outputs"))

    def resolve(value, fallback):
        if isinstance(value, str):
            return Path(value.replace("${outputs.root}", str(root)))
        if value is None:
            return Path(fallback)
        return Path(value)

    training_root = resolve(outputs_cfg.get("training_dir"), root / "training")
    evaluation_root = resolve(outputs_cfg.get("evaluation_dir"), root / "evaluation")

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    training_dir = training_root / run_id
    evaluation_dir = evaluation_root / run_id

    evidence_agent = EvidenceAgent(args.evidence_config)
    train_info = evidence_agent.train(train_dataset, training_dir, seed=args.seed, dry_run=False)

    predictions_path = evaluation_dir / "predictions.jsonl"
    evidence_agent.infer(eval_dataset, Path(train_info["model_path"]), predictions_path, seed=args.seed)
    predictions = list(read_jsonl(predictions_path))

    suggest_cfg = load_yaml(args.suggest_config)
    top_k = suggest_cfg.get("top_k", 3)
    uncertain_band = tuple(suggest_cfg.get("uncertain_band", [0.4, 0.6]))

    criteria_agent = CriteriaAgent(args.criteria_config)
    criteria_results = criteria_agent.aggregate(predictions, top_k=top_k, uncertain_band=uncertain_band)
    criteria_path = evaluation_dir / "criteria.jsonl"
    write_jsonl(criteria_path, criteria_results)

    evaluation_agent = EvaluationAgent(args.calibration_path)
    metrics_paths = evaluation_agent.evaluate(predictions, criteria_results, eval_dataset, evaluation_dir)
    evaluation_agent.run_gate_check(metrics_paths["test_metrics"])

    summary = {
        "run_id": run_id,
        "dataset": str(dataset_path),
        "training_dir": str(training_dir),
        "evaluation_dir": str(evaluation_dir),
        "predictions_path": str(predictions_path),
        "criteria_path": str(criteria_path),
        "metrics": {k: str(v) for k, v in metrics_paths.items()},
    }
    summary_path = evaluation_dir / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with mlflow_run(run_id, {"dataset": str(dataset_path)}) as mlflow:
        if mlflow:
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(criteria_path))
            mlflow.log_artifact(str(metrics_paths["test_metrics"]))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
