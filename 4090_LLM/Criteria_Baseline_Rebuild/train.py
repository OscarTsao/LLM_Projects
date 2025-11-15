"""Hydra-powered CLI entry point for training the criteria baseline rebuild."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src import training


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = training.train(cfg)

    output_dir = Path(str(cfg.paths.output_dir)).expanduser()
    print("=== Training complete ===")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(f"Validation metrics: {summary['val_metrics']}")
    print(f"Test metrics: {summary['test_metrics']}")
    if summary.get("mlflow_run_id"):
        print(f"MLflow run id: {summary['mlflow_run_id']}")
    if summary.get("mlflow_experiment_id"):
        print(f"MLflow experiment id: {summary['mlflow_experiment_id']}")


if __name__ == "__main__":
    main()
