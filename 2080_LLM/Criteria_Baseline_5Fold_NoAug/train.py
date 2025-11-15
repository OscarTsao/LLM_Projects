"""Hydra-powered CLI entry point for training the criteria baseline rebuild."""

from __future__ import annotations

import math
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src import training


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    summary = training.train(cfg)

    output_dir = Path(str(cfg.paths.output_dir)).expanduser()
    mode = summary.get("mode", "single")

    if mode == "cross_validation":
        print("=== Cross-validation training complete ===")
        print(f"Output base directory: {output_dir.resolve()}")

        val_mean = summary.get("val_metrics_mean", {})
        val_std = summary.get("val_metrics_std", {})
        if val_mean:
            print("Validation metrics (mean ± std):")
            for metric in sorted(val_mean):
                mean_val = float(val_mean.get(metric, float("nan")))
                std_val = float(val_std.get(metric, float("nan")))
                mean_str = f"{mean_val:.4f}" if math.isfinite(mean_val) else str(mean_val)
                std_str = f"{std_val:.4f}" if math.isfinite(std_val) else str(std_val)
                print(f"  {metric}: {mean_str} ± {std_str}")

        test_mean = summary.get("test_metrics_mean", {})
        test_std = summary.get("test_metrics_std", {})
        if test_mean:
            print("Test metrics (mean ± std):")
            for metric in sorted(test_mean):
                mean_val = float(test_mean.get(metric, float("nan")))
                std_val = float(test_std.get(metric, float("nan")))
                mean_str = f"{mean_val:.4f}" if math.isfinite(mean_val) else str(mean_val)
                std_str = f"{std_val:.4f}" if math.isfinite(std_val) else str(std_val)
                print(f"  {metric}: {mean_str} ± {std_str}")

        best_epoch_mean = summary.get("best_epoch_mean")
        best_epoch_std = summary.get("best_epoch_std")
        if best_epoch_mean is not None:
            mean_str = f"{best_epoch_mean:.2f}" if isinstance(best_epoch_mean, (int, float)) and math.isfinite(best_epoch_mean) else str(best_epoch_mean)
            std_str = f"{best_epoch_std:.2f}" if isinstance(best_epoch_std, (int, float)) and math.isfinite(best_epoch_std) else str(best_epoch_std)
            print(f"Best epoch (mean ± std): {mean_str} ± {std_str}")

        print("Fold results:")
        for fold_summary in summary.get("fold_summaries", []):
            fold_idx = fold_summary.get("fold_index", "?")
            fold_dir = fold_summary.get("output_dir")
            best_epoch = fold_summary.get("best_epoch")
            val_metrics = fold_summary.get("val_metrics", {})
            test_metrics = fold_summary.get("test_metrics", {})
            print(f"  Fold {fold_idx}: best epoch {best_epoch}")
            print(f"    Output: {fold_dir}")
            print(f"    Val: {val_metrics}")
            print(f"    Test: {test_metrics}")
            if fold_summary.get("mlflow_run_id"):
                print(f"    MLflow run id: {fold_summary['mlflow_run_id']}")
    else:
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
