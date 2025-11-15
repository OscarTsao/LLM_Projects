#!/usr/bin/env python3
"""Export training metrics from MLflow."""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export metrics from MLflow"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="aug_criteria_evidence",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics_export.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--mlruns-dir",
        type=Path,
        default=Path("mlruns"),
        help="MLflow runs directory",
    )
    
    args = parser.parse_args()
    
    try:
        import mlflow
        from psy_agents_aug.utils.mlflow_utils import MLflowLogger
    except ImportError:
        logger.error("MLflow not available")
        sys.exit(1)
    
    logger.info(f"Exporting metrics from experiment: {args.experiment_name}")
    
    # Set tracking URI
    mlflow.set_tracking_uri(f"file://{args.mlruns_dir.absolute()}")
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if not experiment:
        logger.error(f"Experiment not found: {args.experiment_name}")
        sys.exit(1)
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        logger.warning("No runs found")
        return
    
    logger.info(f"Found {len(runs)} runs")
    
    # Export to CSV
    runs.to_csv(args.output, index=False)
    logger.info(f"Metrics exported to: {args.output}")
    
    # Print summary
    logger.info("\nSummary:")
    logger.info(f"  Total runs: {len(runs)}")
    if "metrics.val_f1" in runs.columns:
        best_f1 = runs["metrics.val_f1"].max()
        logger.info(f"  Best val_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
