#!/usr/bin/env python3
"""Export metrics from MLflow runs to CSV/JSON."""

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Export metrics from MLflow tracking"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (CSV or JSON)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Output format",
    )
    
    args = parser.parse_args()
    
    # Set tracking URI
    mlflow.set_tracking_uri(args.tracking_uri)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if not experiment:
        print(f"ERROR: Experiment '{args.experiment_name}' not found")
        return 1
    
    print(f"Exporting metrics from experiment: {args.experiment_name}")
    
    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("No runs found in experiment")
        return 1
    
    print(f"Found {len(runs)} runs")
    
    # Export to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "csv":
        runs.to_csv(output_path, index=False)
        print(f"Metrics exported to {output_path}")
    elif args.format == "json":
        runs.to_json(output_path, orient="records", indent=2)
        print(f"Metrics exported to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
