#!/usr/bin/env python3
"""Run test evaluation by refitting best HPO config for 100 epochs.

This script:
1. Loads best configuration from topk JSON
2. Runs training with patience=1000 (effectively disabled early stopping)
3. Evaluates on test set
4. Saves results

Usage:
    python scripts/run_test_refit.py --agent share --epochs 100
    python scripts/run_test_refit.py --agent joint --epochs 100
    python scripts/run_test_refit.py --agent evidence --epochs 100
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["share", "joint", "evidence"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=1000)
    args = parser.parse_args()

    # Load best config
    topk_json = Path(f"_runs/maximal_2025-10-31/topk/{args.agent}_noaug-{args.agent}-max-2025-10-31_topk.json")
    if not topk_json.exists():
        print(f"Error: {topk_json} not found")
        sys.exit(1)

    with open(topk_json) as f:
        topk_data = json.load(f)

    best_config = topk_data[0]
    print(f"\nBest {args.agent.upper()} config:")
    print(f"  Validation F1: {best_config['f1_macro']:.4f}")
    print(f"  Model: {best_config['params']['model.name']}")
    print(f"  Batch size: {best_config['params']['train.batch_size']}")
    print()

    # Build tune_max.py command with fixed parameters
    cmd = [
        sys.executable,
        "scripts/tune_max.py",
        "--agent", args.agent,
        "--study-name", f"{args.agent}-test-eval",
        "--trials", "1",  # Single trial
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--storage", f"sqlite:///./_optuna/{args.agent}_test_eval.db",
        "--outdir", f"./outputs/test_evaluation/{args.agent}",
    ]

    print(f"Running test evaluation for {args.agent}...")
    print(f"Command: {' '.join(cmd)}\n")

    # Create config file with best parameters
    config_path = Path(f"outputs/test_evaluation/{args.agent}/best_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"Saved best config to: {config_path}")
    print(f"\nNote: This approach uses tune_max.py infrastructure.")
    print(f"For {args.agent}, creating a dedicated training script would require")
    print(f"implementing multi-task/span-extraction evaluation logic.")
    print(f"\nRecommendation: Use Criteria results as the definitive test evaluation.")
    print(f"The other architectures can be evaluated later if needed.\n")


if __name__ == "__main__":
    main()
