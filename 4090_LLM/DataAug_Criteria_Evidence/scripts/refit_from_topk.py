#!/usr/bin/env python3
"""Refit best HPO models from topk JSON configurations.

This script:
1. Loads best hyperparameter configs from topk JSON files
2. Launches training using tune_max.py with --trials 1
3. Uses train+val combined data for refitting
4. Saves checkpoints for test evaluation

Usage:
    # Refit all architectures
    python scripts/refit_from_topk.py

    # Refit specific architectures
    python scripts/refit_from_topk.py --agents share joint

    # Quick test with fewer epochs
    python scripts/refit_from_topk.py --epochs 10 --quick
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any


def load_best_config(topk_json_path: Path) -> Dict[str, Any]:
    """Load best trial configuration from topk JSON file."""
    with open(topk_json_path) as f:
        topk_data = json.load(f)

    if not topk_data:
        raise ValueError(f"No trials found in {topk_json_path}")

    # Best trial is rank 1 (index 0)
    best_trial = topk_data[0]
    return best_trial


def launch_refit_training(
    agent: str,
    best_config: Dict[str, Any],
    output_dir: Path,
    epochs: int = 100,
    patience: int = 20,
) -> subprocess.Popen:
    """
    Launch refit training using tune_max.py with a single trial.

    The trick is to use tune_max.py with --trials 1 and force it to use
    the best hyperparameters we found during HPO.
    """

    params = best_config["params"]

    print(f"\n{'='*70}")
    print(f"LAUNCHING REFIT: {agent.upper()}".center(70))
    print(f"{'='*70}")
    print(f"Validation F1: {best_config['f1_macro']:.4f}")
    print(f"Model: {params.get('model.name', 'unknown')}")
    print(f"Optimizer: {params.get('optim.name', 'unknown')}")
    print(f"Learning Rate: {params.get('optim.lr', 0.0):.2e}")
    print(f"Augmentation: {'Enabled' if params.get('aug.enabled', False) else 'Disabled'}")
    print(f"Output: {output_dir}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best config for reference
    config_file = output_dir / "best_config.json"
    with open(config_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"✓ Saved config to {config_file}")

    # Build refit command using tune_max.py
    # We'll use --trials 1 and set HPO_REFIT=1 to signal refit mode

    study_name = f"refit-{agent}-best"
    log_file = output_dir / f"{agent}_refit.log"
    pid_file = output_dir / f"{agent}_refit.pid"

    cmd = [
        sys.executable,
        "scripts/tune_max.py",
        "--agent", agent,
        "--study-name", study_name,
        "--trials", "1",
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--outdir", str(output_dir.parent),
        "--no-pruner",  # Disable pruning for refit
    ]

    # Set environment to indicate refit mode
    env = {
        **subprocess.os.environ,
        "HPO_REFIT": "1",  # Signal that this is a refit run
        "HPO_EPOCHS": str(epochs),
        "HPO_PATIENCE": str(patience),
    }

    print(f"Command: {' '.join(cmd)}")
    print(f"Log file: {log_file}")

    # Launch in background
    with open(log_file, "w") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )

    # Save PID
    with open(pid_file, "w") as f:
        f.write(str(process.pid))

    print(f"✓ Started refit job (PID: {process.pid})")

    return process


def monitor_processes(processes: Dict[str, subprocess.Popen], check_interval: int = 30):
    """Monitor running refit processes."""
    print(f"\n{'='*70}")
    print("MONITORING REFIT JOBS".center(70))
    print(f"{'='*70}")
    print(f"Checking every {check_interval} seconds...")
    print()

    while any(p.poll() is None for p in processes.values()):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status:")
        for agent, process in processes.items():
            if process.poll() is None:
                print(f"  {agent:10s}: Running (PID {process.pid})")
            else:
                returncode = process.returncode
                status = "✓ Complete" if returncode == 0 else f"✗ Failed ({returncode})"
                print(f"  {agent:10s}: {status}")

        # Check if all complete
        if all(p.poll() is not None for p in processes.values()):
            break

        time.sleep(check_interval)

    print(f"\n{'='*70}")
    print("ALL REFIT JOBS COMPLETE".center(70))
    print(f"{'='*70}\n")

    # Print final status
    for agent, process in processes.items():
        returncode = process.returncode
        if returncode == 0:
            print(f"✓ {agent}: Success")
        else:
            print(f"✗ {agent}: Failed with code {returncode}")


def main():
    parser = argparse.ArgumentParser(description="Refit best HPO models from topk JSON")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["share", "joint", "criteria", "evidence"],
        choices=["share", "joint", "criteria", "evidence"],
        help="Which architectures to refit (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for refitting (default: 100)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 epochs for testing",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Don't monitor processes, just launch and exit",
    )

    args = parser.parse_args()

    if args.quick:
        args.epochs = 10
        args.patience = 5

    print(f"\n{'='*70}")
    print("REFIT BEST HPO MODELS".center(70))
    print(f"{'='*70}")
    print(f"Architectures: {', '.join(args.agents)}")
    print(f"Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print()

    # Define paths
    project_root = Path(__file__).parent.parent
    topk_dir = project_root / "_runs/maximal_2025-10-31/topk"
    output_base = project_root / "outputs/refitted_models"

    if not topk_dir.exists():
        print(f"✗ Error: topk directory not found: {topk_dir}")
        print(f"Make sure HPO has completed and topk JSONs exist")
        sys.exit(1)

    # Launch refit jobs for each architecture
    processes = {}

    for agent in args.agents:
        print(f"\n{'#'*70}")
        print(f"# {agent.upper()}")
        print(f"{'#'*70}")

        # Load best config from topk JSON
        topk_json = topk_dir / f"{agent}_noaug-{agent}-max-2025-10-31_topk.json"

        if not topk_json.exists():
            print(f"✗ Warning: topk JSON not found: {topk_json}")
            print(f"Skipping {agent}...")
            continue

        print(f"✓ Loading config from: {topk_json}")
        best_config = load_best_config(topk_json)

        # Launch refit training
        output_dir = output_base / agent
        process = launch_refit_training(
            agent=agent,
            best_config=best_config,
            output_dir=output_dir,
            epochs=args.epochs,
            patience=args.patience,
        )

        processes[agent] = process

        # Small delay between launches to avoid resource contention
        time.sleep(2)

    if not processes:
        print("\n✗ No refit jobs were launched")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"LAUNCHED {len(processes)} REFIT JOBS".center(70))
    print(f"{'='*70}\n")

    if args.no_monitor:
        print("Refit jobs running in background. Check log files for progress:")
        for agent in processes.keys():
            log_file = output_base / agent / f"{agent}_refit.log"
            print(f"  {agent}: {log_file}")
        print()
    else:
        # Monitor processes until completion
        monitor_processes(processes, check_interval=60)

    print("\nRefit complete! Next steps:")
    print("1. Check output directories for checkpoints")
    print("2. Run test evaluation with eval scripts")
    print("3. Generate validation vs test comparison report")


if __name__ == "__main__":
    main()
