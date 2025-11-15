#!/usr/bin/env python
"""Run distributed HPO across multiple workers/GPUs (Phase 12).

This script provides a CLI for running HPO with parallel execution.

Usage:
    # Run with 4 workers (CPU)
    python scripts/run_distributed_hpo.py \\
        --agent criteria \\
        --n-trials 100 \\
        --n-workers 4

    # Run with 4 GPUs
    python scripts/run_distributed_hpo.py \\
        --agent evidence \\
        --n-trials 200 \\
        --n-workers 4 \\
        --gpu-ids 0,1,2,3

    # Use PostgreSQL storage
    python scripts/run_distributed_hpo.py \\
        --agent criteria \\
        --n-trials 100 \\
        --n-workers 8 \\
        --storage postgresql://localhost/optuna
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import optuna

from psy_agents_noaug.hpo.distributed import (
    ParallelExecutor,
    check_gpu_availability,
    check_storage_health,
    create_distributed_storage,
    get_storage_recommendations,
    recommend_parallel_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def setup_storage(
    storage_url: str | None,
    n_workers: int,
) -> str:
    """Setup storage backend.

    Args:
        storage_url: Storage URL (None = auto-select)
        n_workers: Number of workers

    Returns:
        Storage URL
    """
    if storage_url:
        # Use provided storage
        LOGGER.info("Using provided storage: %s", storage_url)
        health = check_storage_health(storage_url)
        if not health["accessible"]:
            raise ValueError(f"Storage not accessible: {health['error']}")
        return storage_url

    # Get recommendations
    recs = get_storage_recommendations(n_workers=n_workers)
    LOGGER.info("Storage recommendation: %s", recs["recommended_backend"])
    LOGGER.info("Reason: %s", recs["reason"])

    # Create default storage
    if recs["recommended_backend"] == "postgresql":
        LOGGER.warning(
            "PostgreSQL recommended but not configured. Falling back to SQLite."
        )
        LOGGER.warning(
            "For production use, configure PostgreSQL: "
            "--storage postgresql://user:pass@host/db"
        )

    storage_url = "sqlite:///distributed_hpo.db"
    LOGGER.info("Using SQLite storage: %s", storage_url)

    return storage_url


def create_objective_for_agent(agent: str):
    """Create objective function for agent.

    Args:
        agent: Agent name (criteria/evidence/share/joint)

    Returns:
        Objective function
    """
    # Import training utilities
    from psy_agents_noaug.cli import train

    def objective(trial: optuna.Trial) -> float:
        """HPO objective function.

        Args:
            trial: Optuna trial

        Returns:
            Validation metric
        """
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

        # Run training (would need full implementation)
        # For now, return dummy value
        LOGGER.info(
            "Trial %d: lr=%.2e, batch=%d, warmup=%.3f, wd=%.3f",
            trial.number,
            lr,
            batch_size,
            warmup_ratio,
            weight_decay,
        )

        # Simulate training (replace with actual training)
        import time

        time.sleep(1.0)  # Simulate work

        # Return dummy metric (replace with actual validation score)
        return lr * 1000 + batch_size

    return objective


def run_distributed_hpo(
    agent: str,
    n_trials: int,
    n_workers: int,
    gpu_ids: list[int] | None,
    storage_url: str | None,
    study_name: str | None,
) -> None:
    """Run distributed HPO.

    Args:
        agent: Agent name (criteria/evidence/share/joint)
        n_trials: Total number of trials
        n_workers: Number of parallel workers
        gpu_ids: GPU IDs to use (None = CPU)
        storage_url: Storage URL (None = auto)
        study_name: Study name (None = auto-generate)
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Distributed HPO Configuration")
    LOGGER.info("=" * 80)
    LOGGER.info("Agent: %s", agent)
    LOGGER.info("Total trials: %d", n_trials)
    LOGGER.info("Workers: %d", n_workers)
    LOGGER.info("GPUs: %s", gpu_ids if gpu_ids else "CPU only")

    # Setup storage
    storage = setup_storage(storage_url, n_workers)

    # Generate study name
    if study_name is None:
        import time

        study_name = f"{agent}-distributed-{int(time.time())}"

    # Check GPU availability
    if gpu_ids:
        gpu_info = check_gpu_availability()
        if not gpu_info["available"]:
            raise ValueError("GPUs requested but none available")

        invalid_gpus = [g for g in gpu_ids if g >= gpu_info["n_gpus"]]
        if invalid_gpus:
            raise ValueError(
                f"Invalid GPU IDs: {invalid_gpus} (available: 0-{gpu_info['n_gpus']-1})"
            )

        LOGGER.info("GPU availability check passed: %d GPUs", gpu_info["n_gpus"])

    # Get parallel config recommendations
    config = recommend_parallel_config(
        n_trials=n_trials,
        n_gpus=len(gpu_ids) if gpu_ids else 0,
        trial_duration=60.0,
    )
    LOGGER.info("Parallel config recommendation:")
    LOGGER.info("  %s", config["recommendation"])

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=False,
    )
    LOGGER.info("Created study: %s", study_name)

    # Create objective
    objective = create_objective_for_agent(agent)

    # Create executor
    executor = ParallelExecutor(
        n_workers=n_workers,
        gpu_ids=gpu_ids or [],
    )

    # Run optimization
    LOGGER.info("=" * 80)
    LOGGER.info("Starting distributed optimization...")
    LOGGER.info("=" * 80)

    result = executor.optimize(
        study=study,
        objective=objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print results
    LOGGER.info("=" * 80)
    LOGGER.info("Optimization Complete")
    LOGGER.info("=" * 80)
    LOGGER.info("Total trials: %d", result.total_trials)
    LOGGER.info("Completed: %d", result.completed_trials)
    LOGGER.info("Failed: %d", result.failed_trials)
    LOGGER.info("Pruned: %d", result.pruned_trials)
    LOGGER.info("Best value: %s", f"{result.best_value:.6f}" if result.best_value else "N/A")
    LOGGER.info("Execution time: %.2f seconds", result.execution_time)

    if result.worker_stats:
        LOGGER.info("\nWorker Statistics:")
        for worker_id, stats in result.worker_stats.items():
            LOGGER.info(
                "  Worker %d: %d trials (avg: %.2f s/trial)",
                worker_id,
                stats["completed"],
                stats["avg_time"],
            )

    LOGGER.info("\nStudy name: %s", study_name)
    LOGGER.info("Storage: %s", storage)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run distributed HPO with parallel execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
        help="Agent to optimize",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Total number of trials (default: 100)",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). None = CPU only",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Storage URL (e.g., 'postgresql://localhost/optuna'). None = auto",
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (default: auto-generate)",
    )

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Run distributed HPO
    try:
        run_distributed_hpo(
            agent=args.agent,
            n_trials=args.n_trials,
            n_workers=args.n_workers,
            gpu_ids=gpu_ids,
            storage_url=args.storage,
            study_name=args.study_name,
        )

    except Exception as e:
        LOGGER.error("Distributed HPO failed: %s", e)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
