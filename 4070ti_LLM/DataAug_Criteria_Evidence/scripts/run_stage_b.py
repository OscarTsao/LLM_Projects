#!/usr/bin/env python
"""SUPERMAX Stage-B: Multi-objective Optimization with NSGA-II.

This is the second stage of the SUPERMAX HPO pipeline:
- Multi-objective optimization (F1-macro vs. ECE)
- NSGA-II sampler for Pareto front exploration
- Seeds from Stage-A's top-K configurations
- 1200-2400 trials recommended
- Exports Pareto front for Stage-C

Usage:
    # Requires Stage-A output
    python scripts/run_stage_b.py \\
        --agent criteria \\
        --stage-a-results outputs/supermax/stage_a/criteria/criteria_stage_a_top50.json

    # Quick test
    python scripts/run_stage_b.py \\
        --agent criteria \\
        --stage-a-results outputs/supermax/stage_a/criteria/criteria_stage_a_top50.json \\
        --trials 20 --epochs 3

    # Production run
    python scripts/run_stage_b.py \\
        --agent criteria \\
        --stage-a-results outputs/supermax/stage_a/criteria/criteria_stage_a_top50.json \\
        --trials 1500 \\
        --epochs 10 \\
        --outdir outputs/supermax/stage_b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import optuna

from psy_agents_noaug.hpo import (
    DEFAULT_REPORT_DIR,
    ObjectiveBuilder,
    ObjectiveSettings,
    SearchSpace,
    SpaceConstraints,
    create_sampler,
    resolve_profile,
    resolve_storage,
)
from psy_agents_noaug.hpo import utils as hpo_utils

LOGGER = logging.getLogger("stage_b")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SUPERMAX Stage-B: Multi-objective optimization with NSGA-II"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
        help="Agent/architecture to optimize",
    )
    parser.add_argument(
        "--stage-a-results",
        required=True,
        type=Path,
        help="Path to Stage-A top-K JSON file (e.g., criteria_stage_a_top50.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("STAGE_B_TRIALS", "1500")),
        help="Number of trials (default: 1500, recommend 1200-2400 for production)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("STAGE_B_EPOCHS", "10")),
        help="Max epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(os.getenv("HPO_PATIENCE", "3")),
        help="Early stopping patience (default: 3)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("HPO_MAX_SAMPLES", "512")),
        help="Max samples per split (default: 512)",
    )
    parser.add_argument(
        "--seeds",
        default=os.getenv("HPO_SEEDS", "1"),
        help="Comma-separated list of random seeds (default: 1)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite:///supermax_stage_b.db)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Study name (default: supermax-{agent}-stage-b)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--outdir",
        default=os.getenv("STAGE_B_OUTDIR", "outputs/supermax/stage_b"),
        help="Output directory for reports and Pareto front",
    )
    parser.add_argument(
        "--profile",
        default=os.getenv("HPO_PROFILE", "supermax"),
        help="HPO profile name (default: supermax)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("HPO_LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=2025,
        help="Random seed for NSGA-II sampler (default: 2025)",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="NSGA-II population size (default: 50)",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=None,
        help="Number of random startup trials (default: len(stage_a_configs))",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help="Overall timeout in hours (default: None)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_seeds(seed_arg: str) -> list[int]:
    """Parse comma-separated seed string into list of integers."""
    seeds: list[int] = []
    for chunk in seed_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    return seeds or [1]


def load_stage_a_configs(path: Path) -> list[dict[str, Any]]:
    """Load Stage-A top-K configurations from JSON.

    Args:
        path: Path to Stage-A JSON export

    Returns:
        List of configuration dictionaries
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-A results not found: {path}\n"
            f"Run Stage-A first: make stage-a AGENT=<agent>"
        )

    with path.open("r") as f:
        data = json.load(f)

    if "configurations" not in data:
        raise ValueError(f"Invalid Stage-A JSON format: missing 'configurations' key")

    configs = []
    for entry in data["configurations"]:
        if "config" in entry:
            configs.append(entry["config"])

    LOGGER.info("Loaded %d configurations from Stage-A: %s", len(configs), path)
    return configs


def enqueue_stage_a_configs(
    study: optuna.Study,
    configs: list[dict[str, Any]],
    agent: str,
) -> None:
    """Enqueue Stage-A configurations as initial trials for NSGA-II.

    Args:
        study: Optuna study
        configs: List of configuration dictionaries from Stage-A
        agent: Agent name
    """
    search_space = SearchSpace(agent)

    for i, config in enumerate(configs):
        # Remove agent key if present (not a hyperparameter)
        config_clean = {k: v for k, v in config.items() if k != "agent"}

        try:
            study.enqueue_trial(config_clean)
            LOGGER.debug("Enqueued config %d/%d", i + 1, len(configs))
        except Exception as e:
            LOGGER.warning("Failed to enqueue config %d: %s", i + 1, e)

    LOGGER.info("Enqueued %d Stage-A configurations for NSGA-II seeding", len(configs))


def export_pareto_front(
    study: optuna.Study,
    outdir: Path,
    agent: str,
) -> Path:
    """Export Pareto front (non-dominated solutions) to JSON.

    Args:
        study: Completed Optuna study
        outdir: Output directory
        agent: Agent name

    Returns:
        Path to exported JSON file
    """
    # Get Pareto front trials
    pareto_trials = study.best_trials

    if not pareto_trials:
        LOGGER.warning("No Pareto front trials found")
        return outdir / f"{agent}_stage_b_pareto_empty.json"

    # Export Pareto front
    pareto_export = []
    for rank, trial in enumerate(pareto_trials, start=1):
        # Get configuration from user_attrs
        config_json = trial.user_attrs.get("config_json")
        if config_json:
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse config for trial %d", trial.number)
                continue
        else:
            config = dict(trial.params)

        # Add metadata (multi-objective values)
        export_entry = {
            "rank": rank,
            "trial_number": trial.number,
            "f1_macro": trial.values[0] if trial.values else None,
            "ece": trial.values[1] if len(trial.values) > 1 else None,
            "logloss": trial.user_attrs.get("logloss", None),
            "config": config,
        }
        pareto_export.append(export_entry)

    # Write to file
    outfile = outdir / f"{agent}_stage_b_pareto.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("w") as f:
        json.dump(
            {
                "agent": agent,
                "stage": "stage_b",
                "pareto_size": len(pareto_export),
                "total_trials": len(study.trials),
                "pareto_front": pareto_export,
            },
            f,
            indent=2,
        )

    LOGGER.info("Exported Pareto front (%d solutions) to %s", len(pareto_export), outfile)
    return outfile


def main() -> None:
    """Main entry point for Stage-B multi-objective optimization."""
    args = parse_args()
    setup_logging(args.log_level)

    # Validate Stage-A results
    if not args.stage_a_results.exists():
        LOGGER.error("Stage-A results not found: %s", args.stage_a_results)
        LOGGER.error("Run Stage-A first: make stage-a AGENT=%s", args.agent)
        raise FileNotFoundError(f"Stage-A results not found: {args.stage_a_results}")

    # Load Stage-A configurations
    stage_a_configs = load_stage_a_configs(args.stage_a_results)

    # Resolve paths and settings
    storage = resolve_storage(args.storage or "sqlite:///supermax_stage_b.db")
    profile = resolve_profile(args.profile)
    study_name = args.study_name or f"supermax-{args.agent}-stage-b"
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    LOGGER.info("=" * 80)
    LOGGER.info("SUPERMAX STAGE-B: Multi-objective Optimization (NSGA-II)")
    LOGGER.info("=" * 80)
    LOGGER.info("Agent: %s", args.agent)
    LOGGER.info("Study: %s", study_name)
    LOGGER.info("Trials: %d", args.trials)
    LOGGER.info("Epochs: %d (patience=%d)", args.epochs, args.patience)
    LOGGER.info("Seeds: %s", seeds)
    LOGGER.info("Max samples: %s", args.max_samples or "unlimited")
    LOGGER.info("Stage-A configs: %d (will be enqueued)", len(stage_a_configs))
    LOGGER.info("Population size: %d", args.population_size)
    LOGGER.info("Storage: %s", storage)
    LOGGER.info("Output: %s", outdir)
    LOGGER.info("=" * 80)

    # Create NSGA-II sampler (multi-objective)
    sampler = create_sampler(
        multi_objective=True,
        seed=args.sampler_seed,
        sampler="nsga2",
    )
    LOGGER.info("Sampler: NSGA-II (seed=%d, population=%d)", args.sampler_seed, args.population_size)

    # Create study (multi-objective: maximize F1, minimize ECE)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "minimize"],  # F1-macro (max), ECE (min)
        sampler=sampler,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        LOGGER.info("Loaded existing study with %d trials", n_existing)
    else:
        # Enqueue Stage-A configurations for warm start
        enqueue_stage_a_configs(study, stage_a_configs, args.agent)

    # Create search space (no constraints - Stage-A already narrowed it)
    search_space = SearchSpace(args.agent)
    LOGGER.info("Search space: 8 architectures, 6 optimizers, 6 aug params")

    # Create objective settings (multi-objective)
    settings = ObjectiveSettings(
        agent=args.agent,
        study=study_name,
        outdir=outdir,
        epochs=args.epochs,
        seeds=seeds,
        patience=args.patience,
        max_samples=args.max_samples,
        multi_objective=True,  # Key difference from Stage-A
        topk=None,  # Not used in Stage-B
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=f"{profile}-{args.agent}-stage-b",
    )

    # Create objective function
    objective = ObjectiveBuilder(
        search_space,
        settings,
        constraints=None,
    )

    # Run optimization
    timeout_seconds = int(args.timeout_hours * 3600) if args.timeout_hours else None

    LOGGER.info("Starting multi-objective optimization...")
    LOGGER.info(
        "Expected runtime: ~%d hours (%.1f min/trial × %d trials)",
        int(args.trials * 7 / 60),  # Rough estimate: 7 min/trial (longer than Stage-A)
        7.0,
        args.trials,
    )
    LOGGER.info("Optimizing: F1-macro (maximize) vs. ECE (minimize)")

    try:
        study.optimize(
            objective,
            n_trials=args.trials - n_existing,
            timeout=timeout_seconds,
            gc_after_trial=True,
            catch=(RuntimeError,),
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        LOGGER.warning("Optimization interrupted by user")
    except Exception as e:
        LOGGER.error("Optimization failed: %s", e, exc_info=True)
        raise

    # Summary statistics
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    LOGGER.info("=" * 80)
    LOGGER.info("STAGE-B COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Total trials: %d", len(study.trials))
    LOGGER.info("  - Completed: %d", len(completed_trials))
    LOGGER.info("  - Failed: %d", len(failed_trials))

    if completed_trials:
        pareto_trials = study.best_trials
        LOGGER.info("Pareto front size: %d solutions", len(pareto_trials))

        # Show Pareto front summary
        LOGGER.info("\nPareto Front Summary (best trade-offs):")
        for i, trial in enumerate(pareto_trials[:10], 1):  # Show top-10
            f1 = trial.values[0] if trial.values else 0.0
            ece = trial.values[1] if len(trial.values) > 1 else 0.0
            LOGGER.info("  %2d. Trial #%4d: F1=%.4f, ECE=%.4f", i, trial.number, f1, ece)

    # Export Pareto front for Stage-C
    export_path = export_pareto_front(study, outdir, args.agent)
    LOGGER.info("Pareto front exported to: %s", export_path)

    # Generate summary report
    report_path = outdir / f"{args.agent}_stage_b_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("SUPERMAX STAGE-B REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Agent: {args.agent}\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"  - Completed: {len(completed_trials)}\n")
        f.write(f"  - Failed: {len(failed_trials)}\n")
        f.write("\n")

        if completed_trials:
            pareto_trials = study.best_trials
            f.write(f"Pareto front size: {len(pareto_trials)} solutions\n\n")

            f.write("Pareto Front (top-20 by F1-macro):\n")
            sorted_pareto = sorted(pareto_trials, key=lambda t: -t.values[0])[:20]
            for i, trial in enumerate(sorted_pareto, 1):
                f1 = trial.values[0] if trial.values else 0.0
                ece = trial.values[1] if len(trial.values) > 1 else 0.0
                f.write(f"  {i:2d}. Trial #{trial.number:4d}: F1={f1:.4f}, ECE={ece:.4f}\n")

        f.write("\n")
        f.write(f"Pareto front export: {export_path}\n")
        f.write(f"Next step: Run Stage-C with Pareto front configurations\n")

    LOGGER.info("Summary report: %s", report_path)
    LOGGER.info("=" * 80)
    LOGGER.info("Stage-B complete! Proceed to Stage-C for K-fold CV refinement.")


if __name__ == "__main__":
    main()
