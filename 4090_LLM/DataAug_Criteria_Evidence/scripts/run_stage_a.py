#!/usr/bin/env python
"""SUPERMAX Stage-A: Baseline Exploration with TPE+ASHA.

This is the first stage of the SUPERMAX HPO pipeline:
- Broad baseline exploration (900-1200 trials recommended)
- TPE sampler for efficient search space exploration
- ASHA pruner for early stopping of poor trials
- Exports top-50 configurations for Stage-B

Usage:
    python scripts/run_stage_a.py --agent criteria --trials 1000

    # Quick test run
    python scripts/run_stage_a.py --agent criteria --trials 20 --epochs 2

    # Production run with full configuration
    python scripts/run_stage_a.py \\
        --agent criteria \\
        --trials 1000 \\
        --epochs 6 \\
        --topk 50 \\
        --storage sqlite:///supermax_stage_a.db \\
        --outdir outputs/supermax/stage_a
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
    create_pruner,
    create_sampler,
    resolve_profile,
    resolve_storage,
)
from psy_agents_noaug.hpo import utils as hpo_utils

LOGGER = logging.getLogger("stage_a")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SUPERMAX Stage-A: Baseline exploration HPO"
    )
    parser.add_argument(
        "--agent",
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
        help="Agent/architecture to optimize",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=int(os.getenv("STAGE_A_TRIALS", "1000")),
        help="Number of trials (default: 1000, recommend 900-1200 for production)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("STAGE_A_EPOCHS", "6")),
        help="Max epochs per trial (default: 6)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=int(os.getenv("HPO_PATIENCE", "2")),
        help="Early stopping patience (default: 2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=int(os.getenv("HPO_MAX_SAMPLES", "512")),
        help="Max samples per split (default: 512, None for full dataset)",
    )
    parser.add_argument(
        "--seeds",
        default=os.getenv("HPO_SEEDS", "1"),
        help="Comma-separated list of random seeds (default: 1)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Number of top configurations to export for Stage-B (default: 50)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (default: sqlite:///supermax_stage_a.db)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Study name (default: supermax-{agent}-stage-a)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--outdir",
        default=os.getenv("STAGE_A_OUTDIR", "outputs/supermax/stage_a"),
        help="Output directory for reports and configs",
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
        help="Random seed for TPE sampler (default: 2025)",
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


def export_topk_configs(
    study: optuna.Study,
    topk: int,
    outdir: Path,
    agent: str,
) -> Path:
    """Export top-K configurations to JSON for Stage-B.

    Args:
        study: Completed Optuna study
        topk: Number of top configurations to export
        outdir: Output directory
        agent: Agent name

    Returns:
        Path to exported JSON file
    """
    # Get completed trials sorted by primary metric (descending)
    trials = [
        t
        for t in study.get_trials(deepcopy=False)
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]

    if not trials:
        LOGGER.warning("No completed trials to export")
        return outdir / f"{agent}_stage_a_topk_empty.json"

    # Sort by primary metric (F1-macro stored in user_attrs or value)
    trials.sort(
        key=lambda t: -(t.user_attrs.get("primary", t.value or 0.0)),
        reverse=False,
    )

    # Take top-K
    topk_trials = trials[: min(topk, len(trials))]

    # Export configurations
    configs_export = []
    for rank, trial in enumerate(topk_trials, start=1):
        # Get configuration from user_attrs
        config_json = trial.user_attrs.get("config_json")
        if config_json:
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse config for trial %d", trial.number)
                continue
        else:
            # Fallback: use trial.params
            config = dict(trial.params)

        # Add metadata
        export_entry = {
            "rank": rank,
            "trial_number": trial.number,
            "f1_macro": trial.user_attrs.get("primary", trial.value),
            "ece": trial.user_attrs.get("ece", None),
            "logloss": trial.user_attrs.get("logloss", None),
            "config": config,
        }
        configs_export.append(export_entry)

    # Write to file
    outfile = outdir / f"{agent}_stage_a_top{topk}.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("w") as f:
        json.dump(
            {
                "agent": agent,
                "stage": "stage_a",
                "topk": topk,
                "total_trials": len(trials),
                "configurations": configs_export,
            },
            f,
            indent=2,
        )

    LOGGER.info("Exported top-%d configurations to %s", topk, outfile)
    return outfile


def main() -> None:
    """Main entry point for Stage-A baseline exploration."""
    args = parse_args()
    setup_logging(args.log_level)

    # Resolve paths and settings
    storage = resolve_storage(args.storage or "sqlite:///supermax_stage_a.db")
    profile = resolve_profile(args.profile)
    study_name = args.study_name or f"supermax-{args.agent}-stage-a"
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    LOGGER.info("=" * 80)
    LOGGER.info("SUPERMAX STAGE-A: Baseline Exploration")
    LOGGER.info("=" * 80)
    LOGGER.info("Agent: %s", args.agent)
    LOGGER.info("Study: %s", study_name)
    LOGGER.info("Trials: %d", args.trials)
    LOGGER.info("Epochs: %d (patience=%d)", args.epochs, args.patience)
    LOGGER.info("Seeds: %s", seeds)
    LOGGER.info("Max samples: %s", args.max_samples or "unlimited")
    LOGGER.info("Top-K export: %d", args.topk)
    LOGGER.info("Storage: %s", storage)
    LOGGER.info("Output: %s", outdir)
    LOGGER.info("=" * 80)

    # Create TPE sampler (efficient for large search spaces)
    sampler = create_sampler(
        multi_objective=False,
        seed=args.sampler_seed,
        sampler="tpe",
    )
    LOGGER.info("Sampler: TPE (seed=%d)", args.sampler_seed)

    # Create ASHA pruner (aggressive early stopping)
    pruner = create_pruner(
        "asha",
        min_resource=1,
        max_resource=max(1, args.epochs),
        reduction_factor=3,
    )
    LOGGER.info("Pruner: ASHA (min_resource=1, max_resource=%d)", args.epochs)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize"],  # Maximize F1-macro
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        LOGGER.info("Loaded existing study with %d trials", n_existing)

    # Create search space (no constraints for Stage-A - full exploration)
    search_space = SearchSpace(args.agent)
    LOGGER.info("Search space: 8 architectures, 6 optimizers, 6 aug params")

    # Create objective settings
    settings = ObjectiveSettings(
        agent=args.agent,
        study=study_name,
        outdir=outdir,
        epochs=args.epochs,
        seeds=seeds,
        patience=args.patience,
        max_samples=args.max_samples,
        multi_objective=False,
        topk=args.topk,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=f"{profile}-{args.agent}-stage-a",
    )

    # Create objective function (no constraints for baseline exploration)
    objective = ObjectiveBuilder(
        search_space,
        settings,
        constraints=None,  # Full search space
    )

    # Run optimization
    timeout_seconds = int(args.timeout_hours * 3600) if args.timeout_hours else None

    LOGGER.info("Starting optimization...")
    LOGGER.info(
        "Expected runtime: ~%d hours (%.1f min/trial × %d trials)",
        int(args.trials * 5 / 60),  # Rough estimate: 5 min/trial
        5.0,
        args.trials,
    )

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
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    LOGGER.info("=" * 80)
    LOGGER.info("STAGE-A COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info("Total trials: %d", len(study.trials))
    LOGGER.info("  - Completed: %d", len(completed_trials))
    LOGGER.info("  - Pruned: %d", len(pruned_trials))
    LOGGER.info("  - Failed: %d", len(failed_trials))

    if completed_trials:
        best_trial = study.best_trial
        best_f1 = best_trial.user_attrs.get("primary", best_trial.value)
        LOGGER.info("Best trial: #%d", best_trial.number)
        LOGGER.info("  F1-macro: %.4f", best_f1)
        LOGGER.info("  ECE: %.4f", best_trial.user_attrs.get("ece", 0.0))
        LOGGER.info("  Logloss: %.4f", best_trial.user_attrs.get("logloss", 0.0))

    # Export top-K configurations for Stage-B
    export_path = export_topk_configs(study, args.topk, outdir, args.agent)
    LOGGER.info("Top-%d configurations exported to: %s", args.topk, export_path)

    # Generate summary report
    report_path = outdir / f"{args.agent}_stage_a_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("SUPERMAX STAGE-A REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Agent: {args.agent}\n")
        f.write(f"Study: {study_name}\n")
        f.write(f"Total trials: {len(study.trials)}\n")
        f.write(f"  - Completed: {len(completed_trials)}\n")
        f.write(f"  - Pruned: {len(pruned_trials)}\n")
        f.write(f"  - Failed: {len(failed_trials)}\n")
        f.write("\n")

        if completed_trials:
            f.write(f"Best trial: #{best_trial.number}\n")
            f.write(f"  F1-macro: {best_f1:.4f}\n")
            f.write(f"  ECE: {best_trial.user_attrs.get('ece', 0.0):.4f}\n")
            f.write(
                f"  Logloss: {best_trial.user_attrs.get('logloss', 0.0):.4f}\n"
            )
            f.write("\n")

            # Top-10 summary
            top10 = sorted(
                completed_trials,
                key=lambda t: -(t.user_attrs.get("primary", t.value or 0.0)),
            )[:10]
            f.write("Top-10 Trials:\n")
            for i, trial in enumerate(top10, 1):
                f1 = trial.user_attrs.get("primary", trial.value)
                f.write(
                    f"  {i:2d}. Trial #{trial.number:4d}: F1={f1:.4f}\n"
                )

        f.write("\n")
        f.write(f"Top-{args.topk} configurations: {export_path}\n")
        f.write(f"Next step: Run Stage-B with exported configurations\n")

    LOGGER.info("Summary report: %s", report_path)
    LOGGER.info("=" * 80)
    LOGGER.info("Stage-A complete! Proceed to Stage-B for multi-objective optimization.")


if __name__ == "__main__":
    main()
