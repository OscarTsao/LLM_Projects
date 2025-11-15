#!/usr/bin/env python
"""SUPERMAX Phase 10: Warm-Start HPO Studies.

Create new HPO studies initialized with knowledge from previous runs.
This can significantly speed up convergence by starting from good configurations.

Usage:
    # Warm-start from a single source study
    python scripts/warm_start_hpo.py \
        --new-study evidence-warm-started \
        --source-study criteria-maximal-hpo \
        --n-configs 10

    # Warm-start from multiple source studies
    python scripts/warm_start_hpo.py \
        --new-study joint-warm-started \
        --source-studies criteria-maximal-hpo evidence-maximal-hpo \
        --n-configs-per-source 5

    # Transfer learning from related task
    python scripts/warm_start_hpo.py \
        --new-study evidence-transferred \
        --source-task criteria \
        --source-study criteria-maximal-hpo \
        --target-task evidence \
        --n-configs 10

    # Adaptive warm-start (auto-select similar studies)
    python scripts/warm_start_hpo.py \
        --new-study share-adaptive \
        --adaptive \
        --candidate-studies criteria-maximal-hpo evidence-maximal-hpo \
        --n-configs 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from psy_agents_noaug.hpo.meta_learning import (
    AdaptiveWarmStart,
    TransferLearner,
    WarmStartStrategy,
)

LOGGER = logging.getLogger("warm_start_hpo")


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SUPERMAX Warm-Start HPO - Create studies initialized with prior knowledge"
    )

    # New study
    parser.add_argument(
        "--new-study",
        type=str,
        required=True,
        help="Name for new study to create",
    )
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default="minimize",
        help="Optimization direction (default: minimize)",
    )

    # Storage
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna.db",
        help="Optuna storage URL (default: sqlite:///optuna.db)",
    )

    # Source studies
    parser.add_argument(
        "--source-study",
        type=str,
        help="Single source study to warm-start from",
    )
    parser.add_argument(
        "--source-studies",
        nargs="+",
        help="Multiple source studies to warm-start from",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=10,
        help="Number of configs to enqueue from source (default: 10)",
    )
    parser.add_argument(
        "--n-configs-per-source",
        type=int,
        default=5,
        help="Configs per source when using multiple sources (default: 5)",
    )

    # Warm-start strategy
    parser.add_argument(
        "--strategy",
        choices=["best", "diverse", "importance_weighted"],
        default="best",
        help="Config selection strategy (default: best)",
    )

    # Transfer learning
    parser.add_argument(
        "--source-task",
        type=str,
        choices=["criteria", "evidence", "share", "joint"],
        help="Source task for transfer learning",
    )
    parser.add_argument(
        "--target-task",
        type=str,
        choices=["criteria", "evidence", "share", "joint"],
        help="Target task for transfer learning",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["shared_only", "confident", "all"],
        default="confident",
        help="Parameter transfer mode (default: confident)",
    )

    # Adaptive warm-start
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive warm-start (auto-select similar studies)",
    )
    parser.add_argument(
        "--candidate-studies",
        nargs="+",
        help="Candidate studies for adaptive selection",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity for adaptive selection (default: 0.5)",
    )

    # Dry-run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating study",
    )

    # Other options
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.dry_run:
        LOGGER.info("DRY RUN MODE - No study will be created")

    # Validate arguments
    if args.adaptive:
        if not args.candidate_studies:
            LOGGER.error("--adaptive requires --candidate-studies")
            return

        LOGGER.info(
            "Using adaptive warm-start with %d candidates", len(args.candidate_studies)
        )

        if args.dry_run:
            print(f"\nWould create study: {args.new_study}")
            print(f"Adaptive warm-start from candidates: {args.candidate_studies}")
            print(f"Similarity threshold: {args.similarity_threshold}")
            print(f"Total configs: {args.n_configs}")
            return

        # Create adaptive warm-start
        adaptive = AdaptiveWarmStart(
            storage=args.storage,
            similarity_threshold=args.similarity_threshold,
        )

        # Create new study
        import optuna

        new_study = optuna.create_study(
            study_name=args.new_study,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=False,
        )

        # Adaptive warm-start
        n_enqueued = adaptive.adaptive_warm_start(
            target_study=new_study,
            candidate_studies=args.candidate_studies,
            total_configs=args.n_configs,
            max_sources=3,
        )

        print(
            f"\n✅ Created study '{args.new_study}' with {n_enqueued} configs enqueued"
        )
        print(f"   Storage: {args.storage}")

    elif args.source_task and args.target_task:
        # Transfer learning mode
        if not args.source_study:
            LOGGER.error("Transfer learning requires --source-study")
            return

        LOGGER.info(
            "Transfer learning from %s to %s", args.source_task, args.target_task
        )

        if args.dry_run:
            print(f"\nWould create study: {args.new_study}")
            print(f"Transfer from task: {args.source_task} → {args.target_task}")
            print(f"Source study: {args.source_study}")
            print(f"Transfer mode: {args.transfer_mode}")
            print(f"Configs: {args.n_configs}")
            return

        # Create transfer learner
        transfer = TransferLearner(storage=args.storage)

        # Create new study
        import optuna

        new_study = optuna.create_study(
            study_name=args.new_study,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=False,
        )

        # Transfer knowledge
        n_transferred = transfer.transfer_from_task(
            target_study=new_study,
            source_task=args.source_task,
            target_task=args.target_task,
            source_study=args.source_study,
            n_configs=args.n_configs,
            transfer_mode=args.transfer_mode,
        )

        print(
            f"\n✅ Created study '{args.new_study}' with {n_transferred} configs transferred"
        )
        print(f"   Source: {args.source_task} → {args.target_task}")
        print(f"   Storage: {args.storage}")

    elif args.source_studies:
        # Multiple source warm-start
        LOGGER.info("Warm-starting from %d sources", len(args.source_studies))

        if args.dry_run:
            print(f"\nWould create study: {args.new_study}")
            print(f"Warm-start from {len(args.source_studies)} sources:")
            for source in args.source_studies:
                print(f"  - {source} ({args.n_configs_per_source} configs)")
            return

        # Create warm-start strategy
        warm_starter = WarmStartStrategy(storage=args.storage)

        # Create new study with warm-start
        import optuna

        new_study = optuna.create_study(
            study_name=args.new_study,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=False,
        )

        # Warm-start from each source
        total_enqueued = 0
        for source_study in args.source_studies:
            n_enqueued = warm_starter.warm_start_from_study(
                target_study=new_study,
                source_study=source_study,
                n_configs=args.n_configs_per_source,
                strategy=args.strategy,
            )
            total_enqueued += n_enqueued

        print(
            f"\n✅ Created study '{args.new_study}' with {total_enqueued} configs enqueued"
        )
        print(f"   Sources: {len(args.source_studies)}")
        print(f"   Storage: {args.storage}")

    elif args.source_study:
        # Single source warm-start
        LOGGER.info("Warm-starting from: %s", args.source_study)

        if args.dry_run:
            print(f"\nWould create study: {args.new_study}")
            print(f"Warm-start from: {args.source_study}")
            print(f"Strategy: {args.strategy}")
            print(f"Configs: {args.n_configs}")
            return

        # Create warm-start strategy
        warm_starter = WarmStartStrategy(storage=args.storage)

        # Create new study
        import optuna

        new_study = optuna.create_study(
            study_name=args.new_study,
            direction=args.direction,
            storage=args.storage,
            load_if_exists=False,
        )

        # Warm-start
        n_enqueued = warm_starter.warm_start_from_study(
            target_study=new_study,
            source_study=args.source_study,
            n_configs=args.n_configs,
            strategy=args.strategy,
        )

        print(
            f"\n✅ Created study '{args.new_study}' with {n_enqueued} configs enqueued"
        )
        print(f"   Source: {args.source_study}")
        print(f"   Strategy: {args.strategy}")
        print(f"   Storage: {args.storage}")

    else:
        LOGGER.error(
            "Must specify --source-study, --source-studies, or --adaptive with --candidate-studies"
        )
        return

    if not args.dry_run:
        print("\nNext steps:")
        print("1. Run HPO on the new study using your HPO runner")
        print("2. The warm-start configs will be evaluated first")
        print("3. Compare performance with baseline (non-warm-started) runs")


if __name__ == "__main__":
    main()
