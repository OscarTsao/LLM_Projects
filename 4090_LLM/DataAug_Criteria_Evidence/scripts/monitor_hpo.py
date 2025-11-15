#!/usr/bin/env python
"""SUPERMAX Phase 7: HPO Monitoring and Management Tool.

Real-time monitoring for running HPO studies with:
- Live progress tracking with ETA
- Health monitoring (stalled trials, failures)
- Checkpoint status
- Best trial tracking
- Graceful shutdown handler

Usage:
    # Monitor specific study
    python scripts/monitor_hpo.py \\
        --study-name supermax-criteria-stage-a \\
        --storage sqlite:///supermax_stage_a.db

    # Monitor with custom update interval
    python scripts/monitor_hpo.py \\
        --study-name supermax-criteria-stage-a \\
        --storage sqlite:///supermax_stage_a.db \\
        --update-interval 5

    # List all studies in database
    python scripts/monitor_hpo.py \\
        --storage sqlite:///supermax_stage_a.db \\
        --list-studies

    # Check health of study
    python scripts/monitor_hpo.py \\
        --study-name supermax-criteria-stage-a \\
        --storage sqlite:///supermax_stage_a.db \\
        --check-health
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path

import optuna

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from psy_agents_noaug.hpo.monitoring import (
    ProgressTracker,
    CheckpointManager,
    HealthMonitor,
    check_study_health,
)

LOGGER = logging.getLogger("monitor_hpo")

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


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
        description="SUPERMAX HPO Monitor - Real-time study monitoring"
    )
    parser.add_argument(
        "--study-name",
        help="Name of the study to monitor",
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="Optuna storage URL (e.g., sqlite:///supermax_stage_a.db)",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=10.0,
        help="Seconds between progress updates (default: 10)",
    )
    parser.add_argument(
        "--list-studies",
        action="store_true",
        help="List all studies in the database",
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Run health check and exit",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/supermax/checkpoints"),
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/supermax/logs"),
        help="Directory for progress logs",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def signal_handler(signum: int, frame: Any) -> None:
    """Handle graceful shutdown signals.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global SHUTDOWN_REQUESTED
    LOGGER.info("Shutdown signal received (signal %d)", signum)
    SHUTDOWN_REQUESTED = True


def list_studies(storage: str) -> None:
    """List all studies in the storage.

    Args:
        storage: Optuna storage URL
    """
    try:
        study_summaries = optuna.get_all_study_summaries(storage=storage)

        if not study_summaries:
            print("No studies found in database.")
            return

        print("=" * 80)
        print(f"Studies in {storage}")
        print("=" * 80)

        for summary in study_summaries:
            print(f"\nStudy: {summary.study_name}")
            print(f"  Directions: {[str(d) for d in summary.directions]}")
            print(f"  Trials: {summary.n_trials}")
            print(f"  Best value: {summary.best_trial.value if summary.best_trial else 'N/A'}")
            print(f"  Best trial: #{summary.best_trial.number if summary.best_trial else 'N/A'}")
            print(f"  Date start: {summary.datetime_start}")

        print("=" * 80)

    except Exception as e:
        LOGGER.error("Failed to list studies: %s", e)
        sys.exit(1)


def monitor_study(
    study_name: str,
    storage: str,
    update_interval: float,
    checkpoint_dir: Path,
    log_dir: Path,
) -> None:
    """Monitor study with real-time updates.

    Args:
        study_name: Name of study to monitor
        storage: Optuna storage URL
        update_interval: Seconds between updates
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """
    try:
        # Load study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        LOGGER.info("Loaded study: %s", study_name)
        LOGGER.info("Storage: %s", storage)
        LOGGER.info("Update interval: %.1fs", update_interval)

        # Create monitoring components
        log_file = log_dir / f"{study_name}_monitor.log"
        log_dir.mkdir(parents=True, exist_ok=True)

        tracker = ProgressTracker(
            study=study,
            total_trials=len(study.trials),  # Will update dynamically
            update_interval=update_interval,
            log_file=log_file,
        )

        health_monitor = HealthMonitor()

        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        LOGGER.info("Starting monitoring (press Ctrl+C to stop)...")

        # Print initial progress
        tracker.print_progress(force=True)

        iteration = 0
        while not SHUTDOWN_REQUESTED:
            # Update and print progress
            tracker.print_progress()

            # Periodic health checks (every 5 updates)
            if iteration % 5 == 0:
                status = health_monitor.check_health(study)

                if not status.is_healthy():
                    LOGGER.warning("Health check failed: %s", status.summary())
                    for error in status.errors:
                        LOGGER.error("  ✗ %s", error)

                if status.warnings:
                    for warning in status.warnings:
                        LOGGER.warning("  ⚠ %s", warning)

            iteration += 1

            # Sleep until next update
            time.sleep(update_interval)

        # Graceful shutdown
        LOGGER.info("Shutting down monitoring...")
        tracker.finalize()

    except KeyboardInterrupt:
        LOGGER.info("Monitoring interrupted by user")
    except Exception as e:
        LOGGER.error("Monitoring failed: %s", e, exc_info=True)
        sys.exit(1)


def run_health_check(study_name: str, storage: str) -> None:
    """Run one-time health check.

    Args:
        study_name: Name of study to check
        storage: Optuna storage URL
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        status = check_study_health(study, verbose=True)

        if status.is_healthy():
            LOGGER.info("✓ Study is healthy")
            sys.exit(0)
        else:
            LOGGER.error("✗ Study has health issues")
            sys.exit(1)

    except Exception as e:
        LOGGER.error("Health check failed: %s", e)
        sys.exit(1)


def main() -> None:
    """Main entry point for HPO monitoring."""
    args = parse_args()
    setup_logging(args.log_level)

    # List studies mode
    if args.list_studies:
        list_studies(args.storage)
        return

    # Require study name for other modes
    if not args.study_name:
        LOGGER.error("--study-name required (use --list-studies to see available studies)")
        sys.exit(1)

    # Health check mode
    if args.check_health:
        run_health_check(args.study_name, args.storage)
        return

    # Monitoring mode
    monitor_study(
        study_name=args.study_name,
        storage=args.storage,
        update_interval=args.update_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
