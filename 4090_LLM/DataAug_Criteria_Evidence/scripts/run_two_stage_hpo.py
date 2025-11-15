#!/usr/bin/env python3
"""
Run two-stage HPO: Stage A (baseline) → Stage B (augmentation).

This script orchestrates the complete HPO workflow:
1. Stage A: Optimize model architecture and training hyperparameters (no augmentation)
2. Export best baseline configuration
3. Stage B: Optimize augmentation parameters using best baseline model
4. Compare Stage A vs Stage B results

Usage:
    # Run both stages sequentially
    python scripts/run_two_stage_hpo.py --task criteria

    # Run Stage A only
    python scripts/run_two_stage_hpo.py --task criteria --stage-a-only

    # Run Stage B only (requires Stage A completion)
    python scripts/run_two_stage_hpo.py --task criteria --stage-b-only

    # Custom trial counts
    python scripts/run_two_stage_hpo.py --task criteria --stage-a-trials 25 --stage-b-trials 50
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run shell command and check for errors."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        print(f"   Exit code: {result.returncode}")
        return False

    print(f"\n✅ SUCCESS: {description}")
    return True


def check_stage_a_complete():
    """Check if Stage A has completed successfully."""
    best_config = Path("outputs/hpo_stage_a/best_config.yaml")
    return best_config.exists()


def run_stage_a(task: str, n_trials: int = 50):
    """Run Stage A: Baseline model optimization."""

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--config-name=stage_a_baseline",
        f"task={task}",
        f"hpo.n_trials={n_trials}",
    ]

    success = run_command(cmd, "Stage A: Baseline Model Optimization")

    if success:
        # Export best config
        export_cmd = [
            "python",
            "scripts/export_best_config.py",
            "--study-name",
            "stage_a_baseline",
            "--output",
            "outputs/hpo_stage_a/best_config.yaml",
        ]
        run_command(export_cmd, "Exporting Stage A Best Configuration")

    return success


def run_stage_b(task: str, n_trials: int = 100):
    """Run Stage B: Augmentation optimization."""

    if not check_stage_a_complete():
        print("\n❌ ERROR: Stage A must complete first")
        print("   Missing: outputs/hpo_stage_a/best_config.yaml")
        return False

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--config-name=stage_b_augmentation",
        f"task={task}",
        f"hpo.n_trials={n_trials}",
        "--baseline-config",
        "outputs/hpo_stage_a/best_config.yaml",
    ]

    success = run_command(cmd, "Stage B: Augmentation Optimization")

    if success:
        # Export best config
        export_cmd = [
            "python",
            "scripts/export_best_config.py",
            "--study-name",
            "stage_b_augmentation",
            "--output",
            "outputs/hpo_stage_b/best_config.yaml",
        ]
        run_command(export_cmd, "Exporting Stage B Best Configuration")

    return success


def compare_stages():
    """Compare Stage A vs Stage B results."""

    stage_a_config = Path("outputs/hpo_stage_a/best_config.yaml")
    stage_b_config = Path("outputs/hpo_stage_b/best_config.yaml")

    if not (stage_a_config.exists() and stage_b_config.exists()):
        print("\n⚠️  WARNING: Cannot compare stages - missing best configs")
        return

    cmd = [
        "python",
        "scripts/compare_hpo_stages.py",
        "--stage-a",
        str(stage_a_config),
        "--stage-b",
        str(stage_b_config),
    ]

    run_command(cmd, "Comparing Stage A vs Stage B Results")


def main():
    parser = argparse.ArgumentParser(
        description="Run two-stage HPO (baseline + augmentation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        choices=["criteria", "evidence"],
        required=True,
        help="Task to optimize",
    )
    parser.add_argument("--stage-a-only", action="store_true", help="Run Stage A only")
    parser.add_argument(
        "--stage-b-only",
        action="store_true",
        help="Run Stage B only (requires Stage A completion)",
    )
    parser.add_argument(
        "--stage-a-trials", type=int, default=50, help="Number of trials for Stage A"
    )
    parser.add_argument(
        "--stage-b-trials", type=int, default=100, help="Number of trials for Stage B"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip comparison after both stages complete",
    )

    args = parser.parse_args()

    print(
        f"""
╔═══════════════════════════════════════════════════════════╗
║  Two-Stage HPO: Baseline → Augmentation                  ║
╚═══════════════════════════════════════════════════════════╝

Task: {args.task}
Stage A Trials: {args.stage_a_trials}
Stage B Trials: {args.stage_b_trials}
"""
    )

    # Run Stage A
    if not args.stage_b_only:
        if not run_stage_a(args.task, args.stage_a_trials):
            print("\n❌ Stage A failed - aborting")
            sys.exit(1)

        if args.stage_a_only:
            print("\n✅ Stage A complete (--stage-a-only specified)")
            sys.exit(0)

    # Run Stage B
    if not args.stage_a_only and not run_stage_b(args.task, args.stage_b_trials):
        print("\n❌ Stage B failed")
        sys.exit(1)

    # Compare results
    if not args.skip_comparison:
        compare_stages()

    print(
        """
\n╔═══════════════════════════════════════════════════════════╗
║  Two-Stage HPO Complete!                                  ║
╚═══════════════════════════════════════════════════════════╝

Next Steps:
1. Review comparison report: outputs/hpo_comparison_report.txt
2. Check best configs:
   - Stage A: outputs/hpo_stage_a/best_config.yaml
   - Stage B: outputs/hpo_stage_b/best_config.yaml
3. Train final model with best config:
   python -m psy_agents_noaug.cli train --config outputs/hpo_stage_b/best_config.yaml
"""
    )


if __name__ == "__main__":
    main()
