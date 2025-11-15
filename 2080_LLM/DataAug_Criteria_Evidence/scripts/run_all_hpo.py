"""
Run HPO sequentially for all architectures (criteria, evidence, share, joint).

This script provides two modes:
1. Multi-stage HPO: Runs progressive refinement (stage0 -> stage1 -> stage2 -> refit)
2. Maximal HPO: Runs single large optimization with tune_max.py

Usage:
    # Multi-stage HPO for all architectures
    python scripts/run_all_hpo.py --mode multistage

    # Maximal HPO for all architectures
    python scripts/run_all_hpo.py --mode maximal

    # Run specific architectures only
    python scripts/run_all_hpo.py --mode multistage --architectures criteria evidence

    # Custom HPO settings
    python scripts/run_all_hpo.py --mode maximal --n-trials 500 --parallel 2
"""

import argparse
import subprocess
import sys
import time


def run_command(cmd: list[str], description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run as list of strings
        description: Human-readable description for logging

    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(
            f"\n✗ {description} failed after {elapsed:.1f}s with exit code {e.returncode}"
        )
        return False


def run_multistage_hpo(
    architecture: str,
    model: str = "roberta_base",
    skip_sanity: bool = False,
) -> bool:
    """
    Run multi-stage HPO for a single architecture.

    Args:
        architecture: One of: criteria, evidence, share, joint
        model: Model name (default: roberta_base)
        skip_sanity: Skip stage 0 sanity check

    Returns:
        True if all stages succeeded, False otherwise
    """
    print(f"\n{'#' * 80}")
    print(f"# Multi-Stage HPO for {architecture.upper()}")
    print(f"{'#' * 80}\n")

    stages = [
        ("hpo-s0", "Stage 0: Sanity Check"),
        ("hpo-s1", "Stage 1: Coarse Search"),
        ("hpo-s2", "Stage 2: Fine Search"),
        ("refit", "Stage 3: Refit on train+val"),
    ]

    if skip_sanity:
        stages = stages[1:]

    for make_target, description in stages:
        cmd = [
            "make",
            make_target,
            f"HPO_TASK={architecture}",
            f"HPO_MODEL={model}",
        ]

        if not run_command(cmd, f"{architecture} - {description}"):
            print(f"\n✗ Failed at {description} for {architecture}")
            return False

    print(f"\n✓ All stages completed for {architecture}")
    return True


def run_maximal_hpo(
    architecture: str,
    n_trials: int = None,
    parallel: int = 1,
    timeout: int = None,
    outdir: str = None,
) -> bool:
    """
    Run maximal HPO for a single architecture using tune_max.py.

    Args:
        architecture: One of: criteria, evidence, share, joint
        n_trials: Number of trials (default varies by architecture)
        parallel: Number of parallel jobs
        timeout: Timeout in seconds
        outdir: Output directory

    Returns:
        True if succeeded, False otherwise
    """
    print(f"\n{'#' * 80}")
    print(f"# Maximal HPO for {architecture.upper()}")
    print(f"{'#' * 80}\n")

    # Default trial counts per architecture (from Makefile)
    default_trials = {
        "criteria": 800,
        "evidence": 1200,
        "share": 600,
        "joint": 600,
    }

    n_trials = n_trials or default_trials.get(architecture, 200)

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        architecture,
        "--study",
        f"noaug-{architecture}-max",
        "--n-trials",
        str(n_trials),
        "--parallel",
        str(parallel),
    ]

    if outdir:
        cmd.extend(["--outdir", outdir])
    else:
        cmd.extend(["--outdir", "./_runs"])

    if timeout:
        cmd.extend(["--timeout", str(timeout)])

    return run_command(cmd, f"{architecture} - Maximal HPO ({n_trials} trials)")


def main():
    parser = argparse.ArgumentParser(
        description="Run HPO sequentially for all architectures"
    )
    parser.add_argument(
        "--mode",
        choices=["multistage", "maximal"],
        required=True,
        help="HPO mode: multistage (progressive) or maximal (single large run)",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["criteria", "evidence", "share", "joint"],
        choices=["criteria", "evidence", "share", "joint"],
        help="Architectures to run HPO for (default: all)",
    )
    parser.add_argument(
        "--model",
        default="roberta_base",
        help="Model for multi-stage HPO (default: roberta_base)",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip stage 0 sanity check in multi-stage mode",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Number of trials for maximal mode (default varies by architecture)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs for maximal mode (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for maximal mode",
    )
    parser.add_argument(
        "--outdir",
        help="Output directory for maximal mode (default: ./_runs)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution if any architecture fails",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PSY Agents NO-AUG - Sequential HPO Runner")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Architectures: {', '.join(args.architectures)}")
    print("=" * 80)

    results = {}
    start_time = time.time()

    for arch in args.architectures:
        if args.mode == "multistage":
            success = run_multistage_hpo(
                architecture=arch,
                model=args.model,
                skip_sanity=args.skip_sanity,
            )
        elif args.mode == "maximal":
            success = run_maximal_hpo(
                architecture=arch,
                n_trials=args.n_trials,
                parallel=args.parallel,
                timeout=args.timeout,
                outdir=args.outdir,
            )
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)

        results[arch] = success

        if not success and args.stop_on_error:
            print(f"\n✗ Stopping due to error in {arch}")
            break

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for arch, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{arch:15s} {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\nCompleted: {successful}/{total} architectures")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print("=" * 80)

    # Exit with error if any failed
    if successful < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
