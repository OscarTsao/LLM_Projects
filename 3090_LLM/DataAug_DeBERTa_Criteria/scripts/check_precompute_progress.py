#!/usr/bin/env python3
"""Check pre-computation progress and estimate completion time."""

import json
import pickle
import sys
import time
from pathlib import Path

def check_progress():
    """Check and display pre-computation progress."""
    cache_path = Path("experiments/augmentation_cache.pkl")
    checkpoint_path = Path("experiments/augmentation_cache.checkpoint.pkl")
    meta_path = Path("experiments/augmentation_cache.json")

    print("=" * 70)
    print("PRE-COMPUTATION PROGRESS CHECK")
    print("=" * 70)
    print()

    # Check if completed
    if cache_path.exists() and meta_path.exists():
        print("✓✓✓ PRE-COMPUTATION COMPLETED! ✓✓✓")
        print()

        # Load metadata
        with meta_path.open() as f:
            meta = json.load(f)

        print(f"Unique texts: {meta['num_texts']}")
        print(f"Methods: {meta['num_methods']}")
        print(f"Total cached: {meta['total_cached']:,}")
        print(f"Success: {meta['stats']['success']:,}")
        print(f"Failed: {meta['stats']['failed']:,}")
        print(f"Cache size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
        print()

        if meta['stats']['failed_methods']:
            print("Methods with failures:")
            for method, count in sorted(meta['stats']['failed_methods'].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {method}: {count}")

        print()
        print("✓ Ready to start HPO with cached augmentations!")
        return True

    # Check checkpoint
    if checkpoint_path.exists():
        print("⏳ Pre-computation in progress (checkpoint found)")
        print()

        try:
            with checkpoint_path.open("rb") as f:
                cache = pickle.load(f)

            cached_count = len(cache)

            # Estimate total work
            # Assuming ~800 texts × 25 methods = 20,000 items
            estimated_total = 20000
            progress_pct = (cached_count / estimated_total) * 100

            print(f"Cached items: {cached_count:,} / ~{estimated_total:,}")
            print(f"Progress: {progress_pct:.1f}%")
            print(f"Checkpoint size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
            print()

            # Estimate remaining time (rough)
            if cached_count > 100:
                # Assume checkpoint is saved every 2,500 items (100 texts × 25 methods)
                # and each checkpoint takes ~5-10 minutes
                checkpoints_done = cached_count / 2500
                checkpoints_remaining = (estimated_total - cached_count) / 2500
                minutes_remaining = checkpoints_remaining * 7  # avg 7 min per checkpoint

                print(f"Estimated time remaining: {minutes_remaining:.0f} minutes")

        except Exception as e:
            print(f"Could not read checkpoint: {e}")
    else:
        print("⏳ Pre-computation starting...")
        print("   Checkpoint will be created after first 100 texts processed")

    print()
    print("Process status:")
    import subprocess
    result = subprocess.run(
        ["docker", "exec", "75bb13cca2c5", "pgrep", "-f", "precompute"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✓ Process is running")
        pids = result.stdout.strip().split('\n')
        print(f"  ✓ {len(pids)} worker processes")
    else:
        print("  ✗ Process is NOT running")
        print("  Check logs: docker exec 75bb13cca2c5 tail /tmp/precompute_parallel.log")

    print()
    return False

if __name__ == "__main__":
    check_progress()
