#!/usr/bin/env python3
"""
Profile individual augmentation methods.

Measures:
- Average time per augmentation (ms)
- Standard deviation
- Throughput (samples/sec)

Tests all 12 production-ready augmenters with representative evidence texts.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.augmentation import (
    ALL_METHODS,
    NLPAUG_METHODS,
    REGISTRY,
    TEXTATTACK_METHODS,
    AugResources,
)

# Representative evidence texts from clinical domain
SAMPLE_TEXTS = [
    "Patient reports anxiety symptoms.",
    "Depression and mood disturbances noted.",
    "Severe anxiety and panic attacks reported.",
    "No significant psychiatric history documented.",
    "Patient appears stable with minimal symptoms.",
    "Clinical evaluation shows persistent sadness.",
    "Treatment plan includes therapy sessions.",
    "Medication management for mood stabilization.",
    "Patient demonstrates improved coping skills.",
    "Symptoms include difficulty concentrating and sleep disturbance.",
]


def profile_augmenter(
    method_name: str,
    num_samples: int = 1000,
    resources: AugResources | None = None,
) -> dict[str, Any]:
    """
    Profile a single augmentation method.

    Args:
        method_name: Name of the augmentation method
        num_samples: Number of samples to process
        resources: Optional augmentation resources

    Returns:
        Dictionary with profiling metrics
    """
    if method_name not in REGISTRY:
        raise ValueError(f"Unknown augmentation method: {method_name}")

    # Handle special methods that require resources
    if method_name == "nlpaug/word/TfIdfAug":
        if resources is None or resources.tfidf_model_path is None:
            return {
                "method": method_name,
                "status": "skipped",
                "reason": "requires tfidf_model_path",
            }

    if method_name == "nlpaug/word/ReservedAug":
        if resources is None or resources.reserved_map_path is None:
            return {
                "method": method_name,
                "status": "skipped",
                "reason": "requires reserved_map_path",
            }

    try:
        # Get the augmenter factory
        entry = REGISTRY[method_name]

        # Build kwargs
        kwargs = {}
        if method_name == "nlpaug/word/TfIdfAug" and resources:
            kwargs["model_path"] = resources.tfidf_model_path
        elif method_name == "nlpaug/word/ReservedAug" and resources:
            kwargs["reserved_map_path"] = resources.reserved_map_path

        # Create augmenter
        augmenter = entry.factory(**kwargs)

        # Warmup
        for text in SAMPLE_TEXTS[:5]:
            _ = augmenter.augment_one(text)

        # Profile
        times = []
        texts_cycled = SAMPLE_TEXTS * (num_samples // len(SAMPLE_TEXTS) + 1)

        for i in range(num_samples):
            text = texts_cycled[i]

            start = time.perf_counter()
            _ = augmenter.augment_one(text)
            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1000 / avg_time if avg_time > 0 else 0  # samples/sec

        # Determine status
        status = "✅" if avg_time < 10 else "⚠️" if avg_time < 50 else "❌"

        return {
            "method": method_name,
            "status": "success",
            "avg_time_ms": float(avg_time),
            "std_time_ms": float(std_time),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times)),
            "median_time_ms": float(np.median(times)),
            "throughput_samples_per_sec": float(throughput),
            "num_samples": num_samples,
            "performance_status": status,
        }

    except Exception as e:
        return {
            "method": method_name,
            "status": "error",
            "error": str(e),
        }


def profile_all_augmenters(
    num_samples: int = 1000,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Profile all augmentation methods.

    Args:
        num_samples: Number of samples per method
        output_csv: Optional path for CSV output
        output_json: Optional path for JSON output

    Returns:
        List of profiling results
    """
    print("=" * 80)
    print("Augmentation Method Profiling")
    print("=" * 80)
    print(f"Samples per method: {num_samples}")
    print(f"Total methods: {len(ALL_METHODS)}")
    print()

    # Create resources (empty for now, methods requiring resources will be skipped)
    resources = AugResources()

    results = []

    # Group methods by library
    nlpaug_methods = [m for m in ALL_METHODS if m in NLPAUG_METHODS]
    textattack_methods = [m for m in ALL_METHODS if m in TEXTATTACK_METHODS]

    print("NLPAUG Methods:")
    print("-" * 80)

    for method in nlpaug_methods:
        method_short = method.split("/")[-1]
        print(f"Profiling {method_short}...", end=" ", flush=True)

        result = profile_augmenter(method, num_samples, resources)
        results.append(result)

        if result["status"] == "success":
            print(
                f"{result['avg_time_ms']:.2f}ms "
                f"({result['throughput_samples_per_sec']:.1f} samples/sec) "
                f"{result['performance_status']}"
            )
        elif result["status"] == "skipped":
            print(f"SKIPPED ({result['reason']})")
        else:
            print(f"ERROR: {result.get('error', 'Unknown')}")

    print()
    print("TextAttack Methods:")
    print("-" * 80)

    for method in textattack_methods:
        method_short = method.split("/")[-1]
        print(f"Profiling {method_short}...", end=" ", flush=True)

        result = profile_augmenter(method, num_samples, resources)
        results.append(result)

        if result["status"] == "success":
            print(
                f"{result['avg_time_ms']:.2f}ms "
                f"({result['throughput_samples_per_sec']:.1f} samples/sec) "
                f"{result['performance_status']}"
            )
        elif result["status"] == "skipped":
            print(f"SKIPPED ({result['reason']})")
        else:
            print(f"ERROR: {result.get('error', 'Unknown')}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Method':<35} {'Avg Time (ms)':>15} {'Std Dev':>10} {'Samples/sec':>12} {'Status':>8}"
    )
    print("-" * 80)

    for result in results:
        if result["status"] != "success":
            continue

        method_short = (
            result["method"].split("/", 1)[1]
            if "/" in result["method"]
            else result["method"]
        )
        avg_time = result["avg_time_ms"]
        std_dev = result["std_time_ms"]
        throughput = result["throughput_samples_per_sec"]
        status = result["performance_status"]

        print(
            f"{method_short:<35} {avg_time:>15.2f} {std_dev:>10.2f} {throughput:>12.1f} {status:>8}"
        )

    # Save CSV
    if output_csv:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Method",
                    "Library",
                    "Avg Time (ms)",
                    "Std Dev (ms)",
                    "Min Time (ms)",
                    "Max Time (ms)",
                    "Median Time (ms)",
                    "Samples/sec",
                    "Status",
                ]
            )

            for result in results:
                if result["status"] != "success":
                    continue

                lib = "nlpaug" if result["method"] in NLPAUG_METHODS else "textattack"
                method_short = (
                    result["method"].split("/", 1)[1]
                    if "/" in result["method"]
                    else result["method"]
                )

                writer.writerow(
                    [
                        method_short,
                        lib,
                        f"{result['avg_time_ms']:.3f}",
                        f"{result['std_time_ms']:.3f}",
                        f"{result['min_time_ms']:.3f}",
                        f"{result['max_time_ms']:.3f}",
                        f"{result['median_time_ms']:.3f}",
                        f"{result['throughput_samples_per_sec']:.2f}",
                        result["performance_status"],
                    ]
                )

        print(f"\nCSV results saved to: {output_csv}")

    # Save JSON
    if output_json:
        output_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": num_samples,
            "results": results,
        }
        output_json.write_text(json.dumps(output_data, indent=2))
        print(f"JSON results saved to: {output_json}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile augmentation methods")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples per method (default: 1000)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("profiling_augmentation_results.csv"),
        help="Output CSV file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("profiling_augmentation_results.json"),
        help="Output JSON file",
    )

    args = parser.parse_args()

    profile_all_augmenters(
        num_samples=args.num_samples,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
