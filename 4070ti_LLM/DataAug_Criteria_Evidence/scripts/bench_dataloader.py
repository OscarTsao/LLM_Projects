#!/usr/bin/env python3
"""
Comprehensive DataLoader performance benchmark with augmentation support.

Measures:
- Samples per second
- Data loading time per batch
- Step time per batch (with dummy forward pass)
- Data/step time ratio

Tests multiple configurations:
- num_workers: 0, 1, 4, 8
- augmentation: enabled/disabled
- batch_size: 16, 32
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline
from psy_agents_noaug.data.datasets import (
    ClassificationDataset,
    create_classification_collate,
)


def create_dummy_dataframe(size: int = 1000) -> pd.DataFrame:
    """Create dummy dataframe for benchmarking."""
    texts = [
        f"Patient {i} reports anxiety and depression symptoms with mood disturbances. "
        f"Clinical evaluation shows persistent feelings of sadness and worry. "
        f"Treatment plan includes therapy and medication management."
        for i in range(size)
    ]

    return pd.DataFrame(
        {
            "post_id": [f"post_{i}" for i in range(size)],
            "input_text": texts,
            "criterion_text": [
                "Criterion A: Persistent depressed mood or loss of interest in activities."
            ]
            * size,
            "label": np.random.randint(0, 2, size),
            "criterion_index": [0] * size,
        }
    )


def benchmark_dataloader(
    batch_size: int = 32,
    num_workers: int = 4,
    num_batches: int = 100,
    with_augmentation: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Benchmark DataLoader performance.

    Args:
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        num_batches: Number of batches to benchmark
        with_augmentation: Enable augmentation
        device: Device for model simulation

    Returns:
        Dictionary with benchmark metrics
    """
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create augmenter if needed
    augmenter = None
    if with_augmentation:
        aug_config = AugConfig(
            enabled=True,
            methods=[
                "nlpaug/word/SynonymAug(wordnet)",
                "nlpaug/word/RandomWordAug",
            ],
            p_apply=0.5,
            ops_per_sample=1,
            max_replace=0.3,
            seed=42,
        )
        augmenter = AugmenterPipeline(aug_config)

    # Create dataset
    df = create_dummy_dataframe(size=batch_size * (num_batches + 20))

    dataset = ClassificationDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=128,
        text_column="input_text",
        text_pair_column="criterion_text",
        label_column="label",
        augmenter=augmenter,
        lazy_encode=with_augmentation,
    )

    # Create collate function if using augmentation
    collate_fn = None
    if with_augmentation:
        collate_fn = create_classification_collate(
            tokenizer=tokenizer,
            max_length=128,
            has_text_pair=True,
            augmenter=augmenter,
        )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    # Warmup
    print("  Warming up (10 batches)...")
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break

    # Benchmark
    data_times = []
    step_times = []
    total_samples = 0

    print(f"  Benchmarking ({num_batches} batches)...")
    batch_start = time.perf_counter()

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        data_end = time.perf_counter()
        data_time = data_end - batch_start

        # Simulate model forward pass
        step_start = time.perf_counter()
        inputs = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if device == "cuda" and torch.cuda.is_available():
            inputs = inputs.cuda()
            attention_mask = attention_mask.cuda()
            # Simulate forward pass
            _ = inputs.float().mean()
            torch.cuda.synchronize()
        else:
            # Simulate forward pass
            _ = inputs.float().mean()

        step_time = time.perf_counter() - step_start

        data_times.append(data_time * 1000)  # Convert to ms
        step_times.append(step_time * 1000)
        total_samples += len(batch["input_ids"])

        # Start timing for next batch's data loading
        batch_start = time.perf_counter()

    total_time = (sum(data_times) + sum(step_times)) / 1000  # Convert to seconds

    return {
        "data_time_mean_ms": float(np.mean(data_times)),
        "data_time_std_ms": float(np.std(data_times)),
        "step_time_mean_ms": float(np.mean(step_times)),
        "step_time_std_ms": float(np.std(step_times)),
        "data_step_ratio": float(
            np.mean(data_times) / np.mean(step_times) if np.mean(step_times) > 0 else 0
        ),
        "throughput_samples_per_sec": float(total_samples / total_time),
        "total_samples": total_samples,
        "total_time_sec": total_time,
    }


def run_benchmark_suite(output_path: Path | None = None) -> dict[str, Any]:
    """Run comprehensive benchmark suite."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("DataLoader Performance Benchmark Suite")
    print("=" * 80)
    print(f"Device: {device.upper()}")
    print()

    configurations = [
        # Test num_workers without augmentation
        {"batch_size": 32, "num_workers": 0, "with_augmentation": False},
        {"batch_size": 32, "num_workers": 1, "with_augmentation": False},
        {"batch_size": 32, "num_workers": 4, "with_augmentation": False},
        {"batch_size": 32, "num_workers": 8, "with_augmentation": False},
        # Test batch sizes without augmentation
        {"batch_size": 16, "num_workers": 4, "with_augmentation": False},
        {"batch_size": 32, "num_workers": 4, "with_augmentation": False},
        # Test with augmentation
        {"batch_size": 32, "num_workers": 4, "with_augmentation": True},
        {"batch_size": 32, "num_workers": 8, "with_augmentation": True},
    ]

    results = {
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": [],
    }

    for config in configurations:
        aug_label = "WITH AUG" if config["with_augmentation"] else "NO AUG"
        label = f"bs={config['batch_size']} workers={config['num_workers']} {aug_label}"

        print(f"\nBenchmarking: {label}")
        print("-" * 80)

        try:
            metrics = benchmark_dataloader(
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                num_batches=100,
                with_augmentation=config["with_augmentation"],
                device=device,
            )

            result = {
                "configuration": config,
                "metrics": metrics,
            }
            results["benchmarks"].append(result)

            # Print results
            print(
                f"  Data time:   {metrics['data_time_mean_ms']:.2f} ± "
                f"{metrics['data_time_std_ms']:.2f} ms"
            )
            print(
                f"  Step time:   {metrics['step_time_mean_ms']:.2f} ± "
                f"{metrics['step_time_std_ms']:.2f} ms"
            )
            print(f"  Data/Step:   {metrics['data_step_ratio']:.3f}")
            print(
                f"  Throughput:  {metrics['throughput_samples_per_sec']:.1f} samples/sec"
            )

            # Check contract
            if metrics["data_step_ratio"] > 0.40:
                print("  ⚠️  WARNING: Ratio exceeds 0.40 threshold!")
            else:
                print("  ✅ PASS: Ratio within budget")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results["benchmarks"].append(
                {
                    "configuration": config,
                    "error": str(e),
                }
            )

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<40} {'Throughput':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 80)

    for benchmark in results["benchmarks"]:
        if "error" in benchmark:
            continue

        config = benchmark["configuration"]
        metrics = benchmark["metrics"]

        aug = "AUG" if config["with_augmentation"] else "---"
        config_str = f"bs={config['batch_size']:2d} w={config['num_workers']:2d} {aug}"
        throughput = metrics["throughput_samples_per_sec"]
        ratio = metrics["data_step_ratio"]
        status = "✅" if ratio <= 0.40 else "⚠️"

        print(f"{config_str:<40} {throughput:>10.1f}/s {ratio:>8.3f} {status:>8}")

    # Save results
    if output_path:
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive DataLoader performance benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_dataloader_results.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer configurations)",
    )

    args = parser.parse_args()

    run_benchmark_suite(output_path=args.output)


if __name__ == "__main__":
    main()
