#!/usr/bin/env python3
"""
Comprehensive micro-benchmark for augmentation pipeline.
Tests CPU methods, GPU methods, disk cache, and multiprocessing.
"""

import json
import sys
import os
import time
import yaml
from pathlib import Path
import subprocess
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set PYTHONPATH for subprocesses
os.environ['PYTHONPATH'] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    class tqdm:
        def __init__(self, total=None, desc=None, **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

def is_cuda_available():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def count_output_rows(output_root: Path):
    """Count total rows in all generated dataset.parquet files."""
    try:
        import pandas as pd
        datasets = list(output_root.rglob("dataset.parquet"))
        if not datasets:
            return 0
        return sum(len(pd.read_parquet(d)) for d in datasets)
    except Exception as e:
        print(f"Warning: Could not count rows: {e}", file=sys.stderr)
        return 0

def create_methods_yaml(method_ids: list, output_path: Path):
    """Create a filtered methods YAML with only specified methods."""
    # Load full registry
    full_registry_path = PROJECT_ROOT / "conf" / "augment_methods.yaml"
    with open(full_registry_path) as f:
        full_data = yaml.safe_load(f)

    # Filter to only specified methods
    filtered_methods = [m for m in full_data["methods"] if m["id"] in method_ids]

    # Write filtered YAML
    filtered_data = {"methods": filtered_methods}
    with open(output_path, "w") as f:
        yaml.dump(filtered_data, f)

    return output_path

def run_augmentation(output_root: str, method_ids: list = None, num_proc: int = 1,
                     use_cache: bool = True, verbose: bool = False):
    """
    Run augmentation CLI with specified parameters.
    Returns (success, duration_sec, row_count).
    """
    # Create temporary methods YAML if specific methods requested
    temp_yaml = None
    if method_ids:
        temp_yaml = Path(output_root) / "methods.yaml"
        temp_yaml.parent.mkdir(parents=True, exist_ok=True)
        create_methods_yaml(method_ids, temp_yaml)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "generate_augsets.py"),
        "--input", str(PROJECT_ROOT / "tests" / "fixtures" / "mini_annotations.csv"),
        "--text-col", "post_text",
        "--evidence-col", "evidence",
        "--criterion-col", "criterion",
        "--label-col", "label",
        "--id-col", "post_id",
        "--output-root", output_root,
        "--combo-mode", "singletons",
        "--variants-per-sample", "1",
        "--seed", "42",
        "--num-proc", str(num_proc),
    ]

    if temp_yaml:
        cmd.extend(["--methods-yaml", str(temp_yaml)])

    if not use_cache:
        cmd.append("--no-cache")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        duration = time.time() - start_time

        if result.returncode != 0:
            if verbose:
                print(f"Command failed with code {result.returncode}", file=sys.stderr)
                print(f"STDERR: {result.stderr}", file=sys.stderr)
            return False, duration, 0

        row_count = count_output_rows(Path(output_root))
        return True, duration, row_count

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"Command timed out after {duration:.1f}s", file=sys.stderr)
        return False, duration, 0
    except Exception as e:
        duration = time.time() - start_time
        print(f"Command failed: {e}", file=sys.stderr)
        return False, duration, 0

def benchmark_cpu_methods():
    """Benchmark fast CPU-only methods."""
    print("\n[1/4] Benchmarking CPU methods...")

    # Select 5 fast CPU methods
    cpu_methods = [
        "nlp_wordnet_syn",
        "nlp_randchar_ins",
        "ta_wordswap_wordnet",
        "ta_char_rand_del",
        "ta_word_delete"
    ]

    output_root = "/tmp/verify_bench_cpu"
    if Path(output_root).exists():
        shutil.rmtree(output_root)

    success, duration, rows = run_augmentation(
        output_root,
        method_ids=cpu_methods,
        num_proc=1
    )

    if not success:
        print("  WARNING: CPU benchmark failed", file=sys.stderr)
        return {
            "success": False,
            "methods_tested": cpu_methods,
            "duration_sec": duration,
            "total_rows": 0,
            "throughput_rows_per_sec": 0.0
        }

    throughput = rows / duration if duration > 0 else 0.0
    print(f"  Completed: {rows} rows in {duration:.2f}s ({throughput:.1f} rows/sec)")

    return {
        "success": True,
        "methods_tested": cpu_methods,
        "duration_sec": round(duration, 2),
        "total_rows": rows,
        "throughput_rows_per_sec": round(throughput, 2)
    }

def benchmark_gpu_methods():
    """Benchmark GPU method if CUDA available."""
    print("\n[2/4] Benchmarking GPU methods...")

    cuda_available = is_cuda_available()

    if not cuda_available:
        print("  CUDA not available - skipping GPU benchmarks")
        return {
            "cuda_available": False,
            "skipped": True,
            "reason": "CUDA not available"
        }

    # Test one GPU method
    gpu_methods = ["nlp_cwe_sub_roberta"]

    output_root = "/tmp/verify_bench_gpu"
    if Path(output_root).exists():
        shutil.rmtree(output_root)

    success, duration, rows = run_augmentation(
        output_root,
        method_ids=gpu_methods,
        num_proc=1
    )

    if not success:
        print("  WARNING: GPU benchmark failed", file=sys.stderr)
        return {
            "cuda_available": True,
            "success": False,
            "method_tested": gpu_methods[0],
            "duration_sec": duration,
            "total_rows": 0,
            "throughput_rows_per_sec": 0.0
        }

    throughput = rows / duration if duration > 0 else 0.0
    print(f"  Completed: {rows} rows in {duration:.2f}s ({throughput:.1f} rows/sec)")

    return {
        "cuda_available": True,
        "success": True,
        "method_tested": gpu_methods[0],
        "duration_sec": round(duration, 2),
        "total_rows": rows,
        "throughput_rows_per_sec": round(throughput, 2)
    }

def benchmark_disk_cache():
    """Test disk cache effectiveness."""
    print("\n[3/4] Benchmarking disk cache...")

    methods = ["nlp_wordnet_syn"]
    output_root = "/tmp/verify_bench_cache"

    # First run (cold cache)
    if Path(output_root).exists():
        shutil.rmtree(output_root)

    print("  First run (cold cache)...")
    success1, duration1, rows1 = run_augmentation(
        output_root,
        method_ids=methods,
        num_proc=1,
        use_cache=True
    )

    if not success1:
        print("  WARNING: First cache run failed", file=sys.stderr)
        return {
            "success": False,
            "first_run_sec": duration1,
            "second_run_sec": 0,
            "speedup_factor": 0.0
        }

    # Second run (warm cache)
    print("  Second run (warm cache)...")
    success2, duration2, rows2 = run_augmentation(
        output_root,
        method_ids=methods,
        num_proc=1,
        use_cache=True
    )

    if not success2:
        print("  WARNING: Second cache run failed", file=sys.stderr)
        speedup = 0.0
    else:
        speedup = duration1 / duration2 if duration2 > 0 else 0.0
        print(f"  Cache speedup: {speedup:.2f}x (from {duration1:.2f}s to {duration2:.2f}s)")

    return {
        "success": success1 and success2,
        "first_run_sec": round(duration1, 2),
        "second_run_sec": round(duration2, 2),
        "speedup_factor": round(speedup, 2)
    }

def benchmark_multiprocessing():
    """Test multiprocessing speedup."""
    print("\n[4/4] Benchmarking multiprocessing...")

    methods = ["nlp_wordnet_syn", "nlp_randchar_ins", "ta_word_delete"]

    # Single process
    output_root_1 = "/tmp/verify_bench_mp1"
    if Path(output_root_1).exists():
        shutil.rmtree(output_root_1)

    print("  Running with num_proc=1...")
    success1, duration1, rows1 = run_augmentation(
        output_root_1,
        method_ids=methods,
        num_proc=1
    )

    if not success1:
        print("  WARNING: Single-process run failed", file=sys.stderr)
        return {
            "success": False,
            "num_proc_1_sec": duration1,
            "num_proc_4_sec": 0,
            "speedup_factor": 0.0
        }

    # Multi-process
    output_root_4 = "/tmp/verify_bench_mp4"
    if Path(output_root_4).exists():
        shutil.rmtree(output_root_4)

    print("  Running with num_proc=4...")
    success4, duration4, rows4 = run_augmentation(
        output_root_4,
        method_ids=methods,
        num_proc=4
    )

    if not success4:
        print("  WARNING: Multi-process run failed", file=sys.stderr)
        speedup = 0.0
    else:
        speedup = duration1 / duration4 if duration4 > 0 else 0.0
        print(f"  Multiprocessing speedup: {speedup:.2f}x (from {duration1:.2f}s to {duration4:.2f}s)")

    return {
        "success": success1 and success4,
        "num_proc_1_sec": round(duration1, 2),
        "num_proc_4_sec": round(duration4, 2),
        "speedup_factor": round(speedup, 2)
    }

def main():
    """Run all benchmarks and save results."""
    print("=" * 70)
    print("Augmentation Pipeline Micro-Benchmarks")
    print("=" * 70)

    results = {
        "cpu_methods": benchmark_cpu_methods(),
        "gpu_methods": benchmark_gpu_methods(),
        "disk_cache": benchmark_disk_cache(),
        "multiprocessing": benchmark_multiprocessing()
    }

    # Save results
    output_path = PROJECT_ROOT / "tools" / "verify" / "bench_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    if results["cpu_methods"]["success"]:
        print(f"CPU Methods:        {results['cpu_methods']['throughput_rows_per_sec']:.1f} rows/sec")
    else:
        print("CPU Methods:        FAILED")

    if results["gpu_methods"].get("cuda_available"):
        if results["gpu_methods"].get("success"):
            print(f"GPU Methods:        {results['gpu_methods']['throughput_rows_per_sec']:.1f} rows/sec")
        else:
            print("GPU Methods:        FAILED")
    else:
        print("GPU Methods:        SKIPPED (CUDA not available)")

    if results["disk_cache"]["success"]:
        print(f"Disk Cache:         {results['disk_cache']['speedup_factor']:.2f}x speedup")
    else:
        print("Disk Cache:         FAILED")

    if results["multiprocessing"]["success"]:
        print(f"Multiprocessing:    {results['multiprocessing']['speedup_factor']:.2f}x speedup")
    else:
        print("Multiprocessing:    FAILED")

    print(f"\nResults saved to: {output_path}")

    # Exit with error if any benchmark failed
    any_failed = (
        not results["cpu_methods"]["success"] or
        (results["gpu_methods"].get("cuda_available") and not results["gpu_methods"].get("success")) or
        not results["disk_cache"]["success"] or
        not results["multiprocessing"]["success"]
    )

    return 1 if any_failed else 0

if __name__ == "__main__":
    sys.exit(main())
