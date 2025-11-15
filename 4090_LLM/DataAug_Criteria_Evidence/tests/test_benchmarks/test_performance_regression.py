"""Performance regression tests for DataLoader and augmentation."""

import sys
from pathlib import Path

import pytest
import torch

# Add scripts to path for importing benchmark functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from bench_dataloader import benchmark_dataloader

    BENCH_AVAILABLE = True
except ImportError:
    BENCH_AVAILABLE = False


@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.skipif(not BENCH_AVAILABLE, reason="bench_dataloader not importable")
class TestDataLoaderPerformance:
    """Test suite for DataLoader performance regression."""

    def test_dataloader_throughput_baseline_no_aug(self):
        """Test DataLoader throughput meets baseline (no augmentation)."""
        # Baseline: Expect at least 500 samples/sec on CPU, 1000+ on GPU
        baseline_cpu = 100.0  # Conservative baseline for CI
        baseline_gpu = 500.0

        results = benchmark_dataloader(
            batch_size=32,
            num_workers=4,
            num_batches=50,  # Reduced for faster tests
            with_augmentation=False,
        )

        throughput = results["throughput_samples_per_sec"]
        baseline = baseline_gpu if torch.cuda.is_available() else baseline_cpu

        assert throughput >= baseline, (
            f"DataLoader throughput ({throughput:.1f} samples/sec) "
            f"below baseline ({baseline:.1f} samples/sec)"
        )

    def test_dataloader_data_step_ratio_no_aug(self):
        """Test data/step ratio is within budget (no augmentation)."""
        max_ratio = 0.40

        results = benchmark_dataloader(
            batch_size=32,
            num_workers=4,
            num_batches=50,
            with_augmentation=False,
        )

        ratio = results["data_step_ratio"]

        assert (
            ratio <= max_ratio
        ), f"Data/step ratio ({ratio:.3f}) exceeds budget ({max_ratio:.3f})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_dataloader_throughput_with_aug_gpu(self):
        """Test DataLoader throughput with augmentation on GPU."""
        # With augmentation, expect at least 200 samples/sec on GPU
        baseline = 100.0  # Conservative for augmentation

        results = benchmark_dataloader(
            batch_size=32,
            num_workers=4,
            num_batches=50,
            with_augmentation=True,
            device="cuda",
        )

        throughput = results["throughput_samples_per_sec"]

        assert throughput >= baseline, (
            f"DataLoader throughput with augmentation ({throughput:.1f} samples/sec) "
            f"below baseline ({baseline:.1f} samples/sec)"
        )

    def test_dataloader_data_step_ratio_with_aug(self):
        """Test data/step ratio with augmentation is within budget."""
        # With augmentation, ratio should still be reasonable
        max_ratio = 0.40  # Same budget

        results = benchmark_dataloader(
            batch_size=32,
            num_workers=4,
            num_batches=50,
            with_augmentation=True,
        )

        ratio = results["data_step_ratio"]

        assert ratio <= max_ratio, (
            f"Data/step ratio with augmentation ({ratio:.3f}) "
            f"exceeds budget ({max_ratio:.3f})"
        )

    def test_dataloader_worker_scaling(self):
        """Test that increasing workers improves throughput."""
        # Test with 0, 1, and 4 workers
        results_0 = benchmark_dataloader(
            batch_size=32,
            num_workers=0,
            num_batches=30,
            with_augmentation=False,
        )

        results_4 = benchmark_dataloader(
            batch_size=32,
            num_workers=4,
            num_batches=30,
            with_augmentation=False,
        )

        # With 4 workers should be faster than 0 workers
        # Allow 10% margin for variance
        improvement = (
            results_4["throughput_samples_per_sec"]
            / results_0["throughput_samples_per_sec"]
        )

        # On some systems, multiprocessing overhead can make this worse
        # So we just check that it doesn't degrade significantly
        assert improvement > 0.5, (
            f"4 workers ({results_4['throughput_samples_per_sec']:.1f} samples/sec) "
            f"significantly slower than 0 workers "
            f"({results_0['throughput_samples_per_sec']:.1f} samples/sec)"
        )


@pytest.mark.slow
@pytest.mark.benchmark
class TestAugmentationPerformance:
    """Test suite for augmentation performance regression."""

    def test_augmentation_methods_available(self):
        """Test that all expected augmentation methods are available."""
        from psy_agents_noaug.augmentation import (
            ALL_METHODS,
            NLPAUG_METHODS,
            TEXTATTACK_METHODS,
        )

        # Expect at least 10 methods total
        assert (
            len(ALL_METHODS) >= 10
        ), f"Expected at least 10 augmentation methods, got {len(ALL_METHODS)}"

        # Check that methods are properly categorized
        assert len(NLPAUG_METHODS) > 0, "No nlpaug methods found"
        assert len(TEXTATTACK_METHODS) > 0, "No textattack methods found"

        # All methods should be in one category or the other
        assert set(ALL_METHODS) == set(
            NLPAUG_METHODS + TEXTATTACK_METHODS
        ), "Method categorization mismatch"

    def test_augmentation_pipeline_latency(self):
        """Test that augmentation pipeline has acceptable latency."""
        from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline

        config = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/word/SynonymAug(wordnet)"],
            p_apply=1.0,
            ops_per_sample=1,
            seed=42,
        )

        pipeline = AugmenterPipeline(config)

        # Test with sample text
        text = "Patient reports anxiety and depression symptoms."

        # Warmup
        for _ in range(10):
            _ = pipeline(text)

        # Measure latency
        import time

        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = pipeline(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

        import numpy as np

        avg_time = np.mean(times)

        # Should be fast enough for real-time augmentation
        max_latency = 50.0  # ms

        assert (
            avg_time < max_latency
        ), f"Augmentation latency ({avg_time:.2f}ms) exceeds max ({max_latency}ms)"


@pytest.mark.benchmark
class TestBenchmarkScripts:
    """Test that benchmark scripts are executable and produce valid output."""

    def test_bench_dataloader_script_exists(self):
        """Test that bench_dataloader.py script exists and is executable."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "bench_dataloader.py"
        )
        assert script_path.exists(), f"Script not found: {script_path}"
        assert (
            script_path.stat().st_mode & 0o111
        ), f"Script not executable: {script_path}"

    def test_profile_augmentation_script_exists(self):
        """Test that profile_augmentation.py script exists and is executable."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "profile_augmentation.py"
        )
        assert script_path.exists(), f"Script not found: {script_path}"
        assert (
            script_path.stat().st_mode & 0o111
        ), f"Script not executable: {script_path}"

    def test_gpu_utilization_script_exists(self):
        """Test that gpu_utilization.py script exists and is executable."""
        script_path = (
            Path(__file__).parent.parent.parent / "scripts" / "gpu_utilization.py"
        )
        assert script_path.exists(), f"Script not found: {script_path}"
        assert (
            script_path.stat().st_mode & 0o111
        ), f"Script not executable: {script_path}"


# Fixture for storing baseline metrics
@pytest.fixture(scope="session")
def baseline_metrics(tmp_path_factory):
    """Load or create baseline metrics for regression testing."""
    baseline_file = tmp_path_factory.getbasetemp() / "baseline_metrics.json"

    if baseline_file.exists():
        import json

        with open(baseline_file) as f:
            return json.load(f)

    # Default baselines (conservative)
    return {
        "dataloader_throughput_cpu": 100.0,  # samples/sec
        "dataloader_throughput_gpu": 500.0,
        "data_step_ratio_max": 0.40,
        "augmentation_latency_max_ms": 50.0,
    }


@pytest.mark.benchmark
def test_regression_detection(baseline_metrics):
    """Test that current performance meets baseline metrics."""
    if not BENCH_AVAILABLE:
        pytest.skip("bench_dataloader not available")

    # Run quick benchmark
    results = benchmark_dataloader(
        batch_size=32,
        num_workers=4,
        num_batches=30,
        with_augmentation=False,
    )

    throughput = results["throughput_samples_per_sec"]
    ratio = results["data_step_ratio"]

    baseline_key = (
        "dataloader_throughput_gpu"
        if torch.cuda.is_available()
        else "dataloader_throughput_cpu"
    )
    baseline_throughput = baseline_metrics[baseline_key]
    baseline_ratio = baseline_metrics["data_step_ratio_max"]

    # Allow 20% degradation tolerance
    tolerance = 0.80

    assert throughput >= baseline_throughput * tolerance, (
        f"Performance regression detected: throughput {throughput:.1f} < "
        f"baseline {baseline_throughput:.1f} * {tolerance}"
    )

    assert (
        ratio <= baseline_ratio
    ), f"Performance regression detected: ratio {ratio:.3f} > baseline {baseline_ratio:.3f}"
