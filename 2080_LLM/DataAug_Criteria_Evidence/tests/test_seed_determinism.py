"""
Test reproducibility with seeds across multiple workers.

Validates that the seed management system produces deterministic results.
"""

import random

import numpy as np
import pytest
import torch

from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    set_seed,
)


class TestSeedReproducibility:
    """Test that seeds produce reproducible results."""

    def test_set_seed_basic(self):
        """Test that set_seed function executes without error."""
        set_seed(42, deterministic=True, cudnn_benchmark=False)
        # Should not raise exception

    def test_global_seed_sets_python_random(self):
        """Test that set_seed() sets Python random module."""
        set_seed(42)
        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        assert val1 == val2, "Python random not deterministic"

    def test_global_seed_sets_numpy(self):
        """Test that set_seed() sets NumPy random."""
        set_seed(42)
        val1 = np.random.rand()

        set_seed(42)
        val2 = np.random.rand()

        assert val1 == val2, "NumPy random not deterministic"

    def test_torch_determinism_cpu(self):
        """Test PyTorch determinism with same seed on CPU."""
        set_seed(42)
        tensor1 = torch.randn(10, 10)

        set_seed(42)
        tensor2 = torch.randn(10, 10)

        assert torch.allclose(tensor1, tensor2), "PyTorch CPU not deterministic"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_torch_determinism_cuda(self):
        """Test PyTorch determinism with same seed on CUDA."""
        set_seed(42, deterministic=True)
        tensor1 = torch.randn(10, 10, device="cuda")

        set_seed(42, deterministic=True)
        tensor2 = torch.randn(10, 10, device="cuda")

        assert torch.allclose(tensor1, tensor2), "PyTorch CUDA not deterministic"

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        val1 = random.random()

        set_seed(123)
        val2 = random.random()

        assert val1 != val2, "Different seeds should produce different results"

    def test_deterministic_mode_settings(self):
        """Test deterministic mode sets correct flags."""
        set_seed(42, deterministic=True, cudnn_benchmark=False)

        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False

    def test_non_deterministic_mode_settings(self):
        """Test non-deterministic mode allows benchmark."""
        set_seed(42, deterministic=False, cudnn_benchmark=True)

        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic is False
            assert torch.backends.cudnn.benchmark is True


class TestDeviceSelection:
    """Test device selection utilities."""

    def test_get_device_cpu(self):
        """Test get_device returns CPU when not preferring CUDA."""
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_get_device_cuda(self):
        """Test get_device returns CUDA when available and preferred."""
        device = get_device(prefer_cuda=True, device_id=0)
        assert device.type == "cuda"
        assert device.index == 0

    def test_get_device_specific_id(self):
        """Test get_device with specific device ID."""
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = get_device(prefer_cuda=True, device_id=1)
            assert device.index == 1
        else:
            # If only one GPU or no GPU, test falls back gracefully
            device = get_device(prefer_cuda=False)
            assert device.type == "cpu"


class TestDataLoaderOptimization:
    """Test DataLoader configuration optimization."""

    def test_get_optimal_kwargs_cpu(self):
        """Test optimal DataLoader kwargs for CPU."""
        device = torch.device("cpu")
        kwargs = get_optimal_dataloader_kwargs(device)

        assert "num_workers" in kwargs
        assert "pin_memory" in kwargs
        assert kwargs["pin_memory"] is False  # Should be False for CPU

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_get_optimal_kwargs_cuda(self):
        """Test optimal DataLoader kwargs for CUDA."""
        device = torch.device("cuda")
        kwargs = get_optimal_dataloader_kwargs(device)

        assert "num_workers" in kwargs
        assert "pin_memory" in kwargs
        assert kwargs["pin_memory"] is True  # Should be True for CUDA

    def test_get_optimal_kwargs_custom(self):
        """Test optimal DataLoader kwargs with custom values."""
        device = torch.device("cpu")
        kwargs = get_optimal_dataloader_kwargs(
            device, num_workers=4, pin_memory=True, persistent_workers=True
        )

        assert kwargs["num_workers"] == 4
        assert kwargs["pin_memory"] is True
        assert kwargs["persistent_workers"] is True

    def test_persistent_workers_disabled_with_zero_workers(self):
        """Test persistent_workers is disabled when num_workers=0."""
        device = torch.device("cpu")
        kwargs = get_optimal_dataloader_kwargs(device, num_workers=0)

        assert kwargs["num_workers"] == 0
        assert kwargs.get("persistent_workers", False) is False


class TestWorkerSeeding:
    """Test worker initialization for DataLoaders."""

    def test_worker_init_from_pipeline(self):
        """Test worker_init function from augmentation pipeline."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        # Test that worker_init returns correct seed
        seed1 = worker_init(0, 42)
        assert seed1 == 43  # base_seed + worker_id + 1

        seed2 = worker_init(5, 42)
        assert seed2 == 48  # 42 + 5 + 1

    def test_worker_seeds_are_unique(self):
        """Test that different workers get different seeds."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        base_seed = 42
        seeds = [worker_init(i, base_seed) for i in range(10)]

        # All seeds should be unique
        assert len(seeds) == len(set(seeds))

    def test_worker_seed_determinism(self):
        """Test that same worker ID always gets same seed."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        seed1 = worker_init(3, 42)
        seed2 = worker_init(3, 42)

        assert seed1 == seed2


class TestCrossLibraryDeterminism:
    """Test determinism across Python, NumPy, and PyTorch."""

    def test_all_rngs_seeded_together(self):
        """Test that all RNGs are seeded together."""
        set_seed(42)

        # Collect values from all RNGs
        python_val = random.random()
        numpy_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Reset and collect again
        set_seed(42)

        python_val2 = random.random()
        numpy_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        # All should match
        assert python_val == python_val2
        assert numpy_val == numpy_val2
        assert abs(torch_val - torch_val2) < 1e-6

    def test_seed_affects_all_libraries(self):
        """Test that changing seed affects all libraries."""
        set_seed(42)
        vals1 = (random.random(), np.random.rand(), torch.rand(1).item())

        set_seed(123)
        vals2 = (random.random(), np.random.rand(), torch.rand(1).item())

        # All should be different
        assert vals1[0] != vals2[0]
        assert vals1[1] != vals2[1]
        assert vals1[2] != vals2[2]


class TestEdgeCases:
    """Test edge cases in seed management."""

    def test_seed_with_zero(self):
        """Test that seed=0 works."""
        set_seed(0)
        val1 = random.random()

        set_seed(0)
        val2 = random.random()

        assert val1 == val2

    def test_seed_with_large_number(self):
        """Test that large seeds work."""
        large_seed = 2**31 - 1  # Max int32
        set_seed(large_seed)
        val1 = random.random()

        set_seed(large_seed)
        val2 = random.random()

        assert val1 == val2

    def test_repeated_seeding(self):
        """Test that repeated seeding produces same results."""
        for _ in range(5):
            set_seed(42)
            val = random.random()
            assert val == random.Random(42).random()


class TestDeterministicAlgorithms:
    """Test PyTorch deterministic algorithms setting."""

    def test_deterministic_algorithms_callable(self):
        """Test that deterministic algorithms can be set."""
        if hasattr(torch, "use_deterministic_algorithms"):
            # Should not raise exception
            set_seed(42, deterministic=True)

    @pytest.mark.skipif(
        not hasattr(torch, "use_deterministic_algorithms"),
        reason="Requires PyTorch >= 1.7",
    )
    def test_deterministic_algorithms_with_warn_only(self):
        """Test deterministic algorithms with warn_only mode."""
        # This should not raise even if some ops are non-deterministic
        set_seed(42, deterministic=True)
        # If we get here, it worked (either fully deterministic or warn_only mode)


class TestWorkerSeedingDDP:
    """Test worker initialization for multi-GPU setups."""

    def test_worker_init_with_rank(self):
        """Test worker_init function with DDP rank parameter."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        # Rank 0, Worker 0
        seed_r0_w0 = worker_init(
            worker_id=0, base_seed=42, rank=0, num_workers_per_rank=4
        )
        assert seed_r0_w0 == 43  # 42 + (0*4) + 0 + 1

        # Rank 1, Worker 0
        seed_r1_w0 = worker_init(
            worker_id=0, base_seed=42, rank=1, num_workers_per_rank=4
        )
        assert seed_r1_w0 == 47  # 42 + (1*4) + 0 + 1

        # Rank 1, Worker 2
        seed_r1_w2 = worker_init(
            worker_id=2, base_seed=42, rank=1, num_workers_per_rank=4
        )
        assert seed_r1_w2 == 49  # 42 + (1*4) + 2 + 1

    def test_worker_seeds_unique_across_ranks(self):
        """Test that workers across different ranks get unique seeds."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        base_seed = 42
        num_workers = 4
        num_ranks = 2

        seeds = []
        for rank in range(num_ranks):
            for worker_id in range(num_workers):
                seed = worker_init(worker_id, base_seed, rank, num_workers)
                seeds.append(seed)

        # All seeds should be unique
        assert len(seeds) == len(set(seeds)), f"Duplicate seeds found: {seeds}"

    def test_backward_compatibility_no_rank(self):
        """Test backward compatibility when rank is not specified."""
        from psy_agents_noaug.augmentation.pipeline import worker_init

        # Old-style call without rank (should default to rank=0)
        seed1 = worker_init(0, 42)
        seed2 = worker_init(0, 42, rank=0, num_workers_per_rank=1)

        assert seed1 == seed2, "Backward compatibility broken"


class TestAugmenterPipelineCleanup:
    """Test resource cleanup for AugmenterPipeline."""

    def test_context_manager(self):
        """Test AugmenterPipeline as context manager."""
        from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline

        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])

        with AugmenterPipeline(cfg) as pipeline:
            result = pipeline("test text")
            assert isinstance(result, str)

        # Pipeline should be cleaned up after context exit

    def test_close_method(self):
        """Test explicit close() method."""
        from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline

        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)

        # Should have augmenters
        assert hasattr(pipeline, "_augmenters")
        assert len(pipeline._augmenters) > 0

        # Close and cleanup
        pipeline.close()

        # Augmenters should be deleted
        assert not hasattr(pipeline, "_augmenters")

    def test_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline

        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)

        # Call close multiple times - should not raise
        pipeline.close()
        pipeline.close()
        pipeline.close()


class TestSystemInfo:
    """Test system information utilities."""

    def test_print_system_info(self):
        """Test that print_system_info executes without error."""
        from psy_agents_noaug.utils.reproducibility import print_system_info

        # Should not raise exception
        print_system_info()
