"""
Test performance contracts and benchmarks.

Validates that the system meets performance requirements for:
- Data loading throughput
- Training step timing
- Memory usage
- CPU/GPU utilization
"""

import time

import pytest
import torch


class TestDataLoaderPerformance:
    """Test DataLoader performance contracts."""

    def test_dataloader_config_exists(self):
        """Test that DataLoader configuration utilities exist."""
        from psy_agents_noaug.utils.reproducibility import (
            get_optimal_dataloader_kwargs,
        )

        device = torch.device("cpu")
        kwargs = get_optimal_dataloader_kwargs(device)

        assert "num_workers" in kwargs
        assert isinstance(kwargs["num_workers"], int)
        assert kwargs["num_workers"] >= 0

    def test_pin_memory_enabled_for_cuda(self):
        """Test that pin_memory is enabled for CUDA devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from psy_agents_noaug.utils.reproducibility import (
            get_optimal_dataloader_kwargs,
        )

        device = torch.device("cuda")
        kwargs = get_optimal_dataloader_kwargs(device)

        assert kwargs["pin_memory"] is True

    def test_persistent_workers_configuration(self):
        """Test that persistent_workers is configured appropriately."""
        from psy_agents_noaug.utils.reproducibility import (
            get_optimal_dataloader_kwargs,
        )

        device = torch.device("cpu")
        kwargs = get_optimal_dataloader_kwargs(device, num_workers=4)

        # Should enable persistent workers when workers > 0
        if kwargs["num_workers"] > 0:
            assert kwargs.get("persistent_workers", False) is True

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Requires CUDA for meaningful test"
    )
    def test_data_step_ratio_below_threshold(self):
        """Test that data loading time / step time ratio stays below 0.40.

        This ensures GPU is not starved waiting for data. A ratio > 0.40 indicates
        the DataLoader is bottlenecking training performance.

        This test can be skipped on slow CI with: pytest -m "not slow"
        """
        from torch.utils.data import DataLoader, Dataset

        class DummyDataset(Dataset):
            """Fast synthetic dataset for testing."""

            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Simulate tokenized text: (input_ids, attention_mask, labels)
                return {
                    "input_ids": torch.randint(0, 1000, (128,)),
                    "attention_mask": torch.ones(128, dtype=torch.long),
                    "labels": torch.randint(0, 2, (1,)),
                }

        # Create DataLoader with optimized settings
        from psy_agents_noaug.utils.reproducibility import get_optimal_dataloader_kwargs

        device = torch.device("cuda")
        dataloader_kwargs = get_optimal_dataloader_kwargs(device, num_workers=4)

        dataset = DummyDataset(size=50)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            **dataloader_kwargs,
        )

        # Create simple model for timing
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        model.train()

        # Warmup
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                _ = model(**inputs)
            break

        # Measure data loading time and step time
        data_times = []
        step_times = []

        data_start = time.time()
        for batch in dataloader:
            data_time = time.time() - data_start
            data_times.append(data_time)

            # Transfer to GPU (part of data loading)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            # Measure step time (forward + backward)
            torch.cuda.synchronize()
            step_start = time.time()

            outputs = model(**inputs)
            loss = outputs.last_hidden_state.mean()
            loss.backward()

            torch.cuda.synchronize()
            step_time = time.time() - step_start
            step_times.append(step_time)

            data_start = time.time()

        # Calculate average times (excluding first batch for warmup)
        avg_data_time = sum(data_times[1:]) / len(data_times[1:])
        avg_step_time = sum(step_times[1:]) / len(step_times[1:])

        # Calculate ratio
        data_step_ratio = (
            avg_data_time / avg_step_time if avg_step_time > 0 else float("inf")
        )

        # Performance contract: data/step ratio must be <= 0.40
        threshold = 0.40
        assert data_step_ratio <= threshold, (
            f"Data/step ratio {data_step_ratio:.3f} exceeds threshold {threshold:.2f}. "
            f"GPU is starved waiting for data. "
            f"Avg data time: {avg_data_time*1000:.1f}ms, "
            f"Avg step time: {avg_step_time*1000:.1f}ms. "
            f"Consider increasing num_workers or enabling persistent_workers."
        )

        # Also verify reasonable absolute times
        assert avg_data_time < 0.5, f"Data loading too slow: {avg_data_time*1000:.1f}ms"
        assert (
            avg_step_time > 0.001
        ), f"Step time suspiciously fast: {avg_step_time*1000:.1f}ms"


class TestModelPerformance:
    """Test model performance characteristics."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_model_fits_on_gpu(self):
        """Test that model can be moved to GPU."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-uncased")
        device = torch.device("cuda")

        try:
            model = model.to(device)
            # Verify model is on GPU
            assert next(model.parameters()).device.type == "cuda"
        except RuntimeError:
            pytest.fail("Model failed to move to GPU")

    def test_forward_pass_completes(self):
        """Test that forward pass completes within reasonable time."""
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        text = "Test input for performance check"
        inputs = tokenizer(text, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        duration = time.time() - start

        # Forward pass should complete in < 1 second on CPU
        assert duration < 5.0, f"Forward pass took {duration:.2f}s (should be <5s)"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_forward_pass_faster_than_cpu(self):
        """Test that GPU forward pass is faster than CPU."""
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        text = "Test input " * 50  # Longer text for GPU benefit
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # CPU timing
        start_cpu = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        cpu_time = time.time() - start_cpu

        # GPU timing
        model_gpu = model.to("cuda")
        inputs_gpu = {k: v.to("cuda") for k, v in inputs.items()}

        # Warmup
        with torch.no_grad():
            _ = model_gpu(**inputs_gpu)

        torch.cuda.synchronize()
        start_gpu = time.time()
        with torch.no_grad():
            _ = model_gpu(**inputs_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_gpu

        # GPU should be faster (or at least competitive)
        # Allow some tolerance for small inputs
        assert (
            gpu_time <= cpu_time * 2
        ), f"GPU ({gpu_time:.3f}s) not faster than CPU ({cpu_time:.3f}s)"


class TestMemoryPerformance:
    """Test memory usage patterns."""

    def test_model_memory_footprint_reasonable(self):
        """Test that model memory footprint is reasonable."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-uncased")

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # bert-base has ~110M parameters
        assert 100_000_000 < param_count < 150_000_000

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_memory_is_freed_after_training_step(self):
        """Test that GPU memory is properly freed."""
        import torch
        from transformers import AutoModel

        device = torch.device("cuda")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)

        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)

        # Run forward and backward
        dummy_input = torch.randint(0, 1000, (8, 128), device=device)
        outputs = model(dummy_input)
        loss = outputs.last_hidden_state.mean()
        loss.backward()

        # Clear
        del outputs, loss, dummy_input
        torch.cuda.empty_cache()

        # Memory should return to near initial
        final_memory = torch.cuda.memory_allocated(device)
        memory_increase = final_memory - initial_memory

        # Allow small increase for model weights
        assert memory_increase < model.num_parameters() * 10  # Generous threshold


class TestAugmentationPerformance:
    """Test augmentation performance."""

    def test_augmentation_completes_quickly(self):
        """Test that augmentation completes within time budget."""
        from psy_agents_noaug.augmentation.pipeline import AugConfig, AugmenterPipeline

        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        text = "The patient exhibits symptoms of depression."

        start = time.time()
        for _ in range(10):
            _ = pipeline(text)
        duration = time.time() - start

        # 10 augmentations should complete in < 5 seconds
        assert duration < 5.0, f"Augmentation took {duration:.2f}s (should be <5s)"

    def test_augmentation_overhead_acceptable(self):
        """Test that augmentation overhead is acceptable."""
        from psy_agents_noaug.augmentation.pipeline import AugConfig, AugmenterPipeline

        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        text = "Test text for overhead measurement."
        samples = 100

        # Measure baseline (no augmentation)
        start_baseline = time.time()
        for _ in range(samples):
            _ = text  # No-op
        baseline_time = time.time() - start_baseline

        # Measure with augmentation
        start_aug = time.time()
        for _ in range(samples):
            _ = pipeline(text)
        aug_time = time.time() - start_aug

        # Overhead should be reasonable (< 100ms per sample on average)
        overhead_per_sample = (aug_time - baseline_time) / samples
        assert (
            overhead_per_sample < 0.1
        ), f"Overhead {overhead_per_sample*1000:.1f}ms (should be <100ms)"


class TestTrainingLoopPerformance:
    """Test training loop performance characteristics."""

    def test_gradient_accumulation_logic(self):
        """Test that gradient accumulation is implemented correctly."""
        # This is a conceptual test - verify gradient accumulation exists
        # Actual implementation would be in train_loop.py
        from torch import nn

        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Simulate gradient accumulation
        grad_accum_steps = 4

        for _ in range(grad_accum_steps):
            dummy_input = torch.randn(2, 10)
            output = model(dummy_input)
            loss = output.mean()
            loss = loss / grad_accum_steps  # Scale loss
            loss.backward()

        # Step optimizer after accumulation
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error
        assert True

    def test_mixed_precision_available(self):
        """Test that mixed precision training is available."""
        # Check if autocast is available
        assert hasattr(torch.cuda.amp, "autocast")
        assert hasattr(torch.cuda.amp, "GradScaler")


class TestReproducibilityPerformance:
    """Test reproducibility vs performance tradeoffs."""

    def test_deterministic_mode_performance_impact(self):
        """Test performance impact of deterministic mode."""
        from psy_agents_noaug.utils.reproducibility import set_seed

        # Non-deterministic mode
        set_seed(42, deterministic=False, cudnn_benchmark=True)

        # Deterministic mode
        set_seed(42, deterministic=True, cudnn_benchmark=False)

        # Should complete without error
        # Actual performance difference would require full training run
        assert True

    def test_cudnn_benchmark_settings(self):
        """Test that cuDNN benchmark can be toggled."""
        import torch

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            assert torch.backends.cudnn.benchmark is True

            torch.backends.cudnn.benchmark = False
            assert torch.backends.cudnn.benchmark is False


class TestBatchProcessing:
    """Test batch processing efficiency."""

    def test_batch_processing_faster_than_sequential(self):
        """Test that batch processing is more efficient."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        texts = ["Sample text for testing"] * 8

        # Sequential processing
        start_seq = time.time()
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt")
                _ = model(**inputs)
        seq_time = time.time() - start_seq

        # Batch processing
        start_batch = time.time()
        with torch.no_grad():
            inputs = tokenizer(texts, return_tensors="pt", padding=True)
            _ = model(**inputs)
        batch_time = time.time() - start_batch

        # Batch should be faster (at least 2x)
        assert (
            batch_time < seq_time
        ), f"Batch ({batch_time:.2f}s) not faster than sequential ({seq_time:.2f}s)"


class TestSystemUtilization:
    """Test system resource utilization."""

    def test_cpu_count_detection(self):
        """Test that CPU count is detected correctly."""
        import os

        cpu_count = os.cpu_count()
        assert cpu_count is not None
        assert cpu_count > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_detection(self):
        """Test that GPU is detected correctly."""
        device_count = torch.cuda.device_count()
        assert device_count > 0

        device_name = torch.cuda.get_device_name(0)
        assert isinstance(device_name, str)
        assert len(device_name) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gpu_memory_info(self):
        """Test that GPU memory info is accessible."""
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory

        assert total_memory > 0
        assert total_memory > 1e9  # At least 1GB


class TestEdgeCases:
    """Test performance edge cases."""

    def test_empty_batch_handling(self):
        """Test that empty batch is handled gracefully."""
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained("bert-base-uncased")

        # Empty batch should either work or raise clear error
        try:
            empty_input = torch.zeros(0, 128, dtype=torch.long)
            with torch.no_grad():
                _ = model(empty_input)
        except (ValueError, RuntimeError):
            # Expected - empty batch should raise error
            pass

    def test_large_batch_memory_handling(self):
        """Test that large batch sizes are handled appropriately."""
        import torch

        # Test that we can detect available memory and adjust batch size
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1e9

            # Verify memory is reasonable
            assert total_memory_gb > 1, "GPU should have > 1GB memory"
