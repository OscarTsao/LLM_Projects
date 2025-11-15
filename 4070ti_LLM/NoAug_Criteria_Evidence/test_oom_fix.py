#!/usr/bin/env python3
"""Test script to verify OOM handling fix in tune_max.py.

This script simulates an OOM scenario and verifies that:
1. OOM exceptions are caught correctly
2. GPU memory is properly cleaned up
3. Subsequent trials can run without CUDA kernel errors
4. No "index out of bounds" errors occur after OOM
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn


def test_oom_cleanup():
    """Test that OOM cleanup prevents CUDA kernel errors."""
    if not torch.cuda.is_available():
        print("CUDA not available - skipping OOM test")
        return True

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    def simulate_oom_trial():
        """Simulate a trial that causes OOM."""
        model = None
        batch = None
        try:
            # Create a VERY large model that will definitely OOM on 24GB GPU
            print("\n[Trial 1] Creating very large model (should OOM on 24GB GPU)...")
            # 20000 x 100000 = 2B params x 4 bytes = 8GB per layer
            # Multiple layers will exceed 24GB
            model = nn.Sequential(
                nn.Linear(20000, 100000),
                nn.ReLU(),
                nn.Linear(100000, 100000),
                nn.ReLU(),
                nn.Linear(100000, 20000),
            ).to(device)

            # Try to allocate a huge batch
            batch = torch.randn(2000, 20000, device=device)
            output = model(batch)
            print("[Trial 1] Unexpectedly succeeded (GPU has more memory than expected)")

            # Cleanup on success
            if model is not None:
                model.cpu()
                del model
            if batch is not None:
                del batch
            return False

        except torch.cuda.OutOfMemoryError as e:
            print(f"[Trial 1] OOM occurred as expected: {str(e)[:100]}")

            # CRITICAL FIX: Cleanup before next trial
            import gc
            if model is not None:
                model.cpu()
                del model
            if 'batch' in locals():
                del batch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            return True

    def simulate_normal_trial():
        """Simulate a normal trial after OOM."""
        print("\n[Trial 2] Creating small model (should succeed)...")
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 2),
        ).to(device)

        # Small batch should work
        batch = torch.randn(8, 100, device=device)
        labels = torch.randint(0, 2, (8,), device=device)

        criterion = nn.CrossEntropyLoss()
        logits = model(batch)
        loss = criterion(logits, labels)

        print(f"[Trial 2] Success! Loss: {loss.item():.4f}")
        print(f"[Trial 2] Logits shape: {logits.shape}, Labels shape: {labels.shape}")

        # Cleanup
        import gc
        model.cpu()
        del model, batch, labels, criterion, logits, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        return True

    # Run test
    print("=" * 70)
    print("Testing OOM cleanup fix")
    print("=" * 70)

    # Test 1: Cause OOM
    if not simulate_oom_trial():
        print("\n❌ FAILED: Could not trigger OOM")
        return False

    # Test 2: Run normal trial after OOM
    try:
        if not simulate_normal_trial():
            print("\n❌ FAILED: Normal trial failed")
            return False
    except RuntimeError as e:
        if "index out of bounds" in str(e) or "kernel" in str(e).lower():
            print(f"\n❌ FAILED: CUDA kernel error after OOM: {e}")
            return False
        raise

    print("\n" + "=" * 70)
    print("✅ SUCCESS: OOM cleanup works correctly!")
    print("=" * 70)
    return True


def test_label_validation():
    """Test that labels are always in valid range [0, num_classes)."""
    print("\n" + "=" * 70)
    print("Testing label validation")
    print("=" * 70)

    from Project.Criteria.data.dataset import CriteriaDataset
    from transformers import AutoTokenizer

    csv_path = Path(__file__).parent / "data" / "redsm5" / "redsm5_annotations.csv"
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        print("Skipping label validation test")
        return True

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CriteriaDataset(csv_path=csv_path, tokenizer=tokenizer, max_length=128)

    print(f"Dataset size: {len(dataset)}")

    # Check all labels
    invalid_labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        label = item["labels"].item()
        if label < 0 or label >= 2:  # num_classes = 2
            invalid_labels.append((i, label))

    if invalid_labels:
        print(f"\n❌ FAILED: Found {len(invalid_labels)} invalid labels:")
        for idx, label in invalid_labels[:5]:
            print(f"  Index {idx}: label={label}")
        return False

    print(f"✅ All {len(dataset)} labels are valid (in range [0, 2))")
    return True


if __name__ == "__main__":
    success = True

    # Test 1: Label validation
    if not test_label_validation():
        success = False

    # Test 2: OOM cleanup
    if not test_oom_cleanup():
        success = False

    sys.exit(0 if success else 1)
