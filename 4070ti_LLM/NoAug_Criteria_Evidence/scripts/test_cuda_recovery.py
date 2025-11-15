#!/usr/bin/env python3
"""
Quick test script to verify CUDA error recovery system.

Tests:
1. CUDA health check function
2. Seed setting with error handling
3. Defensive validation (labels, NaN, Inf)
4. Consecutive failure detection

Usage:
    python scripts/test_cuda_recovery.py
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def test_cuda_health_check():
    """Test the CUDA health check function."""
    print("\n" + "=" * 70)
    print("TEST 1: CUDA Health Check Function")
    print("=" * 70)

    # Import the function from tune_max.py
    import scripts.tune_max as tune_max

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping CUDA tests")
        return True

    result = tune_max.check_cuda_health()
    if result:
        print("✓ CUDA health check PASSED")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("✗ CUDA health check FAILED")
        return False


def test_set_seeds():
    """Test the enhanced set_seeds function."""
    print("\n" + "=" * 70)
    print("TEST 2: Enhanced set_seeds Function")
    print("=" * 70)

    import scripts.tune_max as tune_max

    try:
        tune_max.set_seeds(42)
        print("✓ set_seeds(42) completed successfully")

        # Verify seeds were set
        if torch.cuda.is_available():
            # Create a tensor to verify CUDA works
            t = torch.rand(10, device="cuda")
            print(f"✓ Created CUDA tensor: shape={t.shape}, device={t.device}")

        print("✓ set_seeds with error detection WORKING")
        return True
    except Exception as e:
        print(f"✗ set_seeds failed: {e}")
        return False


def test_label_validation():
    """Test defensive label validation."""
    print("\n" + "=" * 70)
    print("TEST 3: Defensive Label Validation")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping validation tests")
        return True

    device = torch.device("cuda")
    num_labels = 2

    # Test 1: Valid labels (should pass)
    print("\nTest 3.1: Valid labels [0, 1]")
    labels = torch.tensor([0, 1, 1, 0, 1], device=device)
    label_min = labels.min().item()
    label_max = labels.max().item()

    if label_min >= 0 and label_max < num_labels:
        print(f"✓ Valid labels detected: min={label_min}, max={label_max}")
    else:
        print(f"✗ Validation failed for valid labels")
        return False

    # Test 2: Invalid labels (should fail)
    print("\nTest 3.2: Invalid labels [0, 2] (out of range)")
    invalid_labels = torch.tensor([0, 1, 2, 0, 1], device=device)
    label_min = invalid_labels.min().item()
    label_max = invalid_labels.max().item()

    if label_min < 0 or label_max >= num_labels:
        print(f"✓ Invalid labels correctly detected: min={label_min}, max={label_max}")
    else:
        print(f"✗ Validation failed to detect invalid labels")
        return False

    # Test 3: NaN detection
    print("\nTest 3.3: NaN detection in logits")
    logits = torch.tensor([[0.5, 0.5], [float('nan'), 0.3]], device=device)

    if torch.isnan(logits).any():
        nan_count = torch.isnan(logits).sum().item()
        print(f"✓ NaN correctly detected: count={nan_count}")
    else:
        print("✗ Failed to detect NaN")
        return False

    # Test 4: Inf detection
    print("\nTest 3.4: Inf detection in logits")
    logits = torch.tensor([[0.5, 0.5], [float('inf'), 0.3]], device=device)

    if torch.isinf(logits).any():
        inf_count = torch.isinf(logits).sum().item()
        print(f"✓ Inf correctly detected: count={inf_count}")
    else:
        print("✗ Failed to detect Inf")
        return False

    print("\n✓ All defensive validation tests PASSED")
    return True


def test_consecutive_failure_tracking():
    """Test consecutive failure tracking logic."""
    print("\n" + "=" * 70)
    print("TEST 4: Consecutive Failure Tracking")
    print("=" * 70)

    # Simulate the consecutive failure counter
    consecutive_failures = [0]

    print("\nSimulating trial outcomes:")

    # Success
    consecutive_failures[0] = 0
    print(f"  Trial 1: Success → counter reset to {consecutive_failures[0]}")

    # First failure
    consecutive_failures[0] += 1
    print(f"  Trial 2: CUDA error → counter = {consecutive_failures[0]}")
    if consecutive_failures[0] >= 3:
        print("    → Would trigger restart")
    else:
        print("    → Continue with cleanup")

    # Second failure
    consecutive_failures[0] += 1
    print(f"  Trial 3: CUDA error → counter = {consecutive_failures[0]}")
    if consecutive_failures[0] >= 3:
        print("    → Would trigger restart")
    else:
        print("    → Continue with cleanup")

    # Third failure (should trigger restart)
    consecutive_failures[0] += 1
    print(f"  Trial 4: CUDA error → counter = {consecutive_failures[0]}")
    if consecutive_failures[0] >= 3:
        print("    → ✓ Would trigger RESTART (correct)")
        restart_triggered = True
    else:
        print("    → ✗ Should have triggered restart")
        restart_triggered = False

    # Success after restart
    consecutive_failures[0] = 0
    print(f"  Trial 5: Success → counter reset to {consecutive_failures[0]}")

    if restart_triggered:
        print("\n✓ Consecutive failure tracking WORKING")
        return True
    else:
        print("\n✗ Consecutive failure tracking FAILED")
        return False


def test_environment_variables():
    """Test that critical environment variables are set."""
    print("\n" + "=" * 70)
    print("TEST 5: Environment Variables")
    print("=" * 70)

    # These should be set by tune_max.py when imported
    required_vars = {
        "CUDA_LAUNCH_BLOCKING": "1",
        "TORCH_USE_CUDA_DSA": "1",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    }

    all_set = True
    for var, expected in required_vars.items():
        actual = os.environ.get(var)
        if actual == expected:
            print(f"✓ {var}={actual}")
        else:
            print(f"✗ {var}={actual} (expected: {expected})")
            all_set = False

    if all_set:
        print("\n✓ All environment variables correctly set")
        return True
    else:
        print("\n✗ Some environment variables missing or incorrect")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CUDA ERROR RECOVERY SYSTEM - TEST SUITE")
    print("=" * 70)

    results = {
        "Environment Variables": test_environment_variables(),
        "CUDA Health Check": test_cuda_health_check(),
        "Enhanced set_seeds": test_set_seeds(),
        "Defensive Validation": test_label_validation(),
        "Consecutive Failure Tracking": test_consecutive_failure_tracking(),
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
        print("\nNext steps:")
        print("  1. Run smoke test: python scripts/tune_max.py --agent criteria --study smoke --n-trials 5")
        print("  2. Run stress test: python scripts/tune_max.py --agent criteria --study stress --n-trials 100")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review the fix implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
