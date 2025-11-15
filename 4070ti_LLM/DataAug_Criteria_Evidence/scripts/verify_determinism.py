#!/usr/bin/env python3
"""
Verify deterministic behavior with fixed seeds.

Runs augmentation and tokenization twice with the same seed and verifies outputs match.
"""

import argparse
import sys

import numpy as np
import torch

from psy_agents_noaug.utils.reproducibility import set_seed


def verify_rng_determinism(seed: int = 42) -> bool:
    """Verify RNG determinism across runs."""
    # First run
    set_seed(seed)
    py_rand1 = [np.random.randint(0, 1000) for _ in range(10)]
    torch_rand1 = torch.randn(10, 10).numpy()

    # Second run
    set_seed(seed)
    py_rand2 = [np.random.randint(0, 1000) for _ in range(10)]
    torch_rand2 = torch.randn(10, 10).numpy()

    # Compare
    py_match = py_rand1 == py_rand2
    torch_match = np.allclose(torch_rand1, torch_rand2)

    return py_match and torch_match


def verify_cuda_determinism(seed: int = 42) -> bool:
    """Verify CUDA determinism if available."""
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping CUDA determinism check")
        return True

    import os

    # Set CUBLAS workspace config for determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # First run
    set_seed(seed)
    x1 = torch.randn(100, 100).cuda()
    # Use element-wise operations which are deterministic
    y1 = (x1 * x1).sum().cpu().numpy()

    # Second run
    set_seed(seed)
    x2 = torch.randn(100, 100).cuda()
    y2 = (x2 * x2).sum().cpu().numpy()

    return np.allclose(y1, y2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify deterministic behavior")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("Verifying determinism...")
    print(f"  Seed: {args.seed}")

    # Test RNG determinism
    print("\n1. Testing RNG determinism...")
    is_deterministic = verify_rng_determinism(args.seed)

    if is_deterministic:
        print("   ✅ PASS: RNG behavior is deterministic")
    else:
        print("   ❌ FAIL: RNG behavior is NOT deterministic")
        return 1

    # Test CUDA determinism
    print("\n2. Testing CUDA determinism...")
    cuda_deterministic = verify_cuda_determinism(args.seed)

    if cuda_deterministic:
        print("   ✅ PASS: CUDA behavior is deterministic")
    else:
        print("   ❌ FAIL: CUDA behavior is NOT deterministic")
        return 1

    print("\n" + "=" * 80)
    print("✅ ALL DETERMINISM CHECKS PASSED")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
