#!/usr/bin/env python3
"""
Test script for protobuf/sentencepiece multiprocessing fix.

This script validates that DeBERTa tokenizers can be loaded successfully
in a parallel multiprocessing context without protobuf descriptor errors.

Usage:
    python scripts/test_protobuf_fix.py [--parallel N]

Expected outcome:
    - All workers load tokenizers successfully
    - No protobuf/sentencepiece errors
    - Exit code 0
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_tokenizer_loading(model_name: str) -> dict[str, str]:
    """
    Test loading a tokenizer in a worker process.

    Returns:
        Dict with status and any error message
    """
    try:
        from transformers import AutoTokenizer

        # This is where the error would occur without the fix
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try to use it
        tokens = tokenizer("Test sentence for tokenization", return_tensors="pt")

        return {
            "model": model_name,
            "status": "success",
            "vocab_size": len(tokenizer),
            "error": None
        }
    except Exception as e:
        return {
            "model": model_name,
            "status": "failed",
            "vocab_size": None,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Test protobuf fix for parallel tokenizer loading")
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["microsoft/deberta-v3-base", "microsoft/deberta-v3-large"],
        help="Models to test (default: DeBERTa base and large)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Protobuf/Sentencepiece Multiprocessing Fix Test")
    print("=" * 70)
    print()

    # Check environment variable
    env_var = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
    print(f"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: {env_var or '(not set)'}")

    if env_var != "python":
        print()
        print("WARNING: Environment variable not set to 'python'")
        print("         This test may fail without the fix!")
        print()
        print("To fix, run:")
        print("  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python scripts/test_protobuf_fix.py")
        print()

    print(f"Testing with {args.parallel} parallel workers")
    print(f"Models: {', '.join(args.models)}")
    print()

    # Test each model with parallel loading
    all_success = True

    for model_name in args.models:
        print(f"Testing: {model_name}")
        print(f"  Spawning {args.parallel} workers to load tokenizer...")

        # Create a pool and load tokenizer in parallel
        with multiprocessing.Pool(processes=args.parallel) as pool:
            # Load the same model in multiple workers simultaneously
            results = pool.map(test_tokenizer_loading, [model_name] * args.parallel)

        # Check results
        success_count = sum(1 for r in results if r["status"] == "success")

        if success_count == args.parallel:
            print(f"  ✓ SUCCESS: All {args.parallel} workers loaded tokenizer")
            print(f"  Vocabulary size: {results[0]['vocab_size']}")
        else:
            print(f"  ✗ FAILED: Only {success_count}/{args.parallel} workers succeeded")
            all_success = False

            # Print errors
            for i, result in enumerate(results):
                if result["status"] == "failed":
                    print(f"  Worker {i+1} error: {result['error'][:100]}...")

        print()

    print("=" * 70)
    if all_success:
        print("✓ ALL TESTS PASSED")
        print()
        print("The protobuf/sentencepiece fix is working correctly!")
        print("Parallel HPO with DeBERTa models should work without errors.")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print("The fix may not be working correctly. Check:")
        print("  1. Environment variable is set: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python")
        print("  2. It's set BEFORE importing transformers")
        print("  3. See docs/PROTOBUF_FIX.md for troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
