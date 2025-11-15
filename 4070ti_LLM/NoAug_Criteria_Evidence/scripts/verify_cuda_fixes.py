#!/usr/bin/env python3
"""
Verification script for CUDA fragmentation fixes.

This script verifies that all 5 fixes are properly implemented:
1. PYTORCH_CUDA_ALLOC_CONF environment variable
2. Enhanced CUDA cache clearing
3. Periodic GPU reset mechanism
4. Reduced parallel count (3 instead of 4)
5. Logging updates

Usage:
    python scripts/verify_cuda_fixes.py
"""

import os
import re
import sys
from pathlib import Path


def check_environment_variable():
    """Check if PYTORCH_CUDA_ALLOC_CONF is set in tune_max.py"""
    print("\n[Fix 1] Checking PYTORCH_CUDA_ALLOC_CONF environment variable...")

    tune_max_path = Path(__file__).parent / "tune_max.py"
    content = tune_max_path.read_text()

    # Check for the environment variable setting
    if 'PYTORCH_CUDA_ALLOC_CONF' in content and 'expandable_segments:True' in content:
        print("  ✅ FOUND: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        # Find the line number for reference
        for i, line in enumerate(content.split('\n'), 1):
            if 'PYTORCH_CUDA_ALLOC_CONF' in line and '=' in line:
                print(f"     Location: scripts/tune_max.py line {i}")
                break
        return True
    else:
        print("  ❌ MISSING: PYTORCH_CUDA_ALLOC_CONF environment variable")
        return False


def check_enhanced_cleanup():
    """Check for enhanced CUDA cache clearing in finally block"""
    print("\n[Fix 2] Checking enhanced CUDA cache clearing...")

    tune_max_path = Path(__file__).parent / "tune_max.py"
    content = tune_max_path.read_text()

    # Check for double empty_cache calls in finally block
    finally_blocks = re.findall(r'finally:.*?(?=\n(?:def|class|\Z))', content, re.DOTALL)

    found = False
    for block in finally_blocks:
        # Count empty_cache calls in this finally block
        empty_cache_count = block.count('torch.cuda.empty_cache()')

        if empty_cache_count >= 2:  # Should have at least 2 calls
            print(f"  ✅ FOUND: Enhanced cleanup with {empty_cache_count} empty_cache() calls")

            # Find line number
            block_start = content.find(block)
            line_num = content[:block_start].count('\n') + 1
            print(f"     Location: scripts/tune_max.py around line {line_num}")
            found = True
            break

    if not found:
        print("  ❌ MISSING: Enhanced CUDA cache clearing (double empty_cache)")

    return found


def check_periodic_reset():
    """Check for periodic GPU reset mechanism"""
    print("\n[Fix 3] Checking periodic GPU reset every 50 trials...")

    tune_max_path = Path(__file__).parent / "tune_max.py"
    content = tune_max_path.read_text()

    # Check for successful_trials tracking
    has_counter = 'successful_trials = [0]' in content or 'successful_trials[0]' in content

    # Check for modulo 50 check
    has_periodic_check = '% 50 == 0' in content or '%50==0' in content

    # Check for GPU RESET message
    has_reset_message = 'GPU RESET' in content

    # Check for counter increment
    has_increment = 'successful_trials[0] += 1' in content

    all_present = has_counter and has_periodic_check and has_reset_message and has_increment

    if all_present:
        print("  ✅ FOUND: Complete periodic reset mechanism")
        print("     - successful_trials counter")
        print("     - Modulo 50 check")
        print("     - GPU RESET logging")
        print("     - Counter increment on success")

        # Find line numbers
        for i, line in enumerate(content.split('\n'), 1):
            if 'successful_trials = [0]' in line:
                print(f"     Location: scripts/tune_max.py line {i}")
                break
    else:
        print("  ❌ INCOMPLETE periodic reset mechanism:")
        if not has_counter:
            print("     - Missing successful_trials counter")
        if not has_periodic_check:
            print("     - Missing modulo 50 check")
        if not has_reset_message:
            print("     - Missing GPU RESET logging")
        if not has_increment:
            print("     - Missing counter increment")

    return all_present


def check_parallel_reduction():
    """Check if parallel count reduced from 4 to 3 in Makefile"""
    print("\n[Fix 4] Checking parallel trial reduction (4 → 3)...")

    makefile_path = Path(__file__).parent.parent / "Makefile"
    content = makefile_path.read_text()

    # Find PAR variable definition
    par_match = re.search(r'^PAR\s*\?=\s*(\d+)', content, re.MULTILINE)

    if par_match:
        par_value = int(par_match.group(1))

        if par_value == 3:
            print(f"  ✅ FOUND: PAR = {par_value} (reduced from 4)")

            # Find line number
            line_num = content[:par_match.start()].count('\n') + 1
            print(f"     Location: Makefile line {line_num}")
            return True
        else:
            print(f"  ⚠️  WARNING: PAR = {par_value} (expected 3)")
            print("     You may want to reduce this for memory stability")
            return False
    else:
        print("  ❌ MISSING: PAR variable not found in Makefile")
        return False


def check_logging_update():
    """Check if logging message updated for user awareness"""
    print("\n[Fix 5] Checking logging updates...")

    makefile_path = Path(__file__).parent.parent / "Makefile"
    content = makefile_path.read_text()

    # Check for parallel count message in echo statements
    if 'reduced from 4' in content or 'prevent GPU OOM' in content:
        print("  ✅ FOUND: Updated logging with OOM prevention message")

        # Find line number
        for i, line in enumerate(content.split('\n'), 1):
            if 'prevent GPU OOM' in line or 'reduced from 4' in line:
                print(f"     Location: Makefile line {i}")
                break
        return True
    else:
        print("  ⚠️  INFO: Logging not explicitly updated (optional)")
        return True  # Not critical


def main():
    print("="*70)
    print("CUDA Fragmentation Fix Verification")
    print("="*70)

    results = []

    # Run all checks
    results.append(("Environment Variable", check_environment_variable()))
    results.append(("Enhanced Cleanup", check_enhanced_cleanup()))
    results.append(("Periodic GPU Reset", check_periodic_reset()))
    results.append(("Parallel Reduction", check_parallel_reduction()))
    results.append(("Logging Updates", check_logging_update()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All fixes verified! System is ready for stable HPO.")
        print("\nNext steps:")
        print("  1. Run: make tune-criteria-supermax")
        print("  2. Monitor with: tail -f hpo_supermax_run.log")
        print("  3. Watch for GPU RESET messages every 50 successful trials")
        print("  4. Verify stable operation for 500+ trials")
        return 0
    else:
        print("\n⚠️  Some fixes are missing or incomplete.")
        print("Please review the failed checks above and apply missing fixes.")
        print("\nSee docs/CUDA_FRAGMENTATION_FIX.md for complete fix details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
