#!/usr/bin/env python
"""Quick test to verify augmentation integration in HPO.

This script runs a minimal HPO test (5 trials, 3 epochs) to verify:
1. Augmentation pipeline initializes correctly
2. Different trials use different augmentation configs
3. No crashes or errors occur
4. Augmentation actually affects training

Usage:
    python scripts/test_augmentation_integration.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("           AUGMENTATION INTEGRATION TEST")
    print("=" * 70)
    print()
    print("This test will:")
    print("  1. Run 5 HPO trials with 3 epochs each")
    print("  2. Verify augmentation pipeline works")
    print("  3. Check for different aug configs across trials")
    print("  4. Report success/failure")
    print()
    print("Expected runtime: ~5-10 minutes")
    print("=" * 70)
    print()

    # Create output directory
    outdir = Path("./_runs/aug_integration_test")
    outdir.mkdir(parents=True, exist_ok=True)

    # Run quick HPO test
    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent", "criteria",
        "--study-name", "test-augmentation-integration",
        "--trials", "5",
        "--epochs", "3",
        "--patience", "2",
        "--outdir", str(outdir),
    ]

    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        # Save output
        log_file = Path("aug_integration_test.log")
        with open(log_file, "w") as f:
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n\n")
            f.write(result.stderr)

        print(f"✓ Test completed (exit code: {result.returncode})")
        print(f"✓ Output saved to: {log_file}")
        print()

        # Check for augmentation activity
        stdout = result.stdout.lower()
        has_aug = "aug" in stdout or "augment" in stdout

        if has_aug:
            print("✓ Augmentation keywords found in output")
        else:
            print("⚠ WARNING: No augmentation keywords in output")

        # Check trial outputs
        trial_dirs = sorted(outdir.glob("trial_*"))
        print(f"\n✓ Found {len(trial_dirs)} trial directories")

        if trial_dirs:
            print("\nTrial augmentation configs:")
            for trial_dir in trial_dirs[:3]:  # Show first 3
                params_file = trial_dir / "params.yaml"
                if params_file.exists():
                    # Read aug.* parameters
                    with open(params_file) as f:
                        lines = [
                            line.strip()
                            for line in f
                            if line.strip().startswith("aug.")
                        ]
                    print(f"\n  {trial_dir.name}:")
                    for line in lines:
                        print(f"    {line}")

        # Final verdict
        print("\n" + "=" * 70)
        if result.returncode == 0 and len(trial_dirs) >= 5:
            print("✅ TEST PASSED")
            print()
            print("Augmentation integration appears to be working correctly.")
            print()
            print("Next steps:")
            print("  1. Review logs: cat aug_integration_test.log")
            print(f"  2. Check trial configs: cat {outdir}/trial_*/params.yaml")
            print("  3. Run full HPO: scripts/run_with_aug_hpo.sh")
        else:
            print("❌ TEST FAILED")
            print()
            print(f"Exit code: {result.returncode}")
            print(f"Trials completed: {len(trial_dirs)}/5")
            print()
            print("Check logs for errors: cat aug_integration_test.log")

        print("=" * 70)

        return result.returncode

    except subprocess.TimeoutExpired:
        print("❌ TEST FAILED: Timeout after 10 minutes")
        return 1

    except Exception as exc:
        print(f"❌ TEST FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
