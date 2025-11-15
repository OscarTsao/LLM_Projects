#!/usr/bin/env python3
"""
Environment verification script for ReDSM5.

Checks Python version, dependencies, CUDA availability, and project structure.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} detected (need >= 3.10)")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check required Python packages."""
    required = [
        'torch', 'transformers', 'datasets', 'accelerate',
        'peft', 'scipy', 'sklearn', 'numpy', 'pandas', 'yaml'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)

    return len(missing) == 0


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA available: {device_name} ({memory:.2f} GB)")
            return True
        else:
            print("⚠️  CUDA not available (CPU-only mode)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_project_structure():
    """Check that essential project directories and files exist."""
    root = Path(__file__).parent.parent
    essential_paths = [
        root / "src",
        root / "src" / "train.py",
        root / "src" / "eval.py",
        root / "src" / "data.py",
        root / "configs" / "base.yaml",
        root / "configs" / "labels.yaml",
        root / "tests",
    ]

    all_exist = True
    for path in essential_paths:
        if path.exists():
            print(f"✅ {path.relative_to(root)}")
        else:
            print(f"❌ {path.relative_to(root)} missing")
            all_exist = False

    return all_exist


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("ReDSM5 Environment Verification")
    print("=" * 60)

    print("\n[1/4] Checking Python version...")
    python_ok = check_python_version()

    print("\n[2/4] Checking dependencies...")
    deps_ok = check_dependencies()

    print("\n[3/4] Checking CUDA...")
    cuda_ok = check_cuda()

    print("\n[4/4] Checking project structure...")
    structure_ok = check_project_structure()

    print("\n" + "=" * 60)
    if python_ok and deps_ok and structure_ok:
        print("✅ Environment verification passed!")
        if not cuda_ok:
            print("⚠️  Note: No CUDA detected. Training will be CPU-only.")
        return 0
    else:
        print("❌ Environment verification failed")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
