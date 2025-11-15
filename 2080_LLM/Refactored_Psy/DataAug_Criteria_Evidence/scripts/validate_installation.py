#!/usr/bin/env python3
"""Validate installation and dependencies."""

import sys
from importlib import import_module


def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False


def main():
    """Run installation validation."""
    print("=" * 60)
    print("PSY Agents AUG - Installation Validation")
    print("=" * 60)
    print()

    print("Checking core dependencies:")
    print("-" * 60)
    
    required_modules = [
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "sklearn",
        "datasets",
        "mlflow",
        "optuna",
        "hydra",
        "nlpaug",
        "textattack",
    ]

    results = []
    for module in required_modules:
        results.append(check_import(module))

    print()
    print("Checking project modules:")
    print("-" * 60)

    project_modules = [
        "psy_agents_aug",
    ]

    for module in project_modules:
        try:
            results.append(check_import(module))
        except:
            print(f"⚠ {module} (optional - may have different structure)")

    print()
    print("Checking development tools:")
    print("-" * 60)

    dev_modules = ["pytest", "ruff", "black", "isort"]

    for module in dev_modules:
        check_import(module)  # Don't fail on dev tools

    print()
    print("=" * 60)
    
    if all(results):
        print("✓ All required dependencies are installed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some dependencies are missing. Please run:")
        print("  poetry install")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
