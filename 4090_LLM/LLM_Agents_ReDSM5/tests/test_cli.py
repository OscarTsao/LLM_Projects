"""
Test CLI interfaces for train, eval, and HPO scripts.
"""

import subprocess
import sys
import pytest


def test_train_help():
    """Test that train script shows help message."""
    result = subprocess.run(
        [sys.executable, '-m', 'src.train', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower() or 'options:' in result.stdout.lower()


def test_train_missing_required_args():
    """Test that train script fails without required arguments."""
    result = subprocess.run(
        [sys.executable, '-m', 'src.train'],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode != 0


def test_eval_help():
    """Test that eval script shows help message."""
    result = subprocess.run(
        [sys.executable, '-m', 'src.eval', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower() or 'options:' in result.stdout.lower()


def test_hpo_help():
    """Test that HPO script shows help message."""
    result = subprocess.run(
        [sys.executable, '-m', 'src.hpo', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )

    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower() or 'options:' in result.stdout.lower()


def test_module_invocation():
    """Test that src modules can be invoked as modules."""
    # Test that src.train can be run (even if it fails due to missing args)
    result = subprocess.run(
        [sys.executable, '-c', 'import src.train'],
        capture_output=True,
        text=True,
        timeout=10
    )

    # Should not have import errors
    assert 'ImportError' not in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr
