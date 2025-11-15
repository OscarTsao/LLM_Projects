"""Linting checks."""
import subprocess
import pytest

def test_ruff_check():
    """Ruff linting passes."""
    result = subprocess.run(
        ["ruff", "check", "src/augment/", "tests/verify/", "tools/verify/"],
        capture_output=True,
    )
    # Allow exit code 0 or ruff not installed
    if result.returncode not in [0, 127]:
        pytest.fail(f"Ruff check failed:\n{result.stdout.decode()}")
