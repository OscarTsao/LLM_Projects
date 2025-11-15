"""CLI smoke tests."""
import subprocess
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def test_cli_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        ["python", "tools/generate_augsets.py", "--help"], 
        capture_output=True,
        env=env,
        cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0
    assert b"--input" in result.stdout
    assert b"--combo-mode" in result.stdout

def test_cli_required_args():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        ["python", "tools/generate_augsets.py"], 
        capture_output=True,
        env=env,
        cwd=str(PROJECT_ROOT)
    )
    assert result.returncode != 0  # Should fail without required args
