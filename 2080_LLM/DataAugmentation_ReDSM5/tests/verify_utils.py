"""Shared utilities for verification tests."""
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import pandas as pd
import json

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Project root is two levels up from tests/
PROJECT_ROOT = Path(__file__).parent.parent

# Default column names for mini_annotations.csv
DEFAULT_TEXT_COL = "post_text"
DEFAULT_EVIDENCE_COL = "evidence"
DEFAULT_ID_COL = "post_id"
DEFAULT_CRITERION_COL = "criterion"
DEFAULT_LABEL_COL = "label"

def load_fixture(name: str = "mini_annotations.csv") -> pd.DataFrame:
    """Load test fixture CSV."""
    path = FIXTURES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return pd.read_csv(path)

@contextmanager
def temp_output_dir(prefix="verify_"):
    """Create temporary output directory that auto-cleans."""
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield Path(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def run_cli(*args, cwd=None, add_defaults=True) -> tuple[int, str, str]:
    """
    Run CLI and return (exit_code, stdout, stderr).
    
    Args:
        *args: Command line arguments
        cwd: Working directory (defaults to project root)
        add_defaults: If True, automatically add --text-col, --evidence-col, --id-col 
                     if not already present in args
    """
    import subprocess
    import sys
    
    # Set up environment to include project root in PYTHONPATH
    env = os.environ.copy()
    project_root = str(PROJECT_ROOT)
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{pythonpath}"
    else:
        env["PYTHONPATH"] = project_root
    
    # Add default column arguments if not present
    args_list = list(args)
    if add_defaults:
        if "--text-col" not in args_list:
            args_list.extend(["--text-col", DEFAULT_TEXT_COL])
        if "--evidence-col" not in args_list:
            args_list.extend(["--evidence-col", DEFAULT_EVIDENCE_COL])
        if "--id-col" not in args_list:
            args_list.extend(["--id-col", DEFAULT_ID_COL])
    
    result = subprocess.run(
        [sys.executable, "tools/generate_augsets.py", *args_list],
        cwd=cwd or project_root,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr

def parse_meta_json(path: Path) -> dict:
    """Parse meta.json from combo output."""
    return json.loads((path / "meta.json").read_text())
