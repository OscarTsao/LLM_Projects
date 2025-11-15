from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def workspace_tmp_path() -> Path:
    """Create a temporary directory inside the repo workspace for sandbox compatibility."""

    base = Path.cwd() / ".pytest_workspace"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
