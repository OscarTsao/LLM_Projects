#!/usr/bin/env python
"""Reproducibility management utilities (Phase 15).

This module provides tools to ensure experiment reproducibility:
- Comprehensive seed management
- Environment snapshot and restoration
- Dependency tracking
- Reproducibility validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


@dataclass
class ReproducibilitySnapshot:
    """Snapshot of reproducibility-related state."""

    timestamp: datetime
    seeds: dict[str, int]
    environment: dict[str, str]
    dependencies: dict[str, str]
    git_commit: str | None = None
    torch_version: str = ""
    numpy_version: str = ""
    python_version: str = ""
    cuda_version: str | None = None
    cudnn_version: str | None = None
    deterministic_mode: bool = False
    notes: str = ""


class ReproducibilityManager:
    """Manage experiment reproducibility."""

    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = True,
    ):
        """Initialize reproducibility manager.

        Args:
            seed: Random seed
            deterministic: Enable deterministic mode
        """
        self.seed = seed
        self.deterministic = deterministic
        self.snapshot: ReproducibilitySnapshot | None = None

        LOGGER.info(
            "Initialized ReproducibilityManager (seed=%d, deterministic=%s)",
            seed,
            deterministic,
        )

    def set_seeds(self, seed: int | None = None) -> None:
        """Set all random seeds.

        Args:
            seed: Random seed (None = use configured seed)
        """
        if seed is None:
            seed = self.seed

        # Python random
        random.seed(seed)

        # NumPy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set deterministic mode
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

            # Set environment variable for PyTorch
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        LOGGER.info("Set all random seeds to: %d", seed)

    def capture_snapshot(self) -> ReproducibilitySnapshot:
        """Capture current reproducibility state.

        Returns:
            Reproducibility snapshot
        """
        import sys

        # Get versions
        torch_version = torch.__version__
        numpy_version = np.__version__
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda

        cudnn_version = None
        if torch.backends.cudnn.is_available():
            cudnn_version = str(torch.backends.cudnn.version())

        # Get git commit
        git_commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            git_commit = result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Get dependencies
        dependencies = {}
        try:
            result = subprocess.run(
                ["pip", "freeze"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.strip().split("\n"):
                if "==" in line:
                    pkg, version = line.split("==", 1)
                    dependencies[pkg] = version
        except (subprocess.CalledProcessError, FileNotFoundError):
            LOGGER.warning("Failed to capture dependencies")

        # Create snapshot
        self.snapshot = ReproducibilitySnapshot(
            timestamp=datetime.now(),
            seeds={
                "main_seed": self.seed,
                "python_random_state": random.getstate()[1][0],
                "numpy_random_state": int(np.random.get_state()[1][0]),
            },
            environment=dict(os.environ),
            dependencies=dependencies,
            git_commit=git_commit,
            torch_version=torch_version,
            numpy_version=numpy_version,
            python_version=python_version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            deterministic_mode=self.deterministic,
        )

        LOGGER.info("Captured reproducibility snapshot")
        return self.snapshot

    def save_snapshot(
        self,
        snapshot_path: Path | str,
        snapshot: ReproducibilitySnapshot | None = None,
    ) -> None:
        """Save reproducibility snapshot.

        Args:
            snapshot_path: Path to save snapshot
            snapshot: Snapshot to save (None = current)
        """
        if snapshot is None:
            if self.snapshot is None:
                raise ValueError("No snapshot to save. Call capture_snapshot() first.")
            snapshot = self.snapshot

        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        with snapshot_path.open("w") as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)

        LOGGER.info("Saved reproducibility snapshot to: %s", snapshot_path)

    def load_snapshot(
        self,
        snapshot_path: Path | str,
    ) -> ReproducibilitySnapshot:
        """Load reproducibility snapshot.

        Args:
            snapshot_path: Path to snapshot file

        Returns:
            Loaded snapshot
        """
        snapshot_path = Path(snapshot_path)

        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        with snapshot_path.open() as f:
            data = json.load(f)

        snapshot = ReproducibilitySnapshot(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            seeds=data["seeds"],
            environment=data["environment"],
            dependencies=data["dependencies"],
            git_commit=data.get("git_commit"),
            torch_version=data["torch_version"],
            numpy_version=data["numpy_version"],
            python_version=data["python_version"],
            cuda_version=data.get("cuda_version"),
            cudnn_version=data.get("cudnn_version"),
            deterministic_mode=data.get("deterministic_mode", False),
            notes=data.get("notes", ""),
        )

        self.snapshot = snapshot

        LOGGER.info("Loaded reproducibility snapshot from: %s", snapshot_path)
        return snapshot

    def validate_reproducibility(
        self,
        reference_snapshot: ReproducibilitySnapshot | Path | str,
    ) -> dict[str, Any]:
        """Validate current environment against reference snapshot.

        Args:
            reference_snapshot: Reference snapshot or path

        Returns:
            Validation results
        """
        # Load reference if path provided
        if isinstance(reference_snapshot, (Path, str)):
            reference_snapshot = self.load_snapshot(reference_snapshot)

        # Capture current state
        current_snapshot = self.capture_snapshot()

        # Compare
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
        }

        # Check seeds
        if current_snapshot.seeds["main_seed"] != reference_snapshot.seeds["main_seed"]:
            validation["is_valid"] = False
            validation["issues"].append(
                f"Seed mismatch: {current_snapshot.seeds['main_seed']} "
                f"!= {reference_snapshot.seeds['main_seed']}"
            )

        # Check PyTorch version
        if current_snapshot.torch_version != reference_snapshot.torch_version:
            validation["warnings"].append(
                f"PyTorch version mismatch: {current_snapshot.torch_version} "
                f"!= {reference_snapshot.torch_version}"
            )

        # Check NumPy version
        if current_snapshot.numpy_version != reference_snapshot.numpy_version:
            validation["warnings"].append(
                f"NumPy version mismatch: {current_snapshot.numpy_version} "
                f"!= {reference_snapshot.numpy_version}"
            )

        # Check Python version
        if current_snapshot.python_version != reference_snapshot.python_version:
            validation["warnings"].append(
                f"Python version mismatch: {current_snapshot.python_version} "
                f"!= {reference_snapshot.python_version}"
            )

        # Check CUDA version
        if current_snapshot.cuda_version != reference_snapshot.cuda_version:
            validation["warnings"].append(
                f"CUDA version mismatch: {current_snapshot.cuda_version} "
                f"!= {reference_snapshot.cuda_version}"
            )

        # Check deterministic mode
        if current_snapshot.deterministic_mode != reference_snapshot.deterministic_mode:
            validation["is_valid"] = False
            validation["issues"].append(
                f"Deterministic mode mismatch: {current_snapshot.deterministic_mode} "
                f"!= {reference_snapshot.deterministic_mode}"
            )

        # Check git commit
        if current_snapshot.git_commit and reference_snapshot.git_commit:
            if current_snapshot.git_commit != reference_snapshot.git_commit:
                validation["warnings"].append(
                    f"Git commit mismatch: {current_snapshot.git_commit[:8]} "
                    f"!= {reference_snapshot.git_commit[:8]}"
                )

        # Check key dependencies
        key_packages = ["torch", "numpy", "mlflow", "optuna"]
        for pkg in key_packages:
            current_version = current_snapshot.dependencies.get(pkg)
            reference_version = reference_snapshot.dependencies.get(pkg)

            if current_version != reference_version:
                validation["warnings"].append(
                    f"{pkg} version mismatch: {current_version} != {reference_version}"
                )

        LOGGER.info(
            "Validation complete: %s (issues=%d, warnings=%d)",
            "VALID" if validation["is_valid"] else "INVALID",
            len(validation["issues"]),
            len(validation["warnings"]),
        )

        return validation

    def compute_snapshot_hash(
        self,
        snapshot: ReproducibilitySnapshot | None = None,
    ) -> str:
        """Compute hash of snapshot for quick comparison.

        Args:
            snapshot: Snapshot to hash (None = current)

        Returns:
            Snapshot hash
        """
        if snapshot is None:
            if self.snapshot is None:
                raise ValueError("No snapshot available")
            snapshot = self.snapshot

        # Create deterministic string representation
        snapshot_str = json.dumps(asdict(snapshot), sort_keys=True, default=str)
        return hashlib.sha256(snapshot_str.encode()).hexdigest()[:16]


def ensure_reproducibility(
    seed: int = 42,
    deterministic: bool = True,
    snapshot_path: Path | str | None = None,
) -> ReproducibilityManager:
    """Ensure reproducibility (convenience function).

    Args:
        seed: Random seed
        deterministic: Enable deterministic mode
        snapshot_path: Optional path to save snapshot

    Returns:
        Reproducibility manager
    """
    manager = ReproducibilityManager(seed=seed, deterministic=deterministic)
    manager.set_seeds()
    manager.capture_snapshot()

    if snapshot_path:
        manager.save_snapshot(snapshot_path)

    return manager


def validate_reproducibility(
    reference_snapshot: Path | str,
) -> dict[str, Any]:
    """Validate reproducibility (convenience function).

    Args:
        reference_snapshot: Path to reference snapshot

    Returns:
        Validation results
    """
    manager = ReproducibilityManager()
    return manager.validate_reproducibility(reference_snapshot)
