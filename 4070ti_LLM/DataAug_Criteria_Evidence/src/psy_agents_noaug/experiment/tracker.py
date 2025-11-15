#!/usr/bin/env python
"""Experiment tracking utilities (Phase 15).

This module provides comprehensive experiment tracking including:
- Automatic metadata capture
- Git integration
- Environment snapshots
- Parameter logging
- Artifact management
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow

LOGGER = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for an experiment."""

    experiment_id: str
    experiment_name: str
    run_id: str
    start_time: datetime
    end_time: datetime | None = None

    # Git information
    git_commit: str | None = None
    git_branch: str | None = None
    git_remote: str | None = None
    git_is_dirty: bool = False

    # Environment
    python_version: str = ""
    platform_info: str = ""
    hostname: str = ""
    username: str = ""

    # Configuration
    config_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    # Parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Artifacts
    artifacts: list[str] = field(default_factory=list)

    # Tags
    tags: dict[str, str] = field(default_factory=dict)

    # Status
    status: str = "RUNNING"

    # Notes
    notes: str = ""


class ExperimentTracker:
    """Track experiments with comprehensive metadata."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            artifact_location: Artifact storage location
        """
        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self.experiment_id = experiment.experiment_id

        self.run_id: str | None = None
        self.metadata: ExperimentMetadata | None = None

        LOGGER.info(
            "Initialized ExperimentTracker (experiment=%s, id=%s)",
            experiment_name,
            self.experiment_id,
        )

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> str:
        """Start a new experiment run.

        Args:
            run_name: Name for the run
            tags: Tags to apply
            description: Run description

        Returns:
            Run ID
        """
        # Start MLflow run
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
            description=description,
        )

        self.run_id = run.info.run_id

        # Create metadata
        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            experiment_name=self.experiment_name,
            run_id=self.run_id,
            start_time=datetime.now(),
            tags=tags or {},
            notes=description or "",
        )

        # Capture environment metadata
        self._capture_environment()

        # Capture git metadata
        self._capture_git_info()

        LOGGER.info("Started run: %s", self.run_id)
        return self.run_id

    def _capture_environment(self) -> None:
        """Capture environment metadata."""
        if self.metadata is None:
            return

        import sys

        self.metadata.python_version = (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
        self.metadata.platform_info = platform.platform()
        self.metadata.hostname = platform.node()
        self.metadata.username = os.getenv("USER", "unknown")

        # Log to MLflow
        mlflow.set_tag("python_version", self.metadata.python_version)
        mlflow.set_tag("platform", self.metadata.platform_info)
        mlflow.set_tag("hostname", self.metadata.hostname)

    def _capture_git_info(self) -> None:
        """Capture git repository information."""
        if self.metadata is None:
            return

        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.metadata.git_commit = result.stdout.strip()

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.metadata.git_branch = result.stdout.strip()

            # Get remote URL
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.metadata.git_remote = result.stdout.strip()

            # Check if working directory is dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.metadata.git_is_dirty = bool(result.stdout.strip())

            # Log to MLflow
            mlflow.set_tag("git.commit", self.metadata.git_commit)
            mlflow.set_tag("git.branch", self.metadata.git_branch)
            mlflow.set_tag("git.remote", self.metadata.git_remote)
            mlflow.set_tag("git.is_dirty", str(self.metadata.git_is_dirty))

            LOGGER.info("Captured git info: %s@%s",
                       self.metadata.git_branch,
                       self.metadata.git_commit[:8])

        except (subprocess.CalledProcessError, FileNotFoundError):
            LOGGER.warning("Failed to capture git information")

    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration.

        Args:
            config: Configuration dictionary
        """
        if self.metadata is None:
            raise RuntimeError("No active run. Call start_run() first.")

        # Store config
        self.metadata.config = config

        # Compute config hash
        config_str = json.dumps(config, sort_keys=True)
        self.metadata.config_hash = hashlib.sha256(
            config_str.encode()
        ).hexdigest()[:16]

        # Log as artifact
        config_path = Path("config.json")
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

        mlflow.log_artifact(str(config_path))
        config_path.unlink()

        # Log hash as tag
        mlflow.set_tag("config_hash", self.metadata.config_hash)

        LOGGER.info("Logged config (hash=%s)", self.metadata.config_hash)

    def log_parameters(self, params: dict[str, Any]) -> None:
        """Log experiment parameters.

        Args:
            params: Parameters dictionary
        """
        if self.metadata is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.metadata.parameters.update(params)

        # Log to MLflow
        for key, value in params.items():
            # MLflow parameters must be strings
            mlflow.log_param(key, str(value))

        LOGGER.info("Logged %d parameters", len(params))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log experiment metrics.

        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        if self.metadata is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.metadata.metrics.update(metrics)

        # Log to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

        LOGGER.info("Logged %d metrics", len(metrics))

    def log_artifact(self, artifact_path: Path | str) -> None:
        """Log an artifact.

        Args:
            artifact_path: Path to artifact
        """
        if self.metadata is None:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        mlflow.log_artifact(str(artifact_path))
        self.metadata.artifacts.append(str(artifact_path))

        LOGGER.info("Logged artifact: %s", artifact_path)

    def end_run(
        self,
        status: str = "FINISHED",
    ) -> ExperimentMetadata:
        """End the current run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)

        Returns:
            Experiment metadata
        """
        if self.metadata is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.metadata.end_time = datetime.now()
        self.metadata.status = status

        # Save metadata as artifact
        metadata_path = Path("experiment_metadata.json")
        with metadata_path.open("w") as f:
            json.dump(asdict(self.metadata), f, indent=2, default=str)

        mlflow.log_artifact(str(metadata_path))
        metadata_path.unlink()

        # End MLflow run
        mlflow.end_run(status=status)

        LOGGER.info("Ended run: %s (status=%s)", self.run_id, status)

        return self.metadata

    def get_run_info(self, run_id: str | None = None) -> dict[str, Any]:
        """Get information about a run.

        Args:
            run_id: Run ID (None = current run)

        Returns:
            Run information
        """
        if run_id is None:
            run_id = self.run_id

        if run_id is None:
            raise ValueError("No run ID provided and no active run")

        run = mlflow.get_run(run_id)

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }


def track_experiment(
    experiment_name: str,
    config: dict[str, Any],
    tracking_uri: str | None = None,
) -> ExperimentTracker:
    """Create and start experiment tracker (convenience function).

    Args:
        experiment_name: Experiment name
        config: Configuration
        tracking_uri: MLflow tracking URI

    Returns:
        Started tracker
    """
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    tracker.start_run()
    tracker.log_config(config)

    return tracker
