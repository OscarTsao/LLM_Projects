#!/usr/bin/env python
"""Model lineage and provenance tracking (Phase 20).

This module provides:
- Model lineage tracking
- Data provenance
- Training pipeline tracking
- Dependency graphs
- Reproducibility information
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source information."""

    name: str
    version: str
    path: str
    checksum: str
    num_samples: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "checksum": self.checksum,
            "num_samples": self.num_samples,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class TrainingRun:
    """Training run information."""

    run_id: str
    experiment_id: str
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


@dataclass
class ModelLineage:
    """Complete model lineage information."""

    model_name: str
    version: str
    created_at: datetime

    # Data lineage
    data_sources: list[DataSource] = field(default_factory=list)
    preprocessing_steps: list[dict[str, Any]] = field(default_factory=list)

    # Training lineage
    training_run: TrainingRun | None = None
    parent_models: list[str] = field(default_factory=list)
    derived_models: list[str] = field(default_factory=list)

    # Code lineage
    code_version: str = ""
    git_commit: str = ""
    git_branch: str = ""
    code_checksum: str = ""

    # Environment lineage
    environment: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)

    # Additional metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            # Data lineage
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "preprocessing_steps": self.preprocessing_steps,
            # Training lineage
            "training_run": self.training_run.to_dict() if self.training_run else None,
            "parent_models": self.parent_models,
            "derived_models": self.derived_models,
            # Code lineage
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "code_checksum": self.code_checksum,
            # Environment lineage
            "environment": self.environment,
            "dependencies": self.dependencies,
            # Additional metadata
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelLineage:
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModelLineage instance
        """
        lineage = cls(
            model_name=data["model_name"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )

        # Data lineage
        if "data_sources" in data:
            lineage.data_sources = [
                DataSource(
                    name=ds["name"],
                    version=ds["version"],
                    path=ds["path"],
                    checksum=ds["checksum"],
                    num_samples=ds["num_samples"],
                    created_at=datetime.fromisoformat(ds["created_at"]),
                    metadata=ds.get("metadata", {}),
                )
                for ds in data["data_sources"]
            ]

        lineage.preprocessing_steps = data.get("preprocessing_steps", [])

        # Training lineage
        if data.get("training_run"):
            tr = data["training_run"]
            lineage.training_run = TrainingRun(
                run_id=tr["run_id"],
                experiment_id=tr["experiment_id"],
                started_at=datetime.fromisoformat(tr["started_at"]),
                ended_at=(
                    datetime.fromisoformat(tr["ended_at"]) if tr["ended_at"] else None
                ),
                duration_seconds=tr["duration_seconds"],
                hyperparameters=tr.get("hyperparameters", {}),
                metrics=tr.get("metrics", {}),
                artifacts=tr.get("artifacts", []),
            )

        lineage.parent_models = data.get("parent_models", [])
        lineage.derived_models = data.get("derived_models", [])

        # Code lineage
        lineage.code_version = data.get("code_version", "")
        lineage.git_commit = data.get("git_commit", "")
        lineage.git_branch = data.get("git_branch", "")
        lineage.code_checksum = data.get("code_checksum", "")

        # Environment lineage
        lineage.environment = data.get("environment", {})
        lineage.dependencies = data.get("dependencies", {})

        # Additional metadata
        lineage.tags = data.get("tags", [])
        lineage.notes = data.get("notes", "")

        return lineage


class LineageTracker:
    """Tracker for model lineage."""

    def __init__(self):
        """Initialize lineage tracker."""
        self.lineage_store: dict[str, ModelLineage] = {}
        LOGGER.info("Initialized LineageTracker")

    def track(
        self,
        lineage: ModelLineage,
    ) -> None:
        """Track model lineage.

        Args:
            lineage: Model lineage to track
        """
        key = f"{lineage.model_name}:{lineage.version}"
        self.lineage_store[key] = lineage
        LOGGER.info(f"Tracked lineage for {key}")

    def get_lineage(
        self,
        model_name: str,
        version: str,
    ) -> ModelLineage | None:
        """Get lineage for a model.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            ModelLineage if found
        """
        key = f"{model_name}:{version}"
        return self.lineage_store.get(key)

    def get_ancestors(
        self,
        model_name: str,
        version: str,
    ) -> list[ModelLineage]:
        """Get all ancestor models.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            List of ancestor lineages
        """
        lineage = self.get_lineage(model_name, version)
        if not lineage:
            return []

        ancestors = []
        for parent in lineage.parent_models:
            # Parse parent (format: "name:version")
            if ":" in parent:
                parent_name, parent_version = parent.split(":", 1)
                parent_lineage = self.get_lineage(parent_name, parent_version)
                if parent_lineage:
                    ancestors.append(parent_lineage)
                    # Recursively get parent's ancestors
                    ancestors.extend(self.get_ancestors(parent_name, parent_version))

        return ancestors

    def get_descendants(
        self,
        model_name: str,
        version: str,
    ) -> list[ModelLineage]:
        """Get all descendant models.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            List of descendant lineages
        """
        lineage = self.get_lineage(model_name, version)
        if not lineage:
            return []

        descendants = []
        for derived in lineage.derived_models:
            # Parse derived (format: "name:version")
            if ":" in derived:
                derived_name, derived_version = derived.split(":", 1)
                derived_lineage = self.get_lineage(derived_name, derived_version)
                if derived_lineage:
                    descendants.append(derived_lineage)
                    # Recursively get derived models
                    descendants.extend(
                        self.get_descendants(derived_name, derived_version)
                    )

        return descendants

    def get_lineage_graph(
        self,
        model_name: str,
        version: str,
    ) -> dict[str, Any]:
        """Get complete lineage graph.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Lineage graph dictionary
        """
        lineage = self.get_lineage(model_name, version)
        if not lineage:
            return {}

        ancestors = self.get_ancestors(model_name, version)
        descendants = self.get_descendants(model_name, version)

        return {
            "current": lineage.to_dict(),
            "ancestors": [a.to_dict() for a in ancestors],
            "descendants": [d.to_dict() for d in descendants],
            "depth": {
                "ancestors": len(ancestors),
                "descendants": len(descendants),
            },
        }

    def find_by_data_source(
        self,
        data_source_name: str,
    ) -> list[ModelLineage]:
        """Find models trained on a specific data source.

        Args:
            data_source_name: Data source name

        Returns:
            List of matching lineages
        """
        matching = []
        for lineage in self.lineage_store.values():
            for ds in lineage.data_sources:
                if ds.name == data_source_name:
                    matching.append(lineage)
                    break

        return matching

    def find_by_git_commit(
        self,
        git_commit: str,
    ) -> list[ModelLineage]:
        """Find models from a specific git commit.

        Args:
            git_commit: Git commit hash

        Returns:
            List of matching lineages
        """
        return [
            lineage
            for lineage in self.lineage_store.values()
            if lineage.git_commit == git_commit
        ]

    def export_lineage(
        self,
        model_name: str,
        version: str,
        output_path: str,
    ) -> bool:
        """Export lineage to JSON file.

        Args:
            model_name: Model name
            version: Model version
            output_path: Output file path

        Returns:
            True if successful
        """
        import json
        from pathlib import Path

        lineage_graph = self.get_lineage_graph(model_name, version)

        if not lineage_graph:
            return False

        try:
            with Path(output_path).open("w") as f:
                json.dump(lineage_graph, f, indent=2)
        except Exception:
            LOGGER.exception(f"Failed to export lineage to {output_path}")
            return False
        else:
            LOGGER.info(f"Exported lineage to {output_path}")
            return True


# Convenience function
def track_lineage(
    model_name: str,
    version: str,
    data_sources: list[dict[str, Any]],
    git_commit: str = "",
) -> ModelLineage:
    """Track model lineage (convenience function).

    Args:
        model_name: Model name
        version: Model version
        data_sources: List of data source dicts
        git_commit: Git commit hash

    Returns:
        ModelLineage instance
    """
    lineage = ModelLineage(
        model_name=model_name,
        version=version,
        created_at=datetime.now(),
        git_commit=git_commit,
    )

    # Add data sources
    for ds_dict in data_sources:
        lineage.data_sources.append(
            DataSource(
                name=ds_dict["name"],
                version=ds_dict.get("version", "1.0.0"),
                path=ds_dict["path"],
                checksum=ds_dict.get("checksum", ""),
                num_samples=ds_dict.get("num_samples", 0),
            )
        )

    tracker = LineageTracker()
    tracker.track(lineage)

    return lineage
