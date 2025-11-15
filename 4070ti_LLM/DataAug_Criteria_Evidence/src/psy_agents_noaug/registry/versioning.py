#!/usr/bin/env python
"""Model versioning system (Phase 20).

This module provides:
- Semantic versioning (major.minor.patch)
- Model version management
- Version comparison and ordering
- Tag-based versioning (latest, stable, etc.)
- Version history tracking
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class SemanticVersion:
    """Semantic version (major.minor.patch)."""

    major: int
    minor: int
    patch: int
    prerelease: str | None = None
    build: str | None = None

    def __str__(self) -> str:
        """String representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: SemanticVersion) -> bool:
        """Compare versions."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch

        # Prerelease versions have lower precedence
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False

        return False

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        """Hash for dict keys."""
        return hash((self.major, self.minor, self.patch, self.prerelease))

    @classmethod
    def from_string(cls, version_str: str) -> SemanticVersion:
        """Parse version from string.

        Args:
            version_str: Version string (e.g., "1.2.3", "1.0.0-alpha+build123")

        Returns:
            SemanticVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        # Parse semantic version
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_str)

        if not match:
            msg = f"Invalid semantic version: {version_str}"
            raise ValueError(msg)

        major, minor, patch, prerelease, build = match.groups()

        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build,
        )

    def bump_major(self) -> SemanticVersion:
        """Bump major version."""
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> SemanticVersion:
        """Bump minor version."""
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> SemanticVersion:
        """Bump patch version."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


@dataclass
class ModelVersion:
    """Model version information."""

    version: SemanticVersion
    model_name: str
    stage: str  # dev, staging, production
    created_at: datetime
    created_by: str
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "version": str(self.version),
            "model_name": self.model_name,
            "stage": self.stage,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModelVersion instance
        """
        return cls(
            version=SemanticVersion.from_string(data["version"]),
            model_name=data["model_name"],
            stage=data["stage"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            tags=data.get("tags", []),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            description=data.get("description", ""),
        )


class ModelRegistry:
    """Model registry for version management."""

    def __init__(self, registry_path: str | Path | None = None):
        """Initialize model registry.

        Args:
            registry_path: Path to registry storage
        """
        self.registry_path = (
            Path(registry_path) if registry_path else Path("model_registry")
        )
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # In-memory version tracking
        self.versions: dict[str, list[ModelVersion]] = {}
        self.tags: dict[str, dict[str, SemanticVersion]] = {}

        LOGGER.info(f"Initialized ModelRegistry at {self.registry_path}")

    def register_version(
        self,
        model_name: str,
        version: SemanticVersion | str,
        stage: str = "dev",
        created_by: str = "system",
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
        description: str = "",
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_name: Name of the model
            version: Semantic version
            stage: Deployment stage
            created_by: Creator identifier
            tags: Version tags
            metrics: Model metrics
            artifacts: Artifact paths
            description: Version description

        Returns:
            Registered ModelVersion
        """
        if isinstance(version, str):
            version = SemanticVersion.from_string(version)

        model_version = ModelVersion(
            version=version,
            model_name=model_name,
            stage=stage,
            created_at=datetime.now(),
            created_by=created_by,
            tags=tags or [],
            metrics=metrics or {},
            artifacts=artifacts or {},
            description=description,
        )

        # Add to registry
        if model_name not in self.versions:
            self.versions[model_name] = []

        self.versions[model_name].append(model_version)

        # Update tags
        for tag in model_version.tags:
            if model_name not in self.tags:
                self.tags[model_name] = {}
            self.tags[model_name][tag] = version

        LOGGER.info(f"Registered {model_name} version {version} in stage {stage}")

        return model_version

    def get_version(
        self,
        model_name: str,
        version: SemanticVersion | str | None = None,
        tag: str | None = None,
    ) -> ModelVersion | None:
        """Get a specific model version.

        Args:
            model_name: Model name
            version: Specific version (optional)
            tag: Version tag (optional)

        Returns:
            ModelVersion if found, None otherwise
        """
        if model_name not in self.versions:
            return None

        # Get by tag
        if tag:
            if model_name in self.tags and tag in self.tags[model_name]:
                version = self.tags[model_name][tag]
            else:
                return None

        # Get by version
        if version:
            if isinstance(version, str):
                version = SemanticVersion.from_string(version)

            for mv in self.versions[model_name]:
                if mv.version == version:
                    return mv
            return None

        # Get latest
        return max(
            self.versions[model_name],
            key=lambda v: v.version,
        )

    def list_versions(
        self,
        model_name: str,
        stage: str | None = None,
    ) -> list[ModelVersion]:
        """List all versions of a model.

        Args:
            model_name: Model name
            stage: Filter by stage (optional)

        Returns:
            List of ModelVersions
        """
        if model_name not in self.versions:
            return []

        versions = self.versions[model_name]

        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.version, reverse=True)

    def get_latest_version(
        self,
        model_name: str,
        stage: str | None = None,
    ) -> ModelVersion | None:
        """Get the latest version of a model.

        Args:
            model_name: Model name
            stage: Filter by stage (optional)

        Returns:
            Latest ModelVersion if found
        """
        versions = self.list_versions(model_name, stage)
        return versions[0] if versions else None

    def tag_version(
        self,
        model_name: str,
        version: SemanticVersion | str,
        tag: str,
    ) -> bool:
        """Tag a specific version.

        Args:
            model_name: Model name
            version: Version to tag
            tag: Tag name

        Returns:
            True if successful
        """
        if isinstance(version, str):
            version = SemanticVersion.from_string(version)

        model_version = self.get_version(model_name, version)
        if not model_version:
            return False

        # Add tag
        if tag not in model_version.tags:
            model_version.tags.append(tag)

        # Update tag mapping
        if model_name not in self.tags:
            self.tags[model_name] = {}
        self.tags[model_name][tag] = version

        LOGGER.info(f"Tagged {model_name}:{version} as '{tag}'")
        return True

    def delete_version(
        self,
        model_name: str,
        version: SemanticVersion | str,
    ) -> bool:
        """Delete a model version.

        Args:
            model_name: Model name
            version: Version to delete

        Returns:
            True if successful
        """
        if isinstance(version, str):
            version = SemanticVersion.from_string(version)

        if model_name not in self.versions:
            return False

        # Remove version
        original_count = len(self.versions[model_name])
        self.versions[model_name] = [
            v for v in self.versions[model_name] if v.version != version
        ]

        success = len(self.versions[model_name]) < original_count

        if success:
            LOGGER.info(f"Deleted {model_name}:{version}")

        return success


# Convenience function
def create_version(version_str: str) -> SemanticVersion:
    """Create a semantic version (convenience function).

    Args:
        version_str: Version string (e.g., "1.2.3")

    Returns:
        SemanticVersion instance
    """
    return SemanticVersion.from_string(version_str)
