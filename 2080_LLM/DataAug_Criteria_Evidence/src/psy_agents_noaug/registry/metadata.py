#!/usr/bin/env python
"""Model metadata and versioning (Phase 28).

This module provides data structures for tracking model metadata,
versions, and associated information.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Single model version."""

    version: str
    model_path: Path | str
    created_at: datetime
    created_by: str
    
    stage: ModelStage = ModelStage.DEVELOPMENT
    
    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Training info
    training_params: dict[str, Any] = field(default_factory=dict)
    training_data: str = ""
    
    # Model info
    framework: str = ""
    model_type: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    
    # Governance
    compliance_status: dict[str, bool] = field(default_factory=dict)
    bias_metrics: dict[str, float] = field(default_factory=dict)
    
    # Additional metadata
    tags: list[str] = field(default_factory=list)
    description: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        data = asdict(self)
        # Convert datetime and enum
        data["created_at"] = self.created_at.isoformat()
        data["stage"] = self.stage.value
        data["model_path"] = str(self.model_path)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        """Create from dictionary.

        Args:
            data: Dictionary data

        Returns:
            ModelVersion instance
        """
        # Convert datetime and enum
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["stage"] = ModelStage(data["stage"])
        data["model_path"] = Path(data["model_path"])
        
        return cls(**data)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    name: str
    description: str
    created_at: datetime
    created_by: str
    
    # Versions
    versions: dict[str, ModelVersion] = field(default_factory=dict)
    latest_version: str | None = None
    
    # Production info
    production_version: str | None = None
    
    # Model family
    task: str = ""  # classification, regression, etc.
    domain: str = ""  # medical, finance, etc.
    
    # Tags and search
    tags: list[str] = field(default_factory=list)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_version(self, version: ModelVersion) -> None:
        """Add a new version.

        Args:
            version: Model version to add
        """
        self.versions[version.version] = version
        self.latest_version = version.version
        
        # Auto-promote to production if stage is production
        if version.stage == ModelStage.PRODUCTION:
            self.production_version = version.version

        LOGGER.info(f"Added version {version.version} to model {self.name}")

    def get_version(self, version: str) -> ModelVersion | None:
        """Get specific version.

        Args:
            version: Version string

        Returns:
            Model version or None
        """
        return self.versions.get(version)

    def get_latest_version(self) -> ModelVersion | None:
        """Get latest version.

        Returns:
            Latest model version or None
        """
        if self.latest_version:
            return self.versions.get(self.latest_version)
        return None

    def get_production_version(self) -> ModelVersion | None:
        """Get production version.

        Returns:
            Production model version or None
        """
        if self.production_version:
            return self.versions.get(self.production_version)
        return None

    def promote_version(self, version: str, stage: ModelStage) -> bool:
        """Promote version to stage.

        Args:
            version: Version to promote
            stage: Target stage

        Returns:
            True if promoted successfully
        """
        if version not in self.versions:
            LOGGER.error(f"Version {version} not found")
            return False

        self.versions[version].stage = stage

        # Update production version if promoted to production
        if stage == ModelStage.PRODUCTION:
            # Demote current production version if exists
            if self.production_version and self.production_version != version:
                self.versions[self.production_version].stage = ModelStage.STAGING

            self.production_version = version

        LOGGER.info(f"Promoted version {version} to {stage.value}")
        return True

    def list_versions(
        self,
        stage: ModelStage | None = None,
    ) -> list[ModelVersion]:
        """List versions, optionally filtered by stage.

        Args:
            stage: Filter by stage

        Returns:
            List of model versions
        """
        versions = list(self.versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        # Sort by creation time (newest first)
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "latest_version": self.latest_version,
            "production_version": self.production_version,
            "task": self.task,
            "domain": self.domain,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary.

        Args:
            data: Dictionary data

        Returns:
            ModelMetadata instance
        """
        # Convert datetime
        data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Convert versions
        versions_data = data.pop("versions", {})
        versions = {k: ModelVersion.from_dict(v) for k, v in versions_data.items()}
        data["versions"] = versions

        return cls(**data)
