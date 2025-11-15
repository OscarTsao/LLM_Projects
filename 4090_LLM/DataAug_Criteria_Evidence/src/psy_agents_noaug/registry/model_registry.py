#!/usr/bin/env python
"""Model registry for centralized model management (Phase 28).

This module provides a centralized registry for tracking, versioning,
and managing machine learning models.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from psy_agents_noaug.registry.metadata import ModelMetadata, ModelStage, ModelVersion

LOGGER = logging.getLogger(__name__)


class ModelRegistry:
    """Centralized model registry."""

    def __init__(self, registry_path: Path | str):
        """Initialize model registry.

        Args:
            registry_path: Path to registry storage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Index file
        self.index_file = self.registry_path / "index.json"

        # Load or create index
        self.models: dict[str, ModelMetadata] = {}
        self._load_index()

        LOGGER.info(f"Initialized ModelRegistry at {self.registry_path}")

    def _load_index(self) -> None:
        """Load registry index from disk."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                data = json.load(f)

            for name, model_data in data.items():
                self.models[name] = ModelMetadata.from_dict(model_data)

            LOGGER.info(f"Loaded {len(self.models)} models from registry")
        else:
            LOGGER.info("Created new registry index")

    def _save_index(self) -> None:
        """Save registry index to disk."""
        data = {name: model.to_dict() for name, model in self.models.items()}

        with open(self.index_file, "w") as f:
            json.dump(data, f, indent=2)

        LOGGER.debug("Saved registry index")

    def register_model(
        self,
        name: str,
        description: str,
        created_by: str,
        task: str = "",
        domain: str = "",
        tags: list[str] | None = None,
        **metadata: Any,
    ) -> ModelMetadata:
        """Register a new model.

        Args:
            name: Model name
            description: Model description
            created_by: Creator name
            task: Task type
            domain: Domain
            tags: Tags for search
            **metadata: Additional metadata

        Returns:
            Created model metadata
        """
        if name in self.models:
            msg = f"Model {name} already registered"
            raise ValueError(msg)

        model = ModelMetadata(
            name=name,
            description=description,
            created_at=datetime.now(),
            created_by=created_by,
            task=task,
            domain=domain,
            tags=tags or [],
            metadata=metadata,
        )

        self.models[name] = model
        self._save_index()

        LOGGER.info(f"Registered model: {name}")
        return model

    def add_version(
        self,
        name: str,
        version: str,
        model_path: Path | str,
        created_by: str,
        metrics: dict[str, float] | None = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        **kwargs: Any,
    ) -> ModelVersion:
        """Add a new version to a model.

        Args:
            name: Model name
            version: Version string
            model_path: Path to model files
            created_by: Creator name
            metrics: Performance metrics
            stage: Lifecycle stage
            **kwargs: Additional version metadata

        Returns:
            Created model version
        """
        if name not in self.models:
            msg = f"Model {name} not found"
            raise ValueError(msg)

        # Create version directory
        version_dir = self.registry_path / name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        model_path = Path(model_path)
        if model_path.is_file():
            shutil.copy2(model_path, version_dir / model_path.name)
            stored_path = version_dir / model_path.name
        elif model_path.is_dir():
            shutil.copytree(model_path, version_dir / "model", dirs_exist_ok=True)
            stored_path = version_dir / "model"
        else:
            msg = f"Model path not found: {model_path}"
            raise FileNotFoundError(msg)

        # Create version metadata
        model_version = ModelVersion(
            version=version,
            model_path=stored_path,
            created_at=datetime.now(),
            created_by=created_by,
            stage=stage,
            metrics=metrics or {},
            **kwargs,
        )

        # Add to model
        self.models[name].add_version(model_version)
        self._save_index()

        LOGGER.info(f"Added version {version} to model {name}")
        return model_version

    def get_model(self, name: str) -> ModelMetadata | None:
        """Get model by name.

        Args:
            name: Model name

        Returns:
            Model metadata or None
        """
        return self.models.get(name)

    def get_version(
        self,
        name: str,
        version: str | None = None,
    ) -> ModelVersion | None:
        """Get specific version of a model.

        Args:
            name: Model name
            version: Version string (None for latest)

        Returns:
            Model version or None
        """
        model = self.get_model(name)
        if not model:
            return None

        if version is None:
            return model.get_latest_version()

        return model.get_version(version)

    def promote_version(
        self,
        name: str,
        version: str,
        stage: ModelStage,
    ) -> bool:
        """Promote version to stage.

        Args:
            name: Model name
            version: Version string
            stage: Target stage

        Returns:
            True if promoted successfully
        """
        model = self.get_model(name)
        if not model:
            LOGGER.error(f"Model {name} not found")
            return False

        success = model.promote_version(version, stage)

        if success:
            self._save_index()

        return success

    def list_models(
        self,
        task: str | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelMetadata]:
        """List models with optional filters.

        Args:
            task: Filter by task
            domain: Filter by domain
            tags: Filter by tags (must have all)

        Returns:
            List of model metadata
        """
        models = list(self.models.values())

        if task:
            models = [m for m in models if m.task == task]

        if domain:
            models = [m for m in models if m.domain == domain]

        if tags:
            models = [m for m in models if all(tag in m.tags for tag in tags)]

        return models

    def search_models(self, query: str) -> list[ModelMetadata]:
        """Search models by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        query_lower = query.lower()
        matches = []

        for model in self.models.values():
            # Search in name
            if query_lower in model.name.lower():
                matches.append(model)
                continue

            # Search in description
            if query_lower in model.description.lower():
                matches.append(model)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in model.tags):
                matches.append(model)
                continue

        return matches

    def delete_model(self, name: str) -> bool:
        """Delete a model from registry.

        Args:
            name: Model name

        Returns:
            True if deleted successfully
        """
        if name not in self.models:
            LOGGER.error(f"Model {name} not found")
            return False

        # Delete model directory
        model_dir = self.registry_path / name
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # Remove from index
        del self.models[name]
        self._save_index()

        LOGGER.info(f"Deleted model: {name}")
        return True

    def get_summary(self) -> dict[str, Any]:
        """Get registry summary.

        Returns:
            Summary dictionary
        """
        # Count by stage
        stage_counts = {stage.value: 0 for stage in ModelStage}

        for model in self.models.values():
            for version in model.versions.values():
                stage_counts[version.stage.value] += 1

        # Count by task
        task_counts: dict[str, int] = {}
        for model in self.models.values():
            if model.task:
                task_counts[model.task] = task_counts.get(model.task, 0) + 1

        return {
            "total_models": len(self.models),
            "total_versions": sum(len(m.versions) for m in self.models.values()),
            "versions_by_stage": stage_counts,
            "models_by_task": task_counts,
            "registry_path": str(self.registry_path),
        }


def create_registry(registry_path: Path | str) -> ModelRegistry:
    """Create model registry (convenience function).

    Args:
        registry_path: Path to registry storage

    Returns:
        Model registry instance
    """
    return ModelRegistry(registry_path)
