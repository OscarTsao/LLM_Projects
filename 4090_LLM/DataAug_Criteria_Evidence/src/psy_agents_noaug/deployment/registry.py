#!/usr/bin/env python
"""Model registry integration (Phase 14).

This module provides MLflow Model Registry integration for managing
model versions and lifecycle.

Key Features:
- Register models from checkpoints
- Version management
- Model staging (Staging, Production, Archived)
- Model tagging and metadata
- Model search and discovery
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""

    name: str
    version: int
    stage: str
    tags: dict[str, str]
    description: str | None = None
    creation_timestamp: datetime | None = None
    last_updated_timestamp: datetime | None = None
    source: str | None = None


class ModelRegistry:
    """MLflow Model Registry integration."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ):
        """Initialize model registry.

        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow registry URI (defaults to tracking URI)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        self.client = MlflowClient()

        LOGGER.info(
            "Initialized ModelRegistry (tracking=%s, registry=%s)",
            tracking_uri or "default",
            registry_uri or "default",
        )

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> int:
        """Register model in registry.

        Args:
            model_uri: URI to model (runs:/... or models:/...)
            model_name: Name for registered model
            tags: Tags to apply
            description: Model description

        Returns:
            Model version number
        """
        LOGGER.info("Registering model: %s", model_name)

        # Register model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        # Add tags
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(model_name, version, key, value)

        # Add description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description,
            )

        LOGGER.info("Registered %s version %s", model_name, version)
        return version

    def register_model_from_checkpoint(
        self,
        model_name: str,
        checkpoint_path: Path | str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> int:
        """Register model from checkpoint file.

        Args:
            model_name: Name for registered model
            checkpoint_path: Path to checkpoint file
            tags: Tags to apply
            description: Model description

        Returns:
            Model version number
        """
        import torch

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        LOGGER.info("Registering model from checkpoint: %s", checkpoint_path)

        # Load model from checkpoint
        # weights_only=False is safe here as we trust our own checkpoints
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Log model using PyTorch integration
        with mlflow.start_run() as run:
            # Log model with MLflow PyTorch
            mlflow.pytorch.log_model(model, "model")

            # Log metadata
            if tags:
                mlflow.set_tags(tags)

            model_uri = f"runs:/{run.info.run_id}/model"

        # Register model
        return self.register_model(
            model_uri=model_uri,
            model_name=model_name,
            tags=tags,
            description=description,
        )

    def transition_model_stage(
        self,
        model_name: str,
        version: int | str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Transition model to a stage.

        Args:
            model_name: Model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
        """
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")

        LOGGER.info(
            "Transitioning %s version %s to %s",
            model_name,
            version,
            stage,
        )

        # Archive existing if requested
        if archive_existing and stage in ["Staging", "Production"]:
            existing = self.client.get_latest_versions(model_name, stages=[stage])
            for model_version in existing:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Archived",
                )
                LOGGER.info("Archived existing version %s", model_version.version)

        # Transition to new stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )

        LOGGER.info("Successfully transitioned to %s", stage)

    def get_model_version(
        self,
        model_name: str,
        version: int | str | None = None,
        stage: str | None = None,
    ) -> ModelMetadata:
        """Get model version metadata.

        Args:
            model_name: Model name
            version: Model version (None = latest)
            stage: Filter by stage (None = any stage)

        Returns:
            Model metadata
        """
        if version is not None:
            # Get specific version
            model_version = self.client.get_model_version(model_name, version)
        elif stage is not None:
            # Get latest version in stage
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ValueError(f"No models found in stage: {stage}")
            model_version = versions[0]
        else:
            # Get latest version overall
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"Model not found: {model_name}")
            model_version = max(versions, key=lambda v: int(v.version))

        # Extract metadata
        return ModelMetadata(
            name=model_version.name,
            version=int(model_version.version),
            stage=model_version.current_stage,
            tags=model_version.tags or {},
            description=model_version.description,
            creation_timestamp=datetime.fromtimestamp(
                model_version.creation_timestamp / 1000
            ),
            last_updated_timestamp=datetime.fromtimestamp(
                model_version.last_updated_timestamp / 1000
            ),
            source=model_version.source,
        )

    def list_models(
        self,
        filter_string: str | None = None,
    ) -> list[str]:
        """List registered models.

        Args:
            filter_string: Filter string (optional)

        Returns:
            List of model names
        """
        models = self.client.search_registered_models(filter_string=filter_string)
        return [model.name for model in models]

    def delete_model_version(
        self,
        model_name: str,
        version: int | str,
    ) -> None:
        """Delete a model version.

        Args:
            model_name: Model name
            version: Version to delete
        """
        LOGGER.warning("Deleting %s version %s", model_name, version)
        self.client.delete_model_version(model_name, version)

    def delete_registered_model(self, model_name: str) -> None:
        """Delete entire registered model.

        Args:
            model_name: Model name to delete
        """
        LOGGER.warning("Deleting registered model: %s", model_name)
        self.client.delete_registered_model(model_name)


def get_production_models(
    tracking_uri: str | None = None,
) -> dict[str, ModelMetadata]:
    """Get all models in Production stage.

    Args:
        tracking_uri: MLflow tracking URI

    Returns:
        Dict mapping model name to metadata
    """
    registry = ModelRegistry(tracking_uri=tracking_uri)

    production_models = {}
    model_names = registry.list_models()

    for model_name in model_names:
        try:
            metadata = registry.get_model_version(
                model_name=model_name,
                stage="Production",
            )
            production_models[model_name] = metadata
        except ValueError:
            # No production version for this model
            continue

    LOGGER.info("Found %d models in Production", len(production_models))
    return production_models
