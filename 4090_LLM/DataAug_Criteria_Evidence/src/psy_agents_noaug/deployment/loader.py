#!/usr/bin/env python
"""Model loading utilities (Phase 14).

This module provides utilities for loading models from the registry
for inference and deployment.

Key Features:
- Load models from MLflow registry
- Stage-based loading (Production, Staging)
- Model caching
- Version pinning
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import torch

from psy_agents_noaug.deployment.registry import ModelRegistry

LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load models from registry."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        cache_dir: Path | str | None = None,
    ):
        """Initialize model loader.

        Args:
            registry: Model registry instance
            cache_dir: Directory for caching models
        """
        self.registry = registry or ModelRegistry()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Initialized ModelLoader")

    def load_model(
        self,
        model_name: str,
        version: int | str | None = None,
        stage: str | None = None,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """Load model from registry.

        Args:
            model_name: Model name
            version: Model version (None = latest or by stage)
            stage: Model stage (None = latest version)
            device: Device to load model on

        Returns:
            Loaded PyTorch model
        """
        # Get model version
        metadata = self.registry.get_model_version(
            model_name=model_name,
            version=version,
            stage=stage,
        )

        LOGGER.info(
            "Loading %s version %s (stage=%s)",
            model_name,
            metadata.version,
            metadata.stage,
        )

        # Load model from MLflow
        model_uri = f"models:/{model_name}/{metadata.version}"

        try:
            # Try loading as PyTorch model
            model = mlflow.pytorch.load_model(model_uri, map_location=device)
        except Exception as e:
            LOGGER.warning("Failed to load as pytorch model: %s", e)
            # Try loading as generic model
            model_path = mlflow.artifacts.download_artifacts(model_uri)
            # weights_only=False is safe here as we trust our own checkpoints
            model = torch.load(
                Path(model_path) / "model.pt",
                map_location=device,
                weights_only=False,
            )

        LOGGER.info("Successfully loaded model")
        return model

    def load_production_model(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """Load production version of model.

        Args:
            model_name: Model name
            device: Device to load on

        Returns:
            Loaded model
        """
        return self.load_model(
            model_name=model_name,
            stage="Production",
            device=device,
        )

    def load_staging_model(
        self,
        model_name: str,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """Load staging version of model.

        Args:
            model_name: Model name
            device: Device to load on

        Returns:
            Loaded model
        """
        return self.load_model(
            model_name=model_name,
            stage="Staging",
            device=device,
        )


def load_model_from_registry(
    model_name: str,
    version: int | str | None = None,
    stage: str | None = None,
    tracking_uri: str | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load model from registry (convenience function).

    Args:
        model_name: Model name
        version: Model version
        stage: Model stage
        tracking_uri: MLflow tracking URI
        device: Device to load on

    Returns:
        Loaded model
    """
    registry = ModelRegistry(tracking_uri=tracking_uri)
    loader = ModelLoader(registry=registry)

    return loader.load_model(
        model_name=model_name,
        version=version,
        stage=stage,
        device=device,
    )
