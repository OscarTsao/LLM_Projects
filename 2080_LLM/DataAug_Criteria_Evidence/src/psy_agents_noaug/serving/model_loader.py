#!/usr/bin/env python
"""Model loading for serving (Phase 29).

This module provides utilities for loading models from the registry
or filesystem for serving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load models for serving."""

    def __init__(self, device: str = "cpu"):
        """Initialize model loader.

        Args:
            device: Device to load models on (cpu, cuda, etc.)
        """
        self.device = device
        self.loaded_models: dict[str, Any] = {}

        LOGGER.info(f"Initialized ModelLoader (device={device})")

    def load_pytorch_model(
        self,
        model_path: Path | str,
        model_class: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load PyTorch model.

        Args:
            model_path: Path to model file
            model_class: Model class (if loading architecture)
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model
        """
        model_path = Path(model_path)

        LOGGER.info(f"Loading PyTorch model from {model_path}")

        if model_class is not None:
            # Load with architecture
            model = model_class(**kwargs)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            # Load saved model
            model = torch.load(model_path, map_location=self.device)

        model.to(self.device)
        model.eval()

        LOGGER.info("Model loaded successfully")
        return model

    def load_from_registry(
        self,
        registry: Any,
        model_name: str,
        version: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load model from registry.

        Args:
            registry: ModelRegistry instance
            model_name: Model name in registry
            version: Version to load (None for latest)
            **kwargs: Additional loading arguments

        Returns:
            Loaded model
        """
        LOGGER.info(f"Loading model {model_name} (version={version}) from registry")

        # Get model version
        model_version = registry.get_version(model_name, version)

        if model_version is None:
            msg = f"Model {model_name} version {version} not found in registry"
            raise ValueError(msg)

        # Load model
        model_path = model_version.model_path
        framework = model_version.framework

        if framework == "pytorch" or str(model_path).endswith((".pt", ".pth")):
            model = self.load_pytorch_model(model_path, **kwargs)
        else:
            msg = f"Unsupported framework: {framework}"
            raise ValueError(msg)

        # Cache model
        cache_key = f"{model_name}:{version or 'latest'}"
        self.loaded_models[cache_key] = model

        LOGGER.info(f"Model {model_name} loaded and cached")
        return model

    def get_cached_model(self, cache_key: str) -> Any | None:
        """Get cached model.

        Args:
            cache_key: Cache key

        Returns:
            Cached model or None
        """
        return self.loaded_models.get(cache_key)

    def clear_cache(self, cache_key: str | None = None) -> None:
        """Clear model cache.

        Args:
            cache_key: Specific key to clear (None for all)
        """
        if cache_key:
            if cache_key in self.loaded_models:
                del self.loaded_models[cache_key]
                LOGGER.info(f"Cleared cache for {cache_key}")
        else:
            self.loaded_models.clear()
            LOGGER.info("Cleared all model cache")


def load_model(
    model_path: Path | str,
    device: str = "cpu",
    **kwargs: Any,
) -> Any:
    """Load model (convenience function).

    Args:
        model_path: Path to model
        device: Device to load on
        **kwargs: Additional arguments

    Returns:
        Loaded model
    """
    loader = ModelLoader(device=device)
    return loader.load_pytorch_model(model_path, **kwargs)
