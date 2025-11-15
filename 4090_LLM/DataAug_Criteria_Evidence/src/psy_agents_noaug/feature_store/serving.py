#!/usr/bin/env python
"""Feature serving layer (Phase 24).

This module provides tools for serving features in production, with support
for online and batch serving, feature sets, and monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Collection of features for serving."""

    name: str
    features: list[str]
    version: str = "1.0.0"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_feature_names(self) -> list[str]:
        """Get list of feature names.

        Returns:
            List of feature names
        """
        return self.features

    def get_num_features(self) -> int:
        """Get number of features.

        Returns:
            Number of features
        """
        return len(self.features)


@dataclass
class ServingRequest:
    """Feature serving request."""

    feature_set: str
    entity_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServingResponse:
    """Feature serving response."""

    feature_set: str
    features: dict[str, Any]
    served_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureServer:
    """Server for online and batch feature serving."""

    def __init__(self, enable_cache: bool = True):
        """Initialize feature server.

        Args:
            enable_cache: Enable feature caching
        """
        self.enable_cache = enable_cache
        self.feature_sets: dict[str, FeatureSet] = {}
        self.feature_cache: dict[str, dict[str, Any]] = {}
        self.serving_stats: dict[str, int] = {}
        LOGGER.info(f"Initialized FeatureServer (cache={enable_cache})")

    def register_feature_set(self, feature_set: FeatureSet) -> None:
        """Register a feature set.

        Args:
            feature_set: Feature set to register
        """
        self.feature_sets[feature_set.name] = feature_set
        self.serving_stats[feature_set.name] = 0
        LOGGER.info(f"Registered feature set: {feature_set.name}")

    def get_feature_set(self, name: str) -> FeatureSet | None:
        """Get feature set by name.

        Args:
            name: Feature set name

        Returns:
            Feature set or None
        """
        return self.feature_sets.get(name)

    def serve_online(
        self,
        request: ServingRequest,
        features_data: dict[str, Any],
    ) -> ServingResponse:
        """Serve features online (low latency).

        Args:
            request: Serving request
            features_data: Pre-computed feature values

        Returns:
            Serving response
        """
        start_time = datetime.now()

        # Get feature set
        feature_set = self.get_feature_set(request.feature_set)
        if not feature_set:
            msg = f"Feature set not found: {request.feature_set}"
            raise ValueError(msg)

        # Check cache
        cache_key = f"{request.feature_set}:{request.entity_id}"
        cached = False

        if self.enable_cache and cache_key in self.feature_cache:
            features = self.feature_cache[cache_key]
            cached = True
            LOGGER.debug(f"Cache hit for {cache_key}")
        else:
            # Get features
            features = {name: features_data.get(name) for name in feature_set.features}

            # Cache features
            if self.enable_cache:
                self.feature_cache[cache_key] = features

        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds() * 1000

        # Update stats
        self.serving_stats[request.feature_set] += 1

        return ServingResponse(
            feature_set=request.feature_set,
            features=features,
            served_at=end_time,
            latency_ms=latency,
            cached=cached,
        )

    def serve_batch(
        self,
        feature_set_name: str,
        batch_data: list[dict[str, Any]],
    ) -> list[ServingResponse]:
        """Serve features in batch (high throughput).

        Args:
            feature_set_name: Feature set name
            batch_data: List of feature data dictionaries

        Returns:
            List of serving responses
        """
        feature_set = self.get_feature_set(feature_set_name)
        if not feature_set:
            msg = f"Feature set not found: {feature_set_name}"
            raise ValueError(msg)

        responses = []
        for i, data in enumerate(batch_data):
            # Use index as entity_id to avoid cache collisions
            request = ServingRequest(
                feature_set=feature_set_name,
                entity_id=f"batch_{i}",
            )
            response = self.serve_online(request, data)
            responses.append(response)

        return responses

    def get_feature_vector(
        self,
        feature_set_name: str,
        features_data: dict[str, Any],
        feature_order: list[str] | None = None,
    ) -> np.ndarray:
        """Get feature vector for model input.

        Args:
            feature_set_name: Feature set name
            features_data: Feature values
            feature_order: Order of features (default: feature set order)

        Returns:
            Feature vector (numpy array)
        """
        feature_set = self.get_feature_set(feature_set_name)
        if not feature_set:
            msg = f"Feature set not found: {feature_set_name}"
            raise ValueError(msg)

        # Use feature set order if not specified
        if feature_order is None:
            feature_order = feature_set.features

        # Build vector
        vector = []
        for name in feature_order:
            value = features_data.get(name)
            if value is None:
                msg = f"Missing feature: {name}"
                raise ValueError(msg)

            # Convert to numeric
            if isinstance(value, int | float | bool):
                vector.append(float(value))
            elif isinstance(value, list | np.ndarray):
                # Flatten arrays
                vector.extend([float(v) for v in value])
            else:
                msg = f"Cannot convert feature {name} to numeric: {type(value)}"
                raise TypeError(msg)

        return np.array(vector)

    def invalidate_cache(
        self,
        feature_set_name: str | None = None,
        entity_id: str | None = None,
    ) -> None:
        """Invalidate feature cache.

        Args:
            feature_set_name: Specific feature set (None for all)
            entity_id: Specific entity (None for all)
        """
        if feature_set_name and entity_id:
            # Invalidate specific entry
            cache_key = f"{feature_set_name}:{entity_id}"
            if cache_key in self.feature_cache:
                del self.feature_cache[cache_key]
                LOGGER.info(f"Invalidated cache for {cache_key}")
        elif feature_set_name:
            # Invalidate all for feature set
            keys_to_remove = [
                k for k in self.feature_cache if k.startswith(f"{feature_set_name}:")
            ]
            for key in keys_to_remove:
                del self.feature_cache[key]
            LOGGER.info(f"Invalidated cache for feature set {feature_set_name}")
        else:
            # Invalidate all
            self.feature_cache.clear()
            LOGGER.info("Invalidated all cache")

    def get_serving_stats(self) -> dict[str, Any]:
        """Get serving statistics.

        Returns:
            Statistics dictionary
        """
        total_requests = sum(self.serving_stats.values())
        cache_size = len(self.feature_cache)

        return {
            "total_requests": total_requests,
            "cache_size": cache_size,
            "feature_sets": len(self.feature_sets),
            "requests_by_feature_set": self.serving_stats.copy(),
        }

    def list_feature_sets(self) -> list[str]:
        """List all registered feature sets.

        Returns:
            List of feature set names
        """
        return list(self.feature_sets.keys())


def serve_features(
    feature_set_name: str,
    features_data: dict[str, Any],
    feature_sets: dict[str, FeatureSet],
) -> dict[str, Any]:
    """Serve features (convenience function).

    Args:
        feature_set_name: Feature set name
        features_data: Feature values
        feature_sets: Dictionary of feature sets

    Returns:
        Selected features
    """
    if feature_set_name not in feature_sets:
        msg = f"Feature set not found: {feature_set_name}"
        raise ValueError(msg)

    feature_set = feature_sets[feature_set_name]

    return {name: features_data.get(name) for name in feature_set.features}
