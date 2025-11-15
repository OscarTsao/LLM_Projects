#!/usr/bin/env python
"""Feature registry (Phase 24).

This module provides tools for registering, organizing, and managing feature
metadata in a feature store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Feature data types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TEXT = "text"
    TIMESTAMP = "timestamp"


@dataclass
class Feature:
    """Feature definition."""

    name: str
    feature_type: FeatureType
    description: str
    group: str | None = None
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Computation
    compute_fn: Callable[[Any], Any] | None = None
    dependencies: list[str] = field(default_factory=list)

    def get_full_name(self) -> str:
        """Get fully qualified feature name.

        Returns:
            Full name (group.name:version)
        """
        if self.group:
            return f"{self.group}.{self.name}:{self.version}"
        return f"{self.name}:{self.version}"

    def compute(self, data: Any) -> Any:
        """Compute feature value.

        Args:
            data: Input data

        Returns:
            Computed feature value
        """
        if self.compute_fn is None:
            msg = f"No computation function defined for feature {self.name}"
            raise ValueError(msg)

        return self.compute_fn(data)


@dataclass
class FeatureGroup:
    """Feature group for organizing related features."""

    name: str
    description: str
    features: list[Feature] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_feature(self, feature: Feature) -> None:
        """Add feature to group.

        Args:
            feature: Feature to add
        """
        feature.group = self.name
        self.features.append(feature)
        LOGGER.info(f"Added feature {feature.name} to group {self.name}")

    def get_feature(self, name: str) -> Feature | None:
        """Get feature by name.

        Args:
            name: Feature name

        Returns:
            Feature or None if not found
        """
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def list_features(self) -> list[str]:
        """List all feature names in group.

        Returns:
            List of feature names
        """
        return [f.name for f in self.features]


class FeatureRegistry:
    """Registry for managing features."""

    def __init__(self):
        """Initialize feature registry."""
        self.features: dict[str, Feature] = {}
        self.groups: dict[str, FeatureGroup] = {}
        LOGGER.info("Initialized FeatureRegistry")

    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        description: str,
        **kwargs: Any,
    ) -> Feature:
        """Register a new feature.

        Args:
            name: Feature name
            feature_type: Feature type
            description: Feature description
            **kwargs: Additional feature properties

        Returns:
            Registered feature
        """
        # Create feature
        feature = Feature(
            name=name,
            feature_type=feature_type,
            description=description,
            **kwargs,
        )

        # Register
        full_name = feature.get_full_name()
        self.features[full_name] = feature

        # Add to group if specified
        if feature.group and feature.group in self.groups:
            self.groups[feature.group].add_feature(feature)

        LOGGER.info(f"Registered feature: {full_name}")
        return feature

    def register_group(
        self,
        name: str,
        description: str,
        **kwargs: Any,
    ) -> FeatureGroup:
        """Register a feature group.

        Args:
            name: Group name
            description: Group description
            **kwargs: Additional group properties

        Returns:
            Registered group
        """
        group = FeatureGroup(
            name=name,
            description=description,
            **kwargs,
        )

        self.groups[name] = group
        LOGGER.info(f"Registered feature group: {name}")
        return group

    def get_feature(self, name: str, version: str = "1.0.0") -> Feature | None:
        """Get feature by name and version.

        Args:
            name: Feature name
            version: Feature version

        Returns:
            Feature or None if not found
        """
        # Try with version
        full_name = f"{name}:{version}"
        if full_name in self.features:
            return self.features[full_name]

        # Try without version (default)
        for _full_name, feature in self.features.items():
            if feature.name == name:
                return feature

        return None

    def get_group(self, name: str) -> FeatureGroup | None:
        """Get feature group by name.

        Args:
            name: Group name

        Returns:
            Feature group or None
        """
        return self.groups.get(name)

    def list_features(
        self,
        group: str | None = None,
        feature_type: FeatureType | None = None,
        tags: list[str] | None = None,
    ) -> list[Feature]:
        """List features with optional filters.

        Args:
            group: Filter by group
            feature_type: Filter by type
            tags: Filter by tags (any match)

        Returns:
            List of matching features
        """
        features = list(self.features.values())

        # Filter by group
        if group:
            features = [f for f in features if f.group == group]

        # Filter by type
        if feature_type:
            features = [f for f in features if f.feature_type == feature_type]

        # Filter by tags
        if tags:
            features = [f for f in features if any(tag in f.tags for tag in tags)]

        return features

    def search_features(self, query: str) -> list[Feature]:
        """Search features by name or description.

        Args:
            query: Search query

        Returns:
            List of matching features
        """
        query_lower = query.lower()
        results = []

        for feature in self.features.values():
            if (
                query_lower in feature.name.lower()
                or query_lower in feature.description.lower()
            ):
                results.append(feature)

        return results

    def delete_feature(self, name: str, version: str = "1.0.0") -> bool:
        """Delete a feature.

        Args:
            name: Feature name
            version: Feature version

        Returns:
            True if deleted, False if not found
        """
        full_name = f"{name}:{version}"

        if full_name in self.features:
            feature = self.features[full_name]

            # Remove from group
            if feature.group and feature.group in self.groups:
                group = self.groups[feature.group]
                group.features = [f for f in group.features if f.name != name]

            # Remove from registry
            del self.features[full_name]
            LOGGER.info(f"Deleted feature: {full_name}")
            return True

        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Statistics dictionary
        """
        feature_types: dict[str, int] = {}
        for feature in self.features.values():
            type_name = feature.feature_type.value
            feature_types[type_name] = feature_types.get(type_name, 0) + 1

        return {
            "total_features": len(self.features),
            "total_groups": len(self.groups),
            "features_by_type": feature_types,
            "groups": {
                name: len(group.features) for name, group in self.groups.items()
            },
        }
