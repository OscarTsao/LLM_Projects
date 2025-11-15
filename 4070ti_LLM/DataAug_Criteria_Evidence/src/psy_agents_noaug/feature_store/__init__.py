#!/usr/bin/env python
"""Feature Store & Engineering (Phase 24).

This module provides tools for feature management, versioning, computation,
and serving for production ML systems.
"""

from __future__ import annotations

from psy_agents_noaug.feature_store.computation import (
    FeatureComputationEngine,
    compute_features,
)
from psy_agents_noaug.feature_store.registry import (
    Feature,
    FeatureGroup,
    FeatureRegistry,
    FeatureType,
)
from psy_agents_noaug.feature_store.serving import (
    FeatureServer,
    FeatureSet,
    serve_features,
)
from psy_agents_noaug.feature_store.versioning import (
    FeatureVersion,
    FeatureVersionManager,
    VersionStatus,
)

__all__ = [
    # Registry
    "Feature",
    "FeatureGroup",
    "FeatureRegistry",
    "FeatureType",
    # Versioning
    "FeatureVersion",
    "FeatureVersionManager",
    "VersionStatus",
    # Computation
    "FeatureComputationEngine",
    "compute_features",
    # Serving
    "FeatureServer",
    "FeatureSet",
    "serve_features",
]
