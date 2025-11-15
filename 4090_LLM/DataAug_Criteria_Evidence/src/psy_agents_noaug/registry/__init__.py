#!/usr/bin/env python
"""Model registry and versioning (Phase 20 + Phase 28 Enhanced).

This module provides comprehensive model management including:
- Model versioning and tagging (Phase 20)
- Model lineage and provenance (Phase 20)
- Promotion workflows (Phase 20)
- Centralized registry with lifecycle management (Phase 28)
- Enhanced metadata tracking (Phase 28)
- Search and discovery (Phase 28)

Key Features:
- Semantic versioning support
- Stage-based promotion (development/staging/production/archived)
- Performance metrics and governance tracking
- Lineage tracking for reproducibility
- Tag-based search and filtering
- Integration with MLflow Model Registry
"""

from __future__ import annotations

# Phase 20 - Original registry functionality
from psy_agents_noaug.registry.lineage import (
    LineageTracker,
    ModelLineage,
    track_lineage,
)
from psy_agents_noaug.registry.promotion import (
    ModelPromoter,
    PromotionCriteria,
    promote_model,
)
from psy_agents_noaug.registry.versioning import (
    ModelVersion as ModelVersionV1,
    VersionManager,
    create_version,
)

# Phase 28 - Enhanced registry with centralized management
from psy_agents_noaug.registry.metadata import (
    ModelMetadata,
    ModelStage,
    ModelVersion,
)
from psy_agents_noaug.registry.model_registry import (
    ModelRegistry,
    create_registry,
)

__all__ = [
    # Phase 20 - Original
    "LineageTracker",
    "ModelLineage",
    "track_lineage",
    "ModelPromoter",
    "PromotionCriteria",
    "promote_model",
    "ModelVersionV1",
    "VersionManager",
    "create_version",
    # Phase 28 - Enhanced
    "ModelMetadata",
    "ModelVersion",
    "ModelStage",
    "ModelRegistry",
    "create_registry",
]
