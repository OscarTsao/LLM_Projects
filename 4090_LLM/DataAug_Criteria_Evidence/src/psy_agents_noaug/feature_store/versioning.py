#!/usr/bin/env python
"""Feature versioning (Phase 24).

This module provides tools for versioning features, managing feature lifecycles,
and ensuring backward compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """Feature version status."""

    DRAFT = "draft"  # Under development
    ACTIVE = "active"  # In production use
    DEPRECATED = "deprecated"  # Still available but discouraged
    ARCHIVED = "archived"  # No longer available


@dataclass
class FeatureVersion:
    """Feature version metadata."""

    feature_name: str
    version: str
    status: VersionStatus
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: datetime | None = None
    deprecated_at: datetime | None = None
    archived_at: datetime | None = None

    # Version metadata
    description: str = ""
    changes: list[str] = field(default_factory=list)
    breaking_changes: bool = False
    backward_compatible: bool = True

    # Lineage
    parent_version: str | None = None
    child_versions: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_full_version(self) -> str:
        """Get fully qualified version.

        Returns:
            Full version string (name:version)
        """
        return f"{self.feature_name}:{self.version}"

    def can_transition_to(self, new_status: VersionStatus) -> bool:
        """Check if transition to new status is allowed.

        Args:
            new_status: Target status

        Returns:
            True if transition is allowed
        """
        # Define allowed transitions
        allowed: dict[VersionStatus, list[VersionStatus]] = {
            VersionStatus.DRAFT: [VersionStatus.ACTIVE, VersionStatus.ARCHIVED],
            VersionStatus.ACTIVE: [VersionStatus.DEPRECATED, VersionStatus.ARCHIVED],
            VersionStatus.DEPRECATED: [VersionStatus.ARCHIVED],
            VersionStatus.ARCHIVED: [],  # Cannot transition from archived
        }

        return new_status in allowed.get(self.status, [])


class FeatureVersionManager:
    """Manager for feature versioning."""

    def __init__(self):
        """Initialize version manager."""
        self.versions: dict[str, list[FeatureVersion]] = {}
        LOGGER.info("Initialized FeatureVersionManager")

    def create_version(
        self,
        feature_name: str,
        version: str,
        description: str = "",
        parent_version: str | None = None,
        **kwargs: Any,
    ) -> FeatureVersion:
        """Create a new feature version.

        Args:
            feature_name: Feature name
            version: Version string (e.g., "1.0.0")
            description: Version description
            parent_version: Parent version (for lineage)
            **kwargs: Additional version properties

        Returns:
            Created feature version
        """
        # Create version
        feature_version = FeatureVersion(
            feature_name=feature_name,
            version=version,
            status=VersionStatus.DRAFT,
            description=description,
            parent_version=parent_version,
            **kwargs,
        )

        # Store version
        if feature_name not in self.versions:
            self.versions[feature_name] = []
        self.versions[feature_name].append(feature_version)

        # Update parent's child list
        if parent_version:
            parent = self.get_version(feature_name, parent_version)
            if parent:
                parent.child_versions.append(version)

        LOGGER.info(f"Created version: {feature_version.get_full_version()}")
        return feature_version

    def get_version(
        self,
        feature_name: str,
        version: str,
    ) -> FeatureVersion | None:
        """Get specific feature version.

        Args:
            feature_name: Feature name
            version: Version string

        Returns:
            Feature version or None
        """
        if feature_name not in self.versions:
            return None

        for fv in self.versions[feature_name]:
            if fv.version == version:
                return fv

        return None

    def get_latest_version(
        self,
        feature_name: str,
        status: VersionStatus | None = None,
    ) -> FeatureVersion | None:
        """Get latest version of a feature.

        Args:
            feature_name: Feature name
            status: Filter by status (default: ACTIVE)

        Returns:
            Latest feature version or None
        """
        if feature_name not in self.versions:
            return None

        # Default to ACTIVE status
        if status is None:
            status = VersionStatus.ACTIVE

        # Filter by status and get latest
        matching_versions = [
            fv for fv in self.versions[feature_name] if fv.status == status
        ]

        if not matching_versions:
            return None

        # Sort by created_at and return latest
        return max(matching_versions, key=lambda v: v.created_at)

    def list_versions(
        self,
        feature_name: str,
        status: VersionStatus | None = None,
    ) -> list[FeatureVersion]:
        """List all versions of a feature.

        Args:
            feature_name: Feature name
            status: Filter by status

        Returns:
            List of feature versions
        """
        if feature_name not in self.versions:
            return []

        versions = self.versions[feature_name]

        # Filter by status if provided
        if status:
            versions = [v for v in versions if v.status == status]

        # Sort by version (reverse chronological)
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def activate_version(
        self,
        feature_name: str,
        version: str,
    ) -> bool:
        """Activate a feature version.

        Args:
            feature_name: Feature name
            version: Version to activate

        Returns:
            True if activated, False if failed
        """
        feature_version = self.get_version(feature_name, version)
        if not feature_version:
            return False

        if not feature_version.can_transition_to(VersionStatus.ACTIVE):
            LOGGER.warning(
                f"Cannot activate {feature_version.get_full_version()} "
                f"from status {feature_version.status}"
            )
            return False

        feature_version.status = VersionStatus.ACTIVE
        feature_version.activated_at = datetime.now()

        LOGGER.info(f"Activated version: {feature_version.get_full_version()}")
        return True

    def deprecate_version(
        self,
        feature_name: str,
        version: str,
    ) -> bool:
        """Deprecate a feature version.

        Args:
            feature_name: Feature name
            version: Version to deprecate

        Returns:
            True if deprecated, False if failed
        """
        feature_version = self.get_version(feature_name, version)
        if not feature_version:
            return False

        if not feature_version.can_transition_to(VersionStatus.DEPRECATED):
            LOGGER.warning(
                f"Cannot deprecate {feature_version.get_full_version()} "
                f"from status {feature_version.status}"
            )
            return False

        feature_version.status = VersionStatus.DEPRECATED
        feature_version.deprecated_at = datetime.now()

        LOGGER.info(f"Deprecated version: {feature_version.get_full_version()}")
        return True

    def archive_version(
        self,
        feature_name: str,
        version: str,
    ) -> bool:
        """Archive a feature version.

        Args:
            feature_name: Feature name
            version: Version to archive

        Returns:
            True if archived, False if failed
        """
        feature_version = self.get_version(feature_name, version)
        if not feature_version:
            return False

        if not feature_version.can_transition_to(VersionStatus.ARCHIVED):
            LOGGER.warning(
                f"Cannot archive {feature_version.get_full_version()} "
                f"from status {feature_version.status}"
            )
            return False

        feature_version.status = VersionStatus.ARCHIVED
        feature_version.archived_at = datetime.now()

        LOGGER.info(f"Archived version: {feature_version.get_full_version()}")
        return True

    def get_version_lineage(
        self,
        feature_name: str,
        version: str,
    ) -> dict[str, Any]:
        """Get version lineage (ancestors and descendants).

        Args:
            feature_name: Feature name
            version: Version string

        Returns:
            Lineage information
        """
        feature_version = self.get_version(feature_name, version)
        if not feature_version:
            return {}

        # Get ancestors
        ancestors = []
        current = feature_version
        while current.parent_version:
            parent = self.get_version(feature_name, current.parent_version)
            if not parent:
                break
            ancestors.append(parent.version)
            current = parent

        # Get descendants (breadth-first)
        descendants = []
        queue = list(feature_version.child_versions)
        while queue:
            child_version = queue.pop(0)
            descendants.append(child_version)
            child = self.get_version(feature_name, child_version)
            if child:
                queue.extend(child.child_versions)

        return {
            "version": version,
            "ancestors": ancestors,
            "descendants": descendants,
            "depth": len(ancestors),
        }

    def compare_versions(
        self,
        feature_name: str,
        version1: str,
        version2: str,
    ) -> dict[str, Any]:
        """Compare two versions.

        Args:
            feature_name: Feature name
            version1: First version
            version2: Second version

        Returns:
            Comparison results
        """
        v1 = self.get_version(feature_name, version1)
        v2 = self.get_version(feature_name, version2)

        if not v1 or not v2:
            return {}

        return {
            "version1": version1,
            "version2": version2,
            "status_changed": v1.status != v2.status,
            "breaking_changes": v1.breaking_changes or v2.breaking_changes,
            "backward_compatible": v1.backward_compatible and v2.backward_compatible,
            "time_diff_days": (v2.created_at - v1.created_at).days,
        }
