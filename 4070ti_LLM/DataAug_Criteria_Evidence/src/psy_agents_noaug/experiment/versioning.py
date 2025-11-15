#!/usr/bin/env python
"""Configuration versioning utilities (Phase 15).

This module provides configuration version control including:
- Configuration hashing and comparison
- Version history tracking
- Configuration diff generation
- Rollback support
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfigVersion:
    """A versioned configuration."""

    version_id: str
    config: dict[str, Any]
    config_hash: str
    timestamp: datetime
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    parent_version: str | None = None


class ConfigVersioner:
    """Version control for experiment configurations."""

    def __init__(
        self,
        storage_dir: Path | str = "config_versions",
    ):
        """Initialize config versioner.

        Args:
            storage_dir: Directory to store config versions
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.versions: dict[str, ConfigVersion] = {}
        self._load_versions()

        LOGGER.info("Initialized ConfigVersioner (storage=%s)", self.storage_dir)

    def _compute_hash(self, config: dict[str, Any]) -> str:
        """Compute config hash.

        Args:
            config: Configuration

        Returns:
            Hash string
        """
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _generate_version_id(self) -> str:
        """Generate unique version ID.

        Returns:
            Version ID (timestamp-based)
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _load_versions(self) -> None:
        """Load existing versions from storage."""
        if not self.storage_dir.exists():
            return

        for version_file in self.storage_dir.glob("*.json"):
            try:
                with version_file.open() as f:
                    data = json.load(f)

                version = ConfigVersion(
                    version_id=data["version_id"],
                    config=data["config"],
                    config_hash=data["config_hash"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    description=data.get("description", ""),
                    tags=data.get("tags", {}),
                    parent_version=data.get("parent_version"),
                )

                self.versions[version.version_id] = version

            except Exception as e:
                LOGGER.warning("Failed to load version %s: %s", version_file, e)

        LOGGER.info("Loaded %d config versions", len(self.versions))

    def save_version(
        self,
        config: dict[str, Any],
        description: str = "",
        tags: dict[str, str] | None = None,
        parent_version: str | None = None,
    ) -> ConfigVersion:
        """Save a new config version.

        Args:
            config: Configuration to version
            description: Version description
            tags: Tags for the version
            parent_version: Parent version ID

        Returns:
            Created version
        """
        version_id = self._generate_version_id()
        config_hash = self._compute_hash(config)

        # Check if config already exists
        for existing_version in self.versions.values():
            if existing_version.config_hash == config_hash:
                LOGGER.warning(
                    "Config already exists as version: %s",
                    existing_version.version_id,
                )
                return existing_version

        # Create new version
        version = ConfigVersion(
            version_id=version_id,
            config=config,
            config_hash=config_hash,
            timestamp=datetime.now(),
            description=description,
            tags=tags or {},
            parent_version=parent_version,
        )

        # Save to storage
        version_file = self.storage_dir / f"{version_id}.json"
        with version_file.open("w") as f:
            json.dump(
                {
                    "version_id": version.version_id,
                    "config": version.config,
                    "config_hash": version.config_hash,
                    "timestamp": version.timestamp.isoformat(),
                    "description": version.description,
                    "tags": version.tags,
                    "parent_version": version.parent_version,
                },
                f,
                indent=2,
            )

        self.versions[version_id] = version

        LOGGER.info(
            "Saved config version: %s (hash=%s)",
            version_id,
            config_hash,
        )

        return version

    def get_version(self, version_id: str) -> ConfigVersion:
        """Get a specific version.

        Args:
            version_id: Version ID

        Returns:
            Config version

        Raises:
            ValueError: If version not found
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        return self.versions[version_id]

    def get_latest_version(self) -> ConfigVersion | None:
        """Get the latest version.

        Returns:
            Latest version or None
        """
        if not self.versions:
            return None

        return max(self.versions.values(), key=lambda v: v.timestamp)

    def list_versions(
        self,
        tags: dict[str, str] | None = None,
    ) -> list[ConfigVersion]:
        """List all versions, optionally filtered by tags.

        Args:
            tags: Filter by tags

        Returns:
            List of versions
        """
        versions = list(self.versions.values())

        if tags:
            versions = [
                v
                for v in versions
                if all(v.tags.get(k) == val for k, val in tags.items())
            ]

        return sorted(versions, key=lambda v: v.timestamp, reverse=True)

    def diff_configs(
        self,
        config1: dict[str, Any],
        config2: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute difference between two configs.

        Args:
            config1: First config
            config2: Second config

        Returns:
            Diff information
        """
        diff = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {},
        }

        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            if key not in config1:
                diff["added"][key] = config2[key]
            elif key not in config2:
                diff["removed"][key] = config1[key]
            elif config1[key] != config2[key]:
                diff["modified"][key] = {
                    "old": config1[key],
                    "new": config2[key],
                }
            else:
                diff["unchanged"][key] = config1[key]

        return diff

    def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
    ) -> dict[str, Any]:
        """Compare two config versions.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Comparison results
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)

        diff = self.diff_configs(version1.config, version2.config)

        return {
            "version1": {
                "id": version1.version_id,
                "hash": version1.config_hash,
                "timestamp": version1.timestamp.isoformat(),
            },
            "version2": {
                "id": version2.version_id,
                "hash": version2.config_hash,
                "timestamp": version2.timestamp.isoformat(),
            },
            "diff": diff,
            "n_added": len(diff["added"]),
            "n_removed": len(diff["removed"]),
            "n_modified": len(diff["modified"]),
            "n_unchanged": len(diff["unchanged"]),
        }

    def restore_version(
        self,
        version_id: str,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Restore a config version.

        Args:
            version_id: Version ID to restore
            output_path: Optional path to save restored config

        Returns:
            Restored config
        """
        version = self.get_version(version_id)

        if output_path:
            output_path = Path(output_path)
            with output_path.open("w") as f:
                json.dump(version.config, f, indent=2)

            LOGGER.info("Restored config to: %s", output_path)

        return version.config

    def get_version_history(
        self,
        version_id: str,
    ) -> list[ConfigVersion]:
        """Get version history by following parent links.

        Args:
            version_id: Starting version ID

        Returns:
            Version history (newest to oldest)
        """
        history = []
        current_id = version_id

        while current_id:
            version = self.get_version(current_id)
            history.append(version)
            current_id = version.parent_version

        return history


def version_config(
    config: dict[str, Any],
    storage_dir: Path | str = "config_versions",
    description: str = "",
    tags: dict[str, str] | None = None,
) -> ConfigVersion:
    """Version a config (convenience function).

    Args:
        config: Configuration
        storage_dir: Storage directory
        description: Version description
        tags: Version tags

    Returns:
        Created version
    """
    versioner = ConfigVersioner(storage_dir=storage_dir)
    return versioner.save_version(
        config=config,
        description=description,
        tags=tags,
    )
