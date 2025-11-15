#!/usr/bin/env python
"""Experiment tracking and reproducibility (Phase 15).

This module provides comprehensive experiment management including:
- Experiment tracking and logging
- Configuration versioning
- Reproducibility guarantees
- Experiment comparison tools

Key Features:
- Automatic experiment metadata capture
- Git integration for code versioning
- Environment snapshot and restoration
- Comprehensive experiment comparison
"""

from __future__ import annotations

from psy_agents_noaug.experiment.comparison import (
    ExperimentComparator,
    compare_experiments,
)
from psy_agents_noaug.experiment.reproducibility import (
    ReproducibilityManager,
    ensure_reproducibility,
    validate_reproducibility,
)
from psy_agents_noaug.experiment.tracker import (
    ExperimentTracker,
    track_experiment,
)
from psy_agents_noaug.experiment.versioning import (
    ConfigVersioner,
    version_config,
)

__all__ = [
    # Tracker
    "ExperimentTracker",
    "track_experiment",
    # Versioning
    "ConfigVersioner",
    "version_config",
    # Reproducibility
    "ReproducibilityManager",
    "ensure_reproducibility",
    "validate_reproducibility",
    # Comparison
    "ExperimentComparator",
    "compare_experiments",
]
