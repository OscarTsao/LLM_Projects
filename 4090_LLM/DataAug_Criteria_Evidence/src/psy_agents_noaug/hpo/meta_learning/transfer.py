#!/usr/bin/env python
"""Transfer learning for HPO across tasks (Phase 10).

This module provides utilities for transferring knowledge between
related HPO tasks (e.g., criteria → evidence, share → joint).

Key Features:
- Parameter mapping between different search spaces
- Selective parameter transfer (shared vs task-specific)
- Confidence-based transfer (only transfer reliable knowledge)
- Cross-architecture knowledge transfer
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from psy_agents_noaug.hpo.meta_learning.history import TrialHistoryAnalyzer
from psy_agents_noaug.hpo.meta_learning.warm_start import WarmStartStrategy

LOGGER = logging.getLogger(__name__)


class TransferLearner:
    """Transfer learning for HPO across related tasks."""

    # Common parameters across all architectures
    SHARED_PARAMS = {
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "batch_size",
        "max_grad_norm",
        "dropout",
        "epochs",
    }

    # Architecture-specific parameter mappings
    ARCHITECTURE_MAPPINGS = {
        # From criteria to other architectures
        "criteria->evidence": {
            "shared": SHARED_PARAMS,
            "rename": {},  # No renames needed
            "exclude": set(),  # No exclusions
        },
        "criteria->share": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        "criteria->joint": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        # From evidence to other architectures
        "evidence->criteria": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        "evidence->share": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        "evidence->joint": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        # From share to joint
        "share->joint": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
        # From joint to share
        "joint->share": {
            "shared": SHARED_PARAMS,
            "rename": {},
            "exclude": set(),
        },
    }

    def __init__(
        self,
        storage: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        """Initialize transfer learner.

        Args:
            storage: Optuna storage URL
            confidence_threshold: Minimum confidence to transfer parameters
                                (based on importance scores)
        """
        self.storage = storage
        self.analyzer = TrialHistoryAnalyzer(storage=storage)
        self.warm_starter = WarmStartStrategy(storage=storage, analyzer=self.analyzer)
        self.confidence_threshold = confidence_threshold

    def transfer_from_task(
        self,
        target_study: optuna.Study,
        source_task: str,
        target_task: str,
        source_study: optuna.Study | str,
        n_configs: int = 5,
        transfer_mode: str = "shared_only",
    ) -> int:
        """Transfer knowledge from source task to target task.

        Args:
            target_study: Target study to warm-start
            source_task: Source task name (e.g., "criteria")
            target_task: Target task name (e.g., "evidence")
            source_study: Source study object or name
            n_configs: Number of configurations to transfer
            transfer_mode: Transfer mode:
                - "shared_only": Only transfer shared parameters
                - "confident": Transfer parameters above confidence threshold
                - "all": Transfer all compatible parameters

        Returns:
            Number of configurations transferred
        """
        # Load source study if needed
        if isinstance(source_study, str):
            source_study = self.analyzer.load_study(source_study)

        # Get mapping for this task pair
        mapping_key = f"{source_task}->{target_task}"
        if mapping_key not in self.ARCHITECTURE_MAPPINGS:
            LOGGER.warning(
                "No mapping defined for %s, using shared params only", mapping_key
            )
            mapping = {
                "shared": self.SHARED_PARAMS,
                "rename": {},
                "exclude": set(),
            }
        else:
            mapping = self.ARCHITECTURE_MAPPINGS[mapping_key]

        LOGGER.info(
            "Transferring from %s to %s (mode=%s, n_configs=%d)",
            source_task,
            target_task,
            transfer_mode,
            n_configs,
        )

        # Analyze source study to get parameter importance
        analysis = self.analyzer.analyze_study(source_study, compute_importance=True)

        # Get top K configs from source
        source_configs = self.analyzer.get_top_k_configs(source_study, k=n_configs)

        # Filter and map parameters
        transferred_configs = []
        for config in source_configs:
            transferred_config = self._map_config(
                config=config,
                mapping=mapping,
                param_importance=analysis.param_importance,
                transfer_mode=transfer_mode,
            )
            if transferred_config:
                transferred_configs.append(transferred_config)

        # Enqueue transferred configs
        enqueued_count = 0
        for config in transferred_configs:
            try:
                target_study.enqueue_trial(config)
                enqueued_count += 1
                LOGGER.debug("Transferred config: %s", config)
            except Exception as e:
                LOGGER.warning("Failed to enqueue transferred config: %s", e)

        LOGGER.info(
            "Successfully transferred %d/%d configurations",
            enqueued_count,
            len(transferred_configs),
        )
        return enqueued_count

    def _map_config(
        self,
        config: dict[str, Any],
        mapping: dict[str, Any],
        param_importance: dict[str, float],
        transfer_mode: str,
    ) -> dict[str, Any]:
        """Map a configuration using transfer mapping rules.

        Args:
            config: Source configuration
            mapping: Transfer mapping rules
            param_importance: Parameter importance scores
            transfer_mode: Transfer mode

        Returns:
            Mapped configuration for target task
        """
        mapped_config = {}

        for param_name, param_value in config.items():
            # Skip excluded parameters
            if param_name in mapping.get("exclude", set()):
                continue

            # Rename if needed
            target_param_name = mapping.get("rename", {}).get(param_name, param_name)

            # Check if parameter should be transferred
            should_transfer = False

            if transfer_mode == "shared_only":
                should_transfer = param_name in mapping["shared"]
            elif transfer_mode == "confident":
                # Transfer if shared OR has high importance
                importance = param_importance.get(param_name, 0.0)
                should_transfer = (
                    param_name in mapping["shared"]
                    or importance >= self.confidence_threshold
                )
            elif transfer_mode == "all":
                should_transfer = True

            if should_transfer:
                mapped_config[target_param_name] = param_value

        return mapped_config

    def create_transfer_study(
        self,
        study_name: str,
        task_name: str,
        source_tasks: list[tuple[str, str | optuna.Study]],
        direction: str = "minimize",
        n_configs_per_source: int = 3,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> optuna.Study:
        """Create a new study with transferred knowledge from related tasks.

        Args:
            study_name: Name for new study
            task_name: Target task name (e.g., "evidence")
            source_tasks: List of (source_task_name, source_study) pairs
                         e.g., [("criteria", "criteria-hpo"), ("share", share_study)]
            direction: Optimization direction
            n_configs_per_source: Configs to transfer from each source
            sampler: Sampler for new study
            pruner: Pruner for new study

        Returns:
            New study with transferred configurations
        """
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=False,
        )

        LOGGER.info(
            "Created transfer study '%s' for task '%s' with %d sources",
            study_name,
            task_name,
            len(source_tasks),
        )

        # Transfer from each source
        total_transferred = 0
        for source_task, source_study in source_tasks:
            n_transferred = self.transfer_from_task(
                target_study=study,
                source_task=source_task,
                target_task=task_name,
                source_study=source_study,
                n_configs=n_configs_per_source,
                transfer_mode="confident",
            )
            total_transferred += n_transferred

        LOGGER.info("Total configurations transferred: %d", total_transferred)
        return study


class CrossArchitectureTransfer:
    """Transfer knowledge across different model architectures."""

    def __init__(self, storage: str | None = None):
        """Initialize cross-architecture transfer.

        Args:
            storage: Optuna storage URL
        """
        self.storage = storage
        self.transfer_learner = TransferLearner(storage=storage)

    def transfer_hyperparameters(
        self,
        target_study: optuna.Study,
        source_studies: list[tuple[str, str | optuna.Study]],
        n_configs_per_source: int = 3,
    ) -> int:
        """Transfer hyperparameters from multiple architectures.

        This is useful when starting HPO for a new architecture (e.g., joint)
        and you already have results from simpler architectures (criteria, evidence).

        Args:
            target_study: Target study for new architecture
            source_studies: List of (architecture_name, study) pairs
            n_configs_per_source: Configs to transfer from each source

        Returns:
            Total number of configurations transferred

        Example:
            >>> transfer = CrossArchitectureTransfer(storage="sqlite:///optuna.db")
            >>> joint_study = optuna.create_study(...)
            >>> # Transfer from criteria and evidence to joint
            >>> transfer.transfer_hyperparameters(
            ...     joint_study,
            ...     source_studies=[
            ...         ("criteria", "criteria-maximal-hpo"),
            ...         ("evidence", "evidence-maximal-hpo"),
            ...     ],
            ...     n_configs_per_source=5,
            ... )
        """
        total_transferred = 0

        for source_arch, source_study in source_studies:
            # Infer target architecture from study name
            # (Assume study name contains architecture name)
            target_arch = self._infer_architecture(target_study.study_name)

            if not target_arch:
                LOGGER.warning(
                    "Could not infer target architecture from study name: %s",
                    target_study.study_name,
                )
                continue

            # Transfer using TransferLearner
            n_transferred = self.transfer_learner.transfer_from_task(
                target_study=target_study,
                source_task=source_arch,
                target_task=target_arch,
                source_study=source_study,
                n_configs=n_configs_per_source,
                transfer_mode="confident",
            )
            total_transferred += n_transferred

        return total_transferred

    def _infer_architecture(self, study_name: str) -> str | None:
        """Infer architecture name from study name.

        Args:
            study_name: Study name (e.g., "evidence-maximal-hpo")

        Returns:
            Architecture name (e.g., "evidence") or None
        """
        architectures = ["criteria", "evidence", "share", "joint"]
        study_lower = study_name.lower()

        for arch in architectures:
            if arch in study_lower:
                return arch

        return None


def recommend_transfer_sources(
    target_task: str,
    available_studies: dict[str, optuna.Study | str],
) -> list[tuple[str, str | optuna.Study]]:
    """Recommend which studies to use as transfer sources.

    Args:
        target_task: Target task name (e.g., "joint")
        available_studies: Dict of {task_name: study} for completed studies

    Returns:
        Recommended (task_name, study) pairs for transfer

    Example:
        >>> available = {
        ...     "criteria": "criteria-maximal-hpo",
        ...     "evidence": "evidence-maximal-hpo",
        ...     "share": share_study,
        ... }
        >>> sources = recommend_transfer_sources("joint", available)
        >>> # Returns [("share", share_study), ("criteria", "criteria-maximal-hpo"), ...]
    """
    # Define transfer affinity (which tasks are most relevant)
    TRANSFER_AFFINITY = {
        "criteria": ["criteria", "share", "evidence"],
        "evidence": ["evidence", "share", "criteria"],
        "share": ["share", "criteria", "evidence"],
        "joint": ["share", "joint", "criteria", "evidence"],
    }

    if target_task not in TRANSFER_AFFINITY:
        LOGGER.warning("Unknown target task: %s", target_task)
        return []

    # Get affinity ranking
    affinity_order = TRANSFER_AFFINITY[target_task]

    # Build recommended sources in priority order
    recommended = []
    for source_task in affinity_order:
        if source_task in available_studies:
            recommended.append((source_task, available_studies[source_task]))

    LOGGER.info(
        "Recommended %d transfer sources for %s: %s",
        len(recommended),
        target_task,
        [task for task, _ in recommended],
    )

    return recommended
