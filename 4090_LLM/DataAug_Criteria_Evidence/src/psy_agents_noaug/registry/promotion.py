#!/usr/bin/env python
"""Model promotion workflows (Phase 20).

This module provides:
- Stage-based promotion (dev → staging → production)
- Promotion criteria validation
- Rollback capabilities
- Promotion history tracking
- Approval workflows
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


class Stage(str, Enum):
    """Model deployment stages."""

    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class PromotionCriteria:
    """Criteria for model promotion."""

    # Metric thresholds
    min_accuracy: float | None = None
    min_f1: float | None = None
    min_precision: float | None = None
    min_recall: float | None = None
    max_loss: float | None = None

    # Performance requirements
    max_latency_ms: float | None = None
    max_memory_mb: float | None = None

    # Quality requirements
    min_test_samples: int | None = None
    required_tags: list[str] = field(default_factory=list)

    # Approval requirements
    require_manual_approval: bool = False
    approver: str | None = None

    def validate(self, metadata: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate promotion criteria.

        Args:
            metadata: Model metadata to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        metrics = metadata.get("validation_metrics", {})

        # Check metric thresholds
        if self.min_accuracy and metrics.get("accuracy", 0) < self.min_accuracy:
            errors.append(
                f"Accuracy {metrics.get('accuracy', 0):.3f} < {self.min_accuracy}"
            )

        if self.min_f1 and metrics.get("f1", 0) < self.min_f1:
            errors.append(f"F1 {metrics.get('f1', 0):.3f} < {self.min_f1}")

        if self.min_precision and metrics.get("precision", 0) < self.min_precision:
            errors.append(
                f"Precision {metrics.get('precision', 0):.3f} < {self.min_precision}"
            )

        if self.min_recall and metrics.get("recall", 0) < self.min_recall:
            errors.append(f"Recall {metrics.get('recall', 0):.3f} < {self.min_recall}")

        if self.max_loss and metrics.get("loss", float("inf")) > self.max_loss:
            errors.append(
                f"Loss {metrics.get('loss', float('inf')):.3f} > {self.max_loss}"
            )

        # Check required tags
        model_tags = set(metadata.get("tags", []))
        required = set(self.required_tags)
        if not required.issubset(model_tags):
            missing = required - model_tags
            errors.append(f"Missing required tags: {missing}")

        # Check test samples
        if self.min_test_samples:
            test_samples = metadata.get("test_samples", 0)
            if test_samples < self.min_test_samples:
                errors.append(f"Test samples {test_samples} < {self.min_test_samples}")

        return len(errors) == 0, errors


@dataclass
class PromotionRecord:
    """Record of a model promotion."""

    model_name: str
    version: str
    from_stage: Stage
    to_stage: Stage
    promoted_at: datetime
    promoted_by: str
    criteria_met: bool
    approval_status: str = "pending"  # pending, approved, rejected
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "promoted_at": self.promoted_at.isoformat(),
            "promoted_by": self.promoted_by,
            "criteria_met": self.criteria_met,
            "approval_status": self.approval_status,
            "notes": self.notes,
        }


class PromotionWorkflow:
    """Workflow for model promotion."""

    def __init__(self):
        """Initialize promotion workflow."""
        self.promotion_history: list[PromotionRecord] = []
        self.criteria_registry: dict[tuple[Stage, Stage], PromotionCriteria] = {}

        # Set default criteria
        self._set_default_criteria()

        LOGGER.info("Initialized PromotionWorkflow")

    def _set_default_criteria(self) -> None:
        """Set default promotion criteria."""
        # Dev → Staging: Basic quality checks
        self.criteria_registry[(Stage.DEV, Stage.STAGING)] = PromotionCriteria(
            min_accuracy=0.7,
            min_test_samples=100,
        )

        # Staging → Production: Strict requirements
        self.criteria_registry[(Stage.STAGING, Stage.PRODUCTION)] = PromotionCriteria(
            min_accuracy=0.85,
            min_f1=0.8,
            min_test_samples=500,
            require_manual_approval=True,
        )

    def set_promotion_criteria(
        self,
        from_stage: Stage,
        to_stage: Stage,
        criteria: PromotionCriteria,
    ) -> None:
        """Set promotion criteria for a stage transition.

        Args:
            from_stage: Source stage
            to_stage: Target stage
            criteria: Promotion criteria
        """
        self.criteria_registry[(from_stage, to_stage)] = criteria
        LOGGER.info(f"Set promotion criteria: {from_stage.value} → {to_stage.value}")

    def can_promote(
        self,
        from_stage: Stage,
        to_stage: Stage,
        metadata: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Check if model can be promoted.

        Args:
            from_stage: Source stage
            to_stage: Target stage
            metadata: Model metadata

        Returns:
            Tuple of (can_promote, error_messages)
        """
        # Get criteria
        criteria = self.criteria_registry.get((from_stage, to_stage))

        if not criteria:
            return True, []  # No criteria defined, allow promotion

        # Validate criteria
        return criteria.validate(metadata)

    def promote(
        self,
        model_name: str,
        version: str,
        from_stage: Stage,
        to_stage: Stage,
        metadata: dict[str, Any],
        promoted_by: str = "system",
        notes: str = "",
        skip_validation: bool = False,
    ) -> tuple[bool, PromotionRecord]:
        """Promote a model to a new stage.

        Args:
            model_name: Model name
            version: Model version
            from_stage: Source stage
            to_stage: Target stage
            metadata: Model metadata
            promoted_by: Who promoted the model
            notes: Promotion notes
            skip_validation: Skip validation checks

        Returns:
            Tuple of (success, promotion_record)
        """
        # Check promotion criteria
        criteria_met = True
        if not skip_validation:
            can_promote, errors = self.can_promote(from_stage, to_stage, metadata)
            criteria_met = can_promote

            if not can_promote:
                LOGGER.warning(
                    f"Promotion blocked for {model_name}:{version}: {errors}"
                )

        # Create promotion record
        record = PromotionRecord(
            model_name=model_name,
            version=version,
            from_stage=from_stage,
            to_stage=to_stage,
            promoted_at=datetime.now(),
            promoted_by=promoted_by,
            criteria_met=criteria_met,
            notes=notes,
        )

        # Check if approval required
        criteria = self.criteria_registry.get((from_stage, to_stage))
        if criteria and criteria.require_manual_approval:
            record.approval_status = "pending"
            LOGGER.info(
                f"Promotion requires approval: {model_name}:{version} "
                f"{from_stage.value} → {to_stage.value}"
            )
        else:
            record.approval_status = "approved"

        # Add to history
        self.promotion_history.append(record)

        success = criteria_met and record.approval_status == "approved"

        if success:
            LOGGER.info(
                f"Promoted {model_name}:{version} "
                f"{from_stage.value} → {to_stage.value}"
            )
        else:
            LOGGER.warning(
                f"Promotion pending for {model_name}:{version} "
                f"(criteria_met={criteria_met}, approval={record.approval_status})"
            )

        return success, record

    def approve_promotion(
        self,
        model_name: str,
        version: str,
        approver: str,
    ) -> bool:
        """Approve a pending promotion.

        Args:
            model_name: Model name
            version: Model version
            approver: Who approved

        Returns:
            True if successful
        """
        # Find pending promotion
        for record in reversed(self.promotion_history):
            if (
                record.model_name == model_name
                and record.version == version
                and record.approval_status == "pending"
            ):
                record.approval_status = "approved"
                record.notes += f"\nApproved by {approver}"
                LOGGER.info(
                    f"Approved promotion: {model_name}:{version} "
                    f"{record.from_stage.value} → {record.to_stage.value}"
                )
                return True

        return False

    def reject_promotion(
        self,
        model_name: str,
        version: str,
        rejector: str,
        reason: str = "",
    ) -> bool:
        """Reject a pending promotion.

        Args:
            model_name: Model name
            version: Model version
            rejector: Who rejected
            reason: Rejection reason

        Returns:
            True if successful
        """
        # Find pending promotion
        for record in reversed(self.promotion_history):
            if (
                record.model_name == model_name
                and record.version == version
                and record.approval_status == "pending"
            ):
                record.approval_status = "rejected"
                record.notes += f"\nRejected by {rejector}: {reason}"
                LOGGER.info(f"Rejected promotion: {model_name}:{version} - {reason}")
                return True

        return False

    def get_promotion_history(
        self,
        model_name: str | None = None,
        version: str | None = None,
    ) -> list[PromotionRecord]:
        """Get promotion history.

        Args:
            model_name: Filter by model name (optional)
            version: Filter by version (optional)

        Returns:
            List of promotion records
        """
        history = self.promotion_history

        if model_name:
            history = [r for r in history if r.model_name == model_name]

        if version:
            history = [r for r in history if r.version == version]

        return sorted(history, key=lambda r: r.promoted_at, reverse=True)


class ModelPromoter:
    """High-level model promotion manager."""

    def __init__(self):
        """Initialize model promoter."""
        self.workflow = PromotionWorkflow()
        self.hooks: dict[tuple[Stage, Stage], list[Callable]] = {}
        LOGGER.info("Initialized ModelPromoter")

    def register_hook(
        self,
        from_stage: Stage,
        to_stage: Stage,
        hook: Callable[[str, str], None],
    ) -> None:
        """Register a promotion hook.

        Args:
            from_stage: Source stage
            to_stage: Target stage
            hook: Callback function(model_name, version)
        """
        key = (from_stage, to_stage)
        if key not in self.hooks:
            self.hooks[key] = []
        self.hooks[key].append(hook)
        LOGGER.info(f"Registered promotion hook: {from_stage.value} → {to_stage.value}")

    def promote_model(
        self,
        model_name: str,
        version: str,
        to_stage: Stage,
        metadata: dict[str, Any],
        current_stage: Stage = Stage.DEV,
        promoted_by: str = "system",
    ) -> bool:
        """Promote model to a new stage.

        Args:
            model_name: Model name
            version: Model version
            to_stage: Target stage
            metadata: Model metadata
            current_stage: Current stage
            promoted_by: Who promoted

        Returns:
            True if successful
        """
        # Promote
        success, record = self.workflow.promote(
            model_name=model_name,
            version=version,
            from_stage=current_stage,
            to_stage=to_stage,
            metadata=metadata,
            promoted_by=promoted_by,
        )

        if success:
            # Execute hooks
            key = (current_stage, to_stage)
            if key in self.hooks:
                for hook in self.hooks[key]:
                    try:
                        hook(model_name, version)
                    except Exception:
                        LOGGER.exception(f"Hook failed: {hook}")

        return success


# Convenience function
def promote_model(
    model_name: str,
    version: str,
    to_stage: str,
    metadata: dict[str, Any],
) -> bool:
    """Promote a model (convenience function).

    Args:
        model_name: Model name
        version: Model version
        to_stage: Target stage (dev/staging/production)
        metadata: Model metadata

    Returns:
        True if successful
    """
    promoter = ModelPromoter()
    return promoter.promote_model(
        model_name=model_name,
        version=version,
        to_stage=Stage(to_stage),
        metadata=metadata,
    )
