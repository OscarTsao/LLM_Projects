#!/usr/bin/env python
"""Quality gate validation for CI/CD (Phase 16).

This module provides quality gate enforcement including:
- Metric thresholds
- Test coverage requirements
- Performance benchmarks
- Code quality checks
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


class GateType(Enum):
    """Type of quality gate."""

    METRIC = "metric"
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class GateStatus(Enum):
    """Status of a quality gate."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    gate_name: str
    status: GateStatus
    actual_value: Any
    threshold_value: Any
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityGate:
    """A quality gate definition."""

    name: str
    gate_type: GateType
    threshold: Any
    comparison: str = ">="  # >=, <=, ==, >, <
    value_getter: Callable[[], Any] | None = None
    error_on_fail: bool = True
    description: str = ""


class QualityGateValidator:
    """Validate quality gates."""

    def __init__(self):
        """Initialize validator."""
        self.gates: list[QualityGate] = []
        LOGGER.info("Initialized QualityGateValidator")

    def add_gate(self, gate: QualityGate) -> None:
        """Add a quality gate.

        Args:
            gate: Quality gate
        """
        self.gates.append(gate)
        LOGGER.debug("Added quality gate: %s", gate.name)

    def _compare_values(
        self,
        actual: Any,
        threshold: Any,
        comparison: str,
    ) -> bool:
        """Compare values based on comparison operator.

        Args:
            actual: Actual value
            threshold: Threshold value
            comparison: Comparison operator

        Returns:
            True if comparison passes
        """
        if comparison == ">=":
            return actual >= threshold
        if comparison == "<=":
            return actual <= threshold
        if comparison == "==":
            return actual == threshold
        if comparison == ">":
            return actual > threshold
        if comparison == "<":
            return actual < threshold
        raise ValueError(f"Unknown comparison operator: {comparison}")

    def validate_gate(self, gate: QualityGate) -> QualityGateResult:
        """Validate a single gate.

        Args:
            gate: Quality gate

        Returns:
            Gate result
        """
        LOGGER.info("Validating gate: %s", gate.name)

        # Get actual value
        if gate.value_getter is None:
            return QualityGateResult(
                gate_name=gate.name,
                status=GateStatus.SKIPPED,
                actual_value=None,
                threshold_value=gate.threshold,
                message="No value getter provided",
            )

        try:
            actual_value = gate.value_getter()
        except Exception as e:
            LOGGER.exception("Failed to get value for gate %s: %s", gate.name, e)
            return QualityGateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                actual_value=None,
                threshold_value=gate.threshold,
                message=f"Error getting value: {e}",
            )

        # Compare with threshold
        try:
            passes = self._compare_values(
                actual_value,
                gate.threshold,
                gate.comparison,
            )
        except Exception as e:
            LOGGER.exception("Failed to compare values for gate %s: %s", gate.name, e)
            return QualityGateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                actual_value=actual_value,
                threshold_value=gate.threshold,
                message=f"Error comparing values: {e}",
            )

        # Create result
        if passes:
            status = GateStatus.PASSED
            message = (
                f"✓ {gate.name}: {actual_value} {gate.comparison} {gate.threshold}"
            )
        else:
            status = GateStatus.FAILED if gate.error_on_fail else GateStatus.WARNING
            message = (
                f"✗ {gate.name}: {actual_value} not {gate.comparison} {gate.threshold}"
            )

        result = QualityGateResult(
            gate_name=gate.name,
            status=status,
            actual_value=actual_value,
            threshold_value=gate.threshold,
            message=message,
        )

        LOGGER.info("%s: %s", gate.name, status.value)
        return result

    def validate_all(self) -> dict[str, Any]:
        """Validate all gates.

        Returns:
            Validation results
        """
        LOGGER.info("Validating %d quality gates", len(self.gates))

        results = []
        failed_count = 0
        warning_count = 0

        for gate in self.gates:
            result = self.validate_gate(gate)
            results.append(result)

            if result.status == GateStatus.FAILED:
                failed_count += 1
            elif result.status == GateStatus.WARNING:
                warning_count += 1

        overall_status = "passed" if failed_count == 0 else "failed"

        summary = {
            "status": overall_status,
            "total_gates": len(self.gates),
            "passed": len([r for r in results if r.status == GateStatus.PASSED]),
            "failed": failed_count,
            "warnings": warning_count,
            "skipped": len([r for r in results if r.status == GateStatus.SKIPPED]),
            "results": [
                {
                    "gate_name": r.gate_name,
                    "status": r.status.value,
                    "actual_value": r.actual_value,
                    "threshold_value": r.threshold_value,
                    "message": r.message,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in results
            ],
        }

        LOGGER.info(
            "Quality gates validation complete: %s (passed=%d, failed=%d, warnings=%d)",
            overall_status,
            summary["passed"],
            failed_count,
            warning_count,
        )

        return summary


def validate_quality_gates(
    gates: list[QualityGate],
) -> dict[str, Any]:
    """Validate quality gates (convenience function).

    Args:
        gates: List of quality gates

    Returns:
        Validation results
    """
    validator = QualityGateValidator()

    for gate in gates:
        validator.add_gate(gate)

    return validator.validate_all()
