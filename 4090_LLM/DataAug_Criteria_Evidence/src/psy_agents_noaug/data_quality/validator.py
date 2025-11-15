#!/usr/bin/env python
"""Data validation (Phase 23).

This module provides tools for validating data quality including schema,
types, ranges, and business rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


class ValidationRule(str, Enum):
    """Types of validation rules."""

    TYPE_CHECK = "type_check"
    RANGE_CHECK = "range_check"
    PATTERN_CHECK = "pattern_check"
    CUSTOM = "custom"
    NOT_NULL = "not_null"
    UNIQUE = "unique"


@dataclass
class ValidationResult:
    """Result of data validation."""

    feature_name: str
    rule_name: str
    is_valid: bool
    num_violations: int
    total_samples: int
    violation_rate: float
    violations: list[Any] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get validation summary.

        Returns:
            Summary dictionary
        """
        return {
            "feature": self.feature_name,
            "rule": self.rule_name,
            "valid": self.is_valid,
            "violations": self.num_violations,
            "total": self.total_samples,
            "rate": f"{self.violation_rate:.2%}",
        }


class DataValidator:
    """Validator for data quality."""

    def __init__(self, max_violations_to_store: int = 100):
        """Initialize data validator.

        Args:
            max_violations_to_store: Maximum number of violations to store
        """
        self.max_violations = max_violations_to_store
        LOGGER.info("Initialized DataValidator")

    def validate_type(
        self,
        data: np.ndarray,
        expected_type: type,
        feature_name: str = "feature",
    ) -> ValidationResult:
        """Validate data types.

        Args:
            data: Data array
            expected_type: Expected data type
            feature_name: Feature name

        Returns:
            Validation result
        """
        violations = []
        for i, value in enumerate(data):
            if not isinstance(value, expected_type):
                violations.append(
                    {"index": i, "value": value, "type": type(value).__name__}
                )
                if len(violations) >= self.max_violations:
                    break

        num_violations = len(violations)
        total_samples = len(data)
        violation_rate = num_violations / total_samples if total_samples > 0 else 0.0

        return ValidationResult(
            feature_name=feature_name,
            rule_name=ValidationRule.TYPE_CHECK.value,
            is_valid=num_violations == 0,
            num_violations=num_violations,
            total_samples=total_samples,
            violation_rate=violation_rate,
            violations=violations,
            metadata={"expected_type": expected_type.__name__},
        )

    def validate_range(
        self,
        data: np.ndarray,
        min_value: float | None = None,
        max_value: float | None = None,
        feature_name: str = "feature",
    ) -> ValidationResult:
        """Validate value ranges.

        Args:
            data: Data array
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            feature_name: Feature name

        Returns:
            Validation result
        """
        violations = []

        for i, value in enumerate(data):
            # Skip non-numeric values
            try:
                val = float(value)
            except (ValueError, TypeError):
                violations.append({"index": i, "value": value, "reason": "non_numeric"})
                if len(violations) >= self.max_violations:
                    break
                continue

            # Check range
            if min_value is not None and val < min_value:
                violations.append(
                    {"index": i, "value": val, "reason": f"< {min_value}"}
                )
            elif max_value is not None and val > max_value:
                violations.append(
                    {"index": i, "value": val, "reason": f"> {max_value}"}
                )

            if len(violations) >= self.max_violations:
                break

        num_violations = len(violations)
        total_samples = len(data)
        violation_rate = num_violations / total_samples if total_samples > 0 else 0.0

        return ValidationResult(
            feature_name=feature_name,
            rule_name=ValidationRule.RANGE_CHECK.value,
            is_valid=num_violations == 0,
            num_violations=num_violations,
            total_samples=total_samples,
            violation_rate=violation_rate,
            violations=violations,
            metadata={"min_value": min_value, "max_value": max_value},
        )

    def validate_not_null(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> ValidationResult:
        """Validate no null/missing values.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Validation result
        """
        violations = []

        for i, value in enumerate(data):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                violations.append({"index": i, "value": None})
                if len(violations) >= self.max_violations:
                    break

        num_violations = len(violations)
        total_samples = len(data)
        violation_rate = num_violations / total_samples if total_samples > 0 else 0.0

        return ValidationResult(
            feature_name=feature_name,
            rule_name=ValidationRule.NOT_NULL.value,
            is_valid=num_violations == 0,
            num_violations=num_violations,
            total_samples=total_samples,
            violation_rate=violation_rate,
            violations=violations,
        )

    def validate_unique(
        self,
        data: np.ndarray,
        feature_name: str = "feature",
    ) -> ValidationResult:
        """Validate uniqueness of values.

        Args:
            data: Data array
            feature_name: Feature name

        Returns:
            Validation result
        """
        seen = set()
        violations = []

        for i, value in enumerate(data):
            # Convert to hashable type
            try:
                hashable_value = (
                    value if isinstance(value, (str, int, float)) else str(value)
                )
            except Exception:
                hashable_value = str(value)

            if hashable_value in seen:
                violations.append({"index": i, "value": value})
                if len(violations) >= self.max_violations:
                    break
            else:
                seen.add(hashable_value)

        num_violations = len(violations)
        total_samples = len(data)
        violation_rate = num_violations / total_samples if total_samples > 0 else 0.0

        return ValidationResult(
            feature_name=feature_name,
            rule_name=ValidationRule.UNIQUE.value,
            is_valid=num_violations == 0,
            num_violations=num_violations,
            total_samples=total_samples,
            violation_rate=violation_rate,
            violations=violations,
        )

    def validate_custom(
        self,
        data: np.ndarray,
        check_func: Callable[[Any], bool],
        feature_name: str = "feature",
        rule_name: str = "custom",
    ) -> ValidationResult:
        """Validate using custom function.

        Args:
            data: Data array
            check_func: Function that returns True if valid
            feature_name: Feature name
            rule_name: Rule name

        Returns:
            Validation result
        """
        violations = []

        for i, value in enumerate(data):
            try:
                if not check_func(value):
                    violations.append({"index": i, "value": value})
                    if len(violations) >= self.max_violations:
                        break
            except Exception as e:
                violations.append({"index": i, "value": value, "error": str(e)})
                if len(violations) >= self.max_violations:
                    break

        num_violations = len(violations)
        total_samples = len(data)
        violation_rate = num_violations / total_samples if total_samples > 0 else 0.0

        return ValidationResult(
            feature_name=feature_name,
            rule_name=rule_name,
            is_valid=num_violations == 0,
            num_violations=num_violations,
            total_samples=total_samples,
            violation_rate=violation_rate,
            violations=violations,
        )

    def validate_all(
        self,
        data: dict[str, np.ndarray],
        rules: dict[str, list[dict[str, Any]]],
    ) -> list[ValidationResult]:
        """Validate multiple features with multiple rules.

        Args:
            data: Dictionary of feature_name -> data_array
            rules: Dictionary of feature_name -> list of rule specs

        Returns:
            List of validation results
        """
        results = []

        for feature_name, feature_data in data.items():
            if feature_name not in rules:
                continue

            for rule_spec in rules[feature_name]:
                rule_type = rule_spec.get("type")

                if rule_type == ValidationRule.TYPE_CHECK.value:
                    result = self.validate_type(
                        feature_data,
                        rule_spec["expected_type"],
                        feature_name,
                    )
                elif rule_type == ValidationRule.RANGE_CHECK.value:
                    result = self.validate_range(
                        feature_data,
                        rule_spec.get("min"),
                        rule_spec.get("max"),
                        feature_name,
                    )
                elif rule_type == ValidationRule.NOT_NULL.value:
                    result = self.validate_not_null(feature_data, feature_name)
                elif rule_type == ValidationRule.UNIQUE.value:
                    result = self.validate_unique(feature_data, feature_name)
                else:
                    continue

                results.append(result)

        return results


def validate_data(
    data: np.ndarray,
    rules: list[dict[str, Any]],
    feature_name: str = "feature",
) -> list[ValidationResult]:
    """Validate data (convenience function).

    Args:
        data: Data array
        rules: List of validation rules
        feature_name: Feature name

    Returns:
        List of validation results
    """
    validator = DataValidator()
    results = []

    for rule in rules:
        rule_type = rule.get("type")

        if rule_type == ValidationRule.TYPE_CHECK.value:
            result = validator.validate_type(data, rule["expected_type"], feature_name)
        elif rule_type == ValidationRule.RANGE_CHECK.value:
            result = validator.validate_range(
                data, rule.get("min"), rule.get("max"), feature_name
            )
        elif rule_type == ValidationRule.NOT_NULL.value:
            result = validator.validate_not_null(data, feature_name)
        elif rule_type == ValidationRule.UNIQUE.value:
            result = validator.validate_unique(data, feature_name)
        else:
            continue

        results.append(result)

    return results
