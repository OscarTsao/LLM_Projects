#!/usr/bin/env python
"""Alert system for monitoring thresholds (Phase 26).

This module provides an alert system for notifying about threshold violations,
anomalies, and other monitoring events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class AlertChannel(str, Enum):
    """Alert notification channels (backward compatibility from Phase 17)."""

    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    FILE = "file"


@dataclass
class Alert:
    """Single alert."""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE

    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None

    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()

    def get_duration(self) -> timedelta:
        """Get alert duration.

        Returns:
            Duration from creation to resolution (or now if unresolved)
        """
        end_time = self.resolved_at or datetime.now()
        return end_time - self.created_at

    def get_summary(self) -> dict[str, Any]:
        """Get alert summary.

        Returns:
            Summary dictionary
        """
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "duration_seconds": self.get_duration().total_seconds(),
            "details": self.details,
        }


@dataclass
class AlertRule:
    """Alert rule definition."""

    rule_id: str
    name: str
    condition: Callable[[], bool]  # Function that returns True if alert should fire
    severity: AlertSeverity
    description: str

    # Threshold settings
    cooldown_seconds: int = 300  # Don't re-alert within this period
    consecutive_violations: int = 1  # Require N consecutive violations

    # State
    last_alert_time: datetime | None = None
    consecutive_count: int = 0
    is_enabled: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manage alerts and alert rules."""

    def __init__(self):
        """Initialize alert manager."""
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self._alert_counter = 0

        LOGGER.info("Initialized AlertManager")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.rule_id] = rule
        LOGGER.info(f"Added alert rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule.

        Args:
            rule_id: Rule ID to remove

        Returns:
            True if removed, False if not found
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            LOGGER.info(f"Removed alert rule: {rule_id}")
            return True

        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if enabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].is_enabled = True
            LOGGER.info(f"Enabled alert rule: {rule_id}")
            return True

        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if disabled, False if not found
        """
        if rule_id in self.rules:
            self.rules[rule_id].is_enabled = False
            LOGGER.info(f"Disabled alert rule: {rule_id}")
            return True

        return False

    def check_rules(self) -> list[Alert]:
        """Check all rules and generate alerts.

        Returns:
            List of newly created alerts
        """
        new_alerts = []

        for rule in self.rules.values():
            if not rule.is_enabled:
                continue

            try:
                # Check condition
                is_violated = rule.condition()

                if is_violated:
                    rule.consecutive_count += 1

                    # Check if we should fire alert
                    if rule.consecutive_count >= rule.consecutive_violations:
                        # Check cooldown
                        if self._is_in_cooldown(rule):
                            continue

                        # Create alert
                        alert = self._create_alert(rule)
                        new_alerts.append(alert)

                        # Reset counter and update last alert time
                        rule.consecutive_count = 0
                        rule.last_alert_time = datetime.now()

                else:
                    # Reset counter if condition not violated
                    rule.consecutive_count = 0

            except Exception as e:
                LOGGER.error(f"Error checking rule {rule.rule_id}: {e}")

        return new_alerts

    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is in cooldown period.

        Args:
            rule: Alert rule

        Returns:
            True if in cooldown
        """
        if rule.last_alert_time is None:
            return False

        elapsed = (datetime.now() - rule.last_alert_time).total_seconds()
        return elapsed < rule.cooldown_seconds

    def _create_alert(self, rule: AlertRule) -> Alert:
        """Create alert from rule.

        Args:
            rule: Alert rule

        Returns:
            Created alert
        """
        self._alert_counter += 1
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._alert_counter:06d}"

        alert = Alert(
            alert_id=alert_id,
            title=rule.name,
            description=rule.description,
            severity=rule.severity,
            metadata={"rule_id": rule.rule_id},
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        LOGGER.warning(
            f"Alert fired: {alert.title} (severity={alert.severity.value})"
        )

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if resolved, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()

            # Remove from active
            del self.active_alerts[alert_id]

            LOGGER.info(f"Resolved alert: {alert_id}")
            return True

        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if acknowledged, False if not found
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledge()

            LOGGER.info(f"Acknowledged alert: {alert_id}")
            return True

        return False

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get active alerts.

        Args:
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_alert_history(
        self,
        since: datetime | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get alert history.

        Args:
            since: Get alerts since this time
            severity: Filter by severity

        Returns:
            List of alerts
        """
        alerts = self.alert_history

        if since:
            alerts = [a for a in alerts if a.created_at >= since]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary.

        Returns:
            Summary dictionary
        """
        # Count by severity
        severity_counts = {
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        }

        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        # Count by status
        status_counts = {
            "active": len(self.active_alerts),
            "resolved": sum(
                1 for a in self.alert_history if a.status == AlertStatus.RESOLVED
            ),
            "acknowledged": sum(
                1 for a in self.active_alerts.values()
                if a.status == AlertStatus.ACKNOWLEDGED
            ),
        }

        return {
            "total_active": len(self.active_alerts),
            "total_history": len(self.alert_history),
            "by_severity": severity_counts,
            "by_status": status_counts,
            "num_rules": len(self.rules),
            "num_enabled_rules": sum(1 for r in self.rules.values() if r.is_enabled),
        }


def create_threshold_rule(
    rule_id: str,
    name: str,
    get_value: Callable[[], float],
    threshold: float,
    comparison: str = ">",  # ">", "<", ">=", "<=", "==", "!="
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertRule:
    """Create a threshold-based alert rule.

    Args:
        rule_id: Rule ID
        name: Rule name
        get_value: Function to get current value
        threshold: Threshold value
        comparison: Comparison operator
        severity: Alert severity

    Returns:
        Alert rule

    Example:
        rule = create_threshold_rule(
            rule_id="high_latency",
            name="High Latency",
            get_value=lambda: monitor.get_current_metrics().mean_latency,
            threshold=1.0,
            comparison=">",
            severity=AlertSeverity.WARNING,
        )
    """
    comparisons = {
        ">": lambda v, t: v > t,
        "<": lambda v, t: v < t,
        ">=": lambda v, t: v >= t,
        "<=": lambda v, t: v <= t,
        "==": lambda v, t: v == t,
        "!=": lambda v, t: v != t,
    }

    if comparison not in comparisons:
        msg = f"Invalid comparison: {comparison}"
        raise ValueError(msg)

    def condition() -> bool:
        value = get_value()
        return comparisons[comparison](value, threshold)

    return AlertRule(
        rule_id=rule_id,
        name=name,
        condition=condition,
        severity=severity,
        description=f"{name}: value {comparison} {threshold}",
        metadata={"threshold": threshold, "comparison": comparison},
    )


def create_drift_alert_rule(
    rule_id: str,
    check_drift: Callable[[], bool],  # Returns True if drift detected
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertRule:
    """Create a drift detection alert rule.

    Args:
        rule_id: Rule ID
        check_drift: Function that returns True if drift detected
        severity: Alert severity

    Returns:
        Alert rule
    """
    return AlertRule(
        rule_id=rule_id,
        name="Data Drift Detected",
        condition=check_drift,
        severity=severity,
        description="Prediction distribution has drifted from reference",
        consecutive_violations=2,  # Require 2 consecutive detections
    )
