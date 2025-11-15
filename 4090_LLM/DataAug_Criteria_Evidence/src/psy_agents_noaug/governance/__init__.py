#!/usr/bin/env python
"""Model Governance & Compliance (Phase 25).

This module provides tools for model governance, compliance, bias detection,
and audit trails for production ML systems.
"""

from __future__ import annotations

from psy_agents_noaug.governance.audit_trail import (
    AuditEvent,
    AuditLogger,
    EventType,
    log_event,
)
from psy_agents_noaug.governance.bias_detection import (
    BiasDetector,
    BiasMetrics,
    FairnessMetric,
    detect_bias,
)
from psy_agents_noaug.governance.compliance import (
    ComplianceFramework,
    ComplianceReport,
    ComplianceTracker,
    check_compliance,
)
from psy_agents_noaug.governance.model_card import (
    ModelCard,
    ModelCardGenerator,
    generate_model_card,
)

__all__ = [
    # Model Cards
    "ModelCard",
    "ModelCardGenerator",
    "generate_model_card",
    # Bias Detection
    "BiasDetector",
    "BiasMetrics",
    "FairnessMetric",
    "detect_bias",
    # Compliance
    "ComplianceFramework",
    "ComplianceReport",
    "ComplianceTracker",
    "check_compliance",
    # Audit Trail
    "AuditEvent",
    "AuditLogger",
    "EventType",
    "log_event",
]
