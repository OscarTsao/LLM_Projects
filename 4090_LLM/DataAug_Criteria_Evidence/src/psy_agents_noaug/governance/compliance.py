#!/usr/bin/env python
"""Compliance tracking and reporting (Phase 25).

This module provides tools for tracking compliance with regulatory frameworks
like GDPR, HIPAA, and other data protection regulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""

    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    CCPA = "ccpa"  # California Consumer Privacy Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # Information Security Management


@dataclass
class ComplianceRequirement:
    """Single compliance requirement."""

    requirement_id: str
    framework: ComplianceFramework
    description: str
    is_met: bool
    evidence: str = ""
    last_checked: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance assessment report."""

    framework: ComplianceFramework
    requirements: list[ComplianceRequirement]
    overall_compliant: bool
    compliance_score: float  # 0-1
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get compliance summary.

        Returns:
            Summary dictionary
        """
        total_reqs = len(self.requirements)
        met_reqs = sum(1 for req in self.requirements if req.is_met)

        return {
            "framework": self.framework.value,
            "total_requirements": total_reqs,
            "requirements_met": met_reqs,
            "compliance_score": self.compliance_score,
            "overall_compliant": self.overall_compliant,
            "generated_at": self.generated_at.isoformat(),
        }


class ComplianceTracker:
    """Tracker for regulatory compliance."""

    def __init__(self):
        """Initialize compliance tracker."""
        self.requirements: dict[ComplianceFramework, list[ComplianceRequirement]] = {}
        self._initialize_requirements()
        LOGGER.info("Initialized ComplianceTracker")

    def _initialize_requirements(self) -> None:
        """Initialize standard compliance requirements."""
        # GDPR requirements (simplified)
        self.requirements[ComplianceFramework.GDPR] = [
            ComplianceRequirement(
                requirement_id="GDPR-1",
                framework=ComplianceFramework.GDPR,
                description="Data minimization - collect only necessary data",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="GDPR-2",
                framework=ComplianceFramework.GDPR,
                description="Right to explanation - provide model explanations",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="GDPR-3",
                framework=ComplianceFramework.GDPR,
                description="Right to be forgotten - data deletion capability",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="GDPR-4",
                framework=ComplianceFramework.GDPR,
                description="Data protection impact assessment",
                is_met=False,
            ),
        ]

        # HIPAA requirements (simplified)
        self.requirements[ComplianceFramework.HIPAA] = [
            ComplianceRequirement(
                requirement_id="HIPAA-1",
                framework=ComplianceFramework.HIPAA,
                description="PHI encryption in transit and at rest",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="HIPAA-2",
                framework=ComplianceFramework.HIPAA,
                description="Access controls and audit trails",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="HIPAA-3",
                framework=ComplianceFramework.HIPAA,
                description="Minimum necessary standard",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="HIPAA-4",
                framework=ComplianceFramework.HIPAA,
                description="Data breach notification procedures",
                is_met=False,
            ),
        ]

        # CCPA requirements (simplified)
        self.requirements[ComplianceFramework.CCPA] = [
            ComplianceRequirement(
                requirement_id="CCPA-1",
                framework=ComplianceFramework.CCPA,
                description="Right to know what personal information is collected",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="CCPA-2",
                framework=ComplianceFramework.CCPA,
                description="Right to delete personal information",
                is_met=False,
            ),
            ComplianceRequirement(
                requirement_id="CCPA-3",
                framework=ComplianceFramework.CCPA,
                description="Right to opt-out of sale of personal information",
                is_met=False,
            ),
        ]

    def update_requirement(
        self,
        framework: ComplianceFramework,
        requirement_id: str,
        is_met: bool,
        evidence: str = "",
    ) -> bool:
        """Update compliance requirement status.

        Args:
            framework: Compliance framework
            requirement_id: Requirement ID
            is_met: Whether requirement is met
            evidence: Evidence of compliance

        Returns:
            True if updated, False if not found
        """
        if framework not in self.requirements:
            return False

        for req in self.requirements[framework]:
            if req.requirement_id == requirement_id:
                req.is_met = is_met
                req.evidence = evidence
                req.last_checked = datetime.now()
                LOGGER.info(f"Updated {requirement_id}: is_met={is_met}")
                return True

        return False

    def assess_compliance(
        self,
        framework: ComplianceFramework,
    ) -> ComplianceReport:
        """Assess compliance with framework.

        Args:
            framework: Compliance framework

        Returns:
            Compliance report
        """
        if framework not in self.requirements:
            msg = f"Unknown framework: {framework}"
            raise ValueError(msg)

        requirements = self.requirements[framework]
        total_reqs = len(requirements)
        met_reqs = sum(1 for req in requirements if req.is_met)

        # Calculate compliance score
        compliance_score = met_reqs / total_reqs if total_reqs > 0 else 0.0

        # Overall compliant if all requirements met
        overall_compliant = met_reqs == total_reqs

        return ComplianceReport(
            framework=framework,
            requirements=requirements.copy(),
            overall_compliant=overall_compliant,
            compliance_score=compliance_score,
        )

    def get_all_frameworks(self) -> list[ComplianceFramework]:
        """Get list of all tracked frameworks.

        Returns:
            List of frameworks
        """
        return list(self.requirements.keys())

    def get_requirement(
        self,
        framework: ComplianceFramework,
        requirement_id: str,
    ) -> ComplianceRequirement | None:
        """Get specific requirement.

        Args:
            framework: Compliance framework
            requirement_id: Requirement ID

        Returns:
            Requirement or None if not found
        """
        if framework not in self.requirements:
            return None

        for req in self.requirements[framework]:
            if req.requirement_id == requirement_id:
                return req

        return None

    def add_custom_requirement(
        self,
        framework: ComplianceFramework,
        requirement_id: str,
        description: str,
        is_met: bool = False,
    ) -> None:
        """Add custom compliance requirement.

        Args:
            framework: Compliance framework
            requirement_id: Requirement ID
            description: Requirement description
            is_met: Whether requirement is met
        """
        if framework not in self.requirements:
            self.requirements[framework] = []

        req = ComplianceRequirement(
            requirement_id=requirement_id,
            framework=framework,
            description=description,
            is_met=is_met,
        )

        self.requirements[framework].append(req)
        LOGGER.info(f"Added custom requirement: {requirement_id}")

    def generate_compliance_matrix(self) -> dict[str, Any]:
        """Generate compliance matrix across all frameworks.

        Returns:
            Compliance matrix
        """
        matrix = {}

        for framework in self.requirements:
            report = self.assess_compliance(framework)
            matrix[framework.value] = report.get_summary()

        return {
            "frameworks": matrix,
            "generated_at": datetime.now().isoformat(),
        }


def check_compliance(framework: ComplianceFramework) -> ComplianceReport:
    """Check compliance (convenience function).

    Args:
        framework: Compliance framework

    Returns:
        Compliance report
    """
    tracker = ComplianceTracker()
    return tracker.assess_compliance(framework)
