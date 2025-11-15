#!/usr/bin/env python
"""CI/CD integration and automated workflows (Phase 16).

This module provides comprehensive CI/CD infrastructure including:
- Workflow management and orchestration
- Quality gate validation
- Pipeline automation
- GitHub Actions integration

Key Features:
- Automated testing workflows
- Model validation pipelines
- Deployment automation
- Quality gate enforcement
"""

from __future__ import annotations

from psy_agents_noaug.cicd.gates import (
    QualityGate,
    QualityGateResult,
    validate_quality_gates,
)
from psy_agents_noaug.cicd.pipeline import (
    Pipeline,
    PipelineStage,
    run_pipeline,
)
from psy_agents_noaug.cicd.workflow import (
    WorkflowManager,
    WorkflowStep,
    run_workflow,
)

__all__ = [
    # Workflow
    "WorkflowManager",
    "WorkflowStep",
    "run_workflow",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    "run_pipeline",
    # Quality Gates
    "QualityGate",
    "QualityGateResult",
    "validate_quality_gates",
]
