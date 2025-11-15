#!/usr/bin/env python
"""A/B Testing & Experimentation Framework (Phase 21).

This module provides production-ready A/B testing infrastructure including:
- Traffic splitting and routing
- Experiment configuration and management
- Statistical analysis and significance testing
- Experiment tracking and reporting
- Multi-armed bandit algorithms

Key Features:
- Multiple traffic splitting strategies
- Statistical significance testing (frequentist & Bayesian)
- Automatic winner detection
- Comprehensive experiment tracking
- Integration with model registry
"""

from __future__ import annotations

from psy_agents_noaug.ab_testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentManager,
    Variant,
    create_experiment,
)
from psy_agents_noaug.ab_testing.stats import (
    BayesianAnalyzer,
    FrequentistAnalyzer,
    StatisticalTest,
    calculate_sample_size,
)
from psy_agents_noaug.ab_testing.tracking import (
    ExperimentTracker,
    MetricAggregator,
    MetricEvent,
    track_conversion,
)
from psy_agents_noaug.ab_testing.traffic import (
    SplitStrategy,
    TrafficAllocation,
    TrafficSplitter,
    split_traffic,
)

__all__ = [
    # Experiment
    "Experiment",
    "ExperimentConfig",
    "ExperimentManager",
    "Variant",
    "create_experiment",
    # Statistics
    "BayesianAnalyzer",
    "FrequentistAnalyzer",
    "StatisticalTest",
    "calculate_sample_size",
    # Tracking
    "ExperimentTracker",
    "MetricAggregator",
    "MetricEvent",
    "track_conversion",
    # Traffic
    "SplitStrategy",
    "TrafficAllocation",
    "TrafficSplitter",
    "split_traffic",
]
