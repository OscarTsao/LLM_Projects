"""Ensemble selection and combination utilities for SUPERMAX Phase 6.

This module provides tools for building powerful ensembles from Pareto fronts:
- Diversity-based selection from multi-objective optimization
- Weighted voting ensembles
- Stacking with meta-learners
- Ensemble weight optimization
"""

from psy_agents_noaug.ensemble.selection import (
    DiversitySelector,
    select_diverse_pareto,
    compute_config_distance,
)
from psy_agents_noaug.ensemble.voting import (
    WeightedVotingEnsemble,
    optimize_voting_weights,
)
from psy_agents_noaug.ensemble.stacking import (
    StackingEnsemble,
    train_meta_learner,
)

__all__ = [
    "DiversitySelector",
    "select_diverse_pareto",
    "compute_config_distance",
    "WeightedVotingEnsemble",
    "optimize_voting_weights",
    "StackingEnsemble",
    "train_meta_learner",
]
