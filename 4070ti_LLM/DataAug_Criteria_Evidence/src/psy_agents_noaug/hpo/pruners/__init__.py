"""Advanced pruning strategies for efficient HPO (Phase 8).

This module provides sophisticated pruning algorithms to dramatically reduce
HPO runtime by early stopping underperforming trials:
- Hyperband: Aggressive successive halving
- BOHB: Bayesian Optimization + Hyperband
- Adaptive strategies: Dynamic resource allocation
- Comparison utilities: Benchmark different pruners

Also re-exports legacy pruner functions for backward compatibility.
"""

# Legacy functions (backward compatibility)
# Phase 8: Advanced pruners
from psy_agents_noaug.hpo.pruners.factory import (
    create_advanced_pruner,
    create_hyperband_pruner,
    create_median_pruner,
    create_percentile_pruner,
)
from psy_agents_noaug.hpo.pruners.resource import (
    AdaptiveResourceAllocator,
    estimate_trial_budget,
)
from psy_agents_noaug.hpo.pruners.strategies import (
    PrunerStrategy,
    compare_pruners,
    get_recommended_pruner,
    print_pruner_comparison,
)
from psy_agents_noaug.hpo.pruners_legacy import (
    create_pruner,
    stage_pruner,
)

__all__ = [
    # Legacy (backward compatibility)
    "create_pruner",
    "stage_pruner",
    # Phase 8: Advanced pruners
    "create_advanced_pruner",
    "create_hyperband_pruner",
    "create_median_pruner",
    "create_percentile_pruner",
    "PrunerStrategy",
    "get_recommended_pruner",
    "compare_pruners",
    "print_pruner_comparison",
    "AdaptiveResourceAllocator",
    "estimate_trial_budget",
]
