"""Advanced sampling strategies for efficient HPO (Phase 9).

This module provides sophisticated sampling algorithms that combine
Bayesian Optimization with multi-fidelity optimization:
- TPE: Tree-structured Parzen Estimator (efficient BO)
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- NSGA-II: Multi-objective genetic algorithm
- BOHB-style: TPE + Hyperband integration
- Intelligent selection strategies
- Comparison utilities

Also re-exports legacy sampler functions for backward compatibility.
"""

# Legacy functions (backward compatibility)
from psy_agents_noaug.hpo.samplers_legacy import (
    create_sampler,
)

# Phase 9: Advanced samplers
from psy_agents_noaug.hpo.samplers.factory import (
    create_advanced_sampler,
    create_bohb_sampler,
    create_cmaes_sampler,
    create_nsga2_sampler,
    create_random_sampler,
    create_tpe_sampler,
    get_sampler_info,
)
from psy_agents_noaug.hpo.samplers.strategies import (
    SamplerStrategy,
    compare_samplers,
    get_recommended_sampler,
    print_sampler_comparison,
)

__all__ = [
    # Legacy (backward compatibility)
    "create_sampler",
    # Factory functions
    "create_advanced_sampler",
    "create_tpe_sampler",
    "create_cmaes_sampler",
    "create_nsga2_sampler",
    "create_random_sampler",
    "create_bohb_sampler",
    "get_sampler_info",
    # Strategy functions
    "SamplerStrategy",
    "get_recommended_sampler",
    "compare_samplers",
    "print_sampler_comparison",
]
