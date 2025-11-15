"""Advanced sampler factory with BOHB-inspired strategies.

This module provides sophisticated sampling strategies that combine
Bayesian Optimization with multi-fidelity optimization:
- TPE (Tree-structured Parzen Estimator): Efficient BO
- CMA-ES: Covariance Matrix Adaptation Evolution Strategy
- GP: Gaussian Process-based Bayesian Optimization
- BOHB-style: TPE + Hyperband integration
- Multi-objective: Pareto optimization
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    NSGAIISampler,
    RandomSampler,
    TPESampler,
)

LOGGER = logging.getLogger(__name__)


def create_advanced_sampler(
    sampler_type: Literal[
        "tpe", "cmaes", "random", "nsga2", "bohb", "adaptive"
    ] = "tpe",
    *,
    seed: int | None = None,
    multi_objective: bool = False,
    n_startup_trials: int = 10,
    multivariate: bool = True,
    constant_liar: bool = True,
    **kwargs: Any,
) -> BaseSampler:
    """Create an advanced sampler with smart defaults.

    Args:
        sampler_type: Type of sampler to create
            - tpe: Tree-structured Parzen Estimator (default, efficient BO)
            - cmaes: CMA-ES for continuous optimization
            - random: Random search baseline
            - nsga2: Multi-objective genetic algorithm
            - bohb: TPE with BOHB-style configuration (multivariate + pruning)
            - adaptive: Auto-select based on problem characteristics
        seed: Random seed for reproducibility
        multi_objective: Whether this is multi-objective optimization
        n_startup_trials: Random trials before model-based sampling
        multivariate: Use multivariate TPE (considers parameter interactions)
        constant_liar: Parallel optimization support
        **kwargs: Additional sampler-specific arguments

    Returns:
        Configured sampler instance

    Raises:
        ValueError: If sampler_type is unknown

    Example:
        >>> # Standard TPE for single-objective
        >>> sampler = create_advanced_sampler("tpe", seed=42)
        >>>
        >>> # BOHB-style for aggressive optimization
        >>> sampler = create_advanced_sampler(
        ...     "bohb",
        ...     n_startup_trials=5,
        ...     multivariate=True,
        ... )
        >>>
        >>> # Multi-objective optimization
        >>> sampler = create_advanced_sampler(
        ...     "nsga2",
        ...     multi_objective=True,
        ...     population_size=50,
        ... )
    """
    if sampler_type == "random":
        LOGGER.info("Creating Random sampler (baseline)")
        return create_random_sampler(seed=seed)

    if sampler_type == "cmaes":
        LOGGER.info("Creating CMA-ES sampler")
        return create_cmaes_sampler(seed=seed, **kwargs)

    if sampler_type == "nsga2" or multi_objective:
        LOGGER.info("Creating NSGA-II sampler (multi-objective)")
        return create_nsga2_sampler(seed=seed, **kwargs)

    if sampler_type == "bohb":
        LOGGER.info("Creating BOHB-style TPE sampler")
        return create_bohb_sampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
            constant_liar=constant_liar,
            **kwargs,
        )

    if sampler_type == "adaptive":
        # Auto-select based on characteristics
        if multi_objective:
            return create_nsga2_sampler(seed=seed, **kwargs)
        return create_bohb_sampler(
            seed=seed,
            n_startup_trials=n_startup_trials,
            multivariate=multivariate,
            **kwargs,
        )

    # Default to TPE
    LOGGER.info("Creating TPE sampler (default)")
    return create_tpe_sampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        constant_liar=constant_liar,
        **kwargs,
    )


def create_tpe_sampler(
    *,
    seed: int | None = None,
    n_startup_trials: int = 10,
    multivariate: bool = False,
    constant_liar: bool = True,
    n_ei_candidates: int = 24,
) -> TPESampler:
    """Create TPE sampler with best practices.

    TPE (Tree-structured Parzen Estimator) is an efficient Bayesian
    optimization algorithm that models P(x|y) instead of P(y|x).

    Best for:
    - General-purpose hyperparameter optimization
    - Mixed continuous/categorical parameters
    - When you want efficient BO without GP overhead

    Args:
        seed: Random seed
        n_startup_trials: Random trials before model-based sampling
        multivariate: Consider parameter interactions (recommended)
        constant_liar: Enable parallel optimization
        n_ei_candidates: Expected improvement candidates (more = better but slower)

    Returns:
        Configured TPESampler

    Example:
        >>> sampler = create_tpe_sampler(
        ...     seed=42,
        ...     n_startup_trials=10,
        ...     multivariate=True,
        ... )
    """
    LOGGER.info(
        "TPE sampler: startup=%d, multivariate=%s, constant_liar=%s",
        n_startup_trials,
        multivariate,
        constant_liar,
    )

    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        constant_liar=constant_liar,
        n_ei_candidates=n_ei_candidates,
    )


def create_cmaes_sampler(
    *,
    seed: int | None = None,
    sigma0: float = 0.2,
    n_startup_trials: int = 5,
) -> CmaEsSampler:
    """Create CMA-ES sampler for continuous optimization.

    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is
    a powerful black-box optimizer for continuous parameters.

    Best for:
    - Continuous-only search spaces
    - Non-convex optimization landscapes
    - When you have a reasonable initialization

    Note:
    - Does NOT work with categorical/discrete parameters
    - Requires all parameters to be continuous

    Args:
        seed: Random seed
        sigma0: Initial standard deviation (0.1-0.3 typical)
        n_startup_trials: Trials before CMA-ES starts

    Returns:
        Configured CmaEsSampler

    Example:
        >>> sampler = create_cmaes_sampler(
        ...     seed=42,
        ...     sigma0=0.2,
        ... )
    """
    LOGGER.info(
        "CMA-ES sampler: sigma0=%.2f, startup=%d",
        sigma0,
        n_startup_trials,
    )

    return CmaEsSampler(
        seed=seed,
        sigma0=sigma0,
        n_startup_trials=n_startup_trials,
    )


def create_nsga2_sampler(
    *,
    seed: int | None = None,
    population_size: int = 50,
    mutation_prob: float | None = None,
    crossover_prob: float = 0.9,
) -> NSGAIISampler:
    """Create NSGA-II sampler for multi-objective optimization.

    NSGA-II (Non-dominated Sorting Genetic Algorithm II) finds
    the Pareto front for multi-objective problems.

    Best for:
    - Multi-objective optimization (2-3 objectives)
    - Finding trade-off solutions (Pareto front)
    - When you want diverse solution sets

    Args:
        seed: Random seed
        population_size: Population size (50-100 typical)
        mutation_prob: Mutation probability (None = auto)
        crossover_prob: Crossover probability (0.7-0.9)

    Returns:
        Configured NSGAIISampler

    Example:
        >>> sampler = create_nsga2_sampler(
        ...     seed=42,
        ...     population_size=50,
        ... )
    """
    LOGGER.info(
        "NSGA-II sampler: pop=%d, crossover=%.2f",
        population_size,
        crossover_prob,
    )

    return NSGAIISampler(
        seed=seed,
        population_size=population_size,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
    )


def create_random_sampler(
    *,
    seed: int | None = None,
) -> RandomSampler:
    """Create random sampler (baseline).

    Random search is surprisingly competitive and should always
    be used as a baseline for comparison.

    Best for:
    - Baseline comparisons
    - Quick sanity checks
    - When BO overhead isn't worth it

    Args:
        seed: Random seed

    Returns:
        Configured RandomSampler

    Example:
        >>> sampler = create_random_sampler(seed=42)
    """
    LOGGER.info("Random sampler: baseline")
    return RandomSampler(seed=seed)


def create_bohb_sampler(
    *,
    seed: int | None = None,
    n_startup_trials: int = 5,
    multivariate: bool = True,
    constant_liar: bool = True,
    n_ei_candidates: int = 32,
) -> TPESampler:
    """Create BOHB-style sampler (TPE optimized for Hyperband).

    BOHB combines Bayesian Optimization with HyperBand pruning.
    This creates a TPE sampler configured for aggressive early stopping.

    Strategy:
    - Few startup trials (aggressive model-based sampling)
    - Multivariate TPE (parameter interactions)
    - More EI candidates (better exploitation)
    - Constant liar for parallel trials

    Combine with HyperbandPruner for full BOHB effect!

    Best for:
    - Large-scale HPO (100s-1000s trials)
    - Expensive evaluations
    - When you pair with Hyperband pruner

    Args:
        seed: Random seed
        n_startup_trials: Random trials (fewer than standard TPE)
        multivariate: Always True for BOHB
        constant_liar: Parallel support
        n_ei_candidates: More candidates for better exploitation

    Returns:
        TPESampler configured for BOHB

    Example:
        >>> from psy_agents_noaug.hpo.pruners import create_hyperband_pruner
        >>> sampler = create_bohb_sampler(seed=42, n_startup_trials=5)
        >>> pruner = create_hyperband_pruner(min_resource=1, max_resource=27)
        >>> study = optuna.create_study(sampler=sampler, pruner=pruner)
    """
    LOGGER.info(
        "BOHB-style sampler: startup=%d, multivariate=%s, ei_candidates=%d",
        n_startup_trials,
        multivariate,
        n_ei_candidates,
    )

    return TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=multivariate,
        constant_liar=constant_liar,
        n_ei_candidates=n_ei_candidates,
    )


def get_sampler_info(sampler: BaseSampler) -> dict[str, Any]:
    """Get information about a sampler.

    Args:
        sampler: Sampler instance

    Returns:
        Dictionary with sampler information
    """
    info = {
        "type": type(sampler).__name__,
        "model_based": isinstance(sampler, (TPESampler, CmaEsSampler)),
        "multi_objective": isinstance(sampler, NSGAIISampler),
        "random": isinstance(sampler, RandomSampler),
    }

    # Add type-specific info
    if isinstance(sampler, TPESampler):
        # Access protected attributes (no public API)
        info["n_startup_trials"] = getattr(sampler, "_n_startup_trials", None)
        info["multivariate"] = getattr(sampler, "_multivariate", None)
    elif isinstance(sampler, CmaEsSampler):
        info["sigma0"] = getattr(sampler, "_sigma0", None)
    elif isinstance(sampler, NSGAIISampler):
        info["population_size"] = getattr(sampler, "_population_size", None)

    return info
