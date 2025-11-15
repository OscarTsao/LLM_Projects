#!/usr/bin/env python
"""Feature computation engine (Phase 24).

This module provides tools for computing features, managing dependencies,
and caching intermediate results.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ComputationResult:
    """Result of feature computation."""

    feature_name: str
    value: Any
    computed_at: datetime = field(default_factory=datetime.now)
    computation_time_ms: float = 0.0
    cached: bool = False
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureComputationEngine:
    """Engine for computing features with dependency resolution."""

    def __init__(self, enable_cache: bool = True):
        """Initialize computation engine.

        Args:
            enable_cache: Enable result caching
        """
        self.enable_cache = enable_cache
        self.cache: dict[str, ComputationResult] = {}
        self.computation_graph: dict[str, list[str]] = {}
        LOGGER.info(f"Initialized FeatureComputationEngine (cache={enable_cache})")

    def register_feature(
        self,
        feature_name: str,
        dependencies: list[str] | None = None,
    ) -> None:
        """Register feature and its dependencies.

        Args:
            feature_name: Feature name
            dependencies: List of dependency feature names
        """
        self.computation_graph[feature_name] = dependencies or []
        LOGGER.debug(
            f"Registered feature {feature_name} with {len(dependencies or [])} dependencies"
        )

    def get_cache_key(self, feature_name: str, data: Any) -> str:
        """Generate cache key for feature and data.

        Args:
            feature_name: Feature name
            data: Input data

        Returns:
            Cache key (hash)
        """
        # Simple hash of feature name and data
        data_str = str(data)
        key_input = f"{feature_name}:{data_str}"
        return hashlib.md5(key_input.encode()).hexdigest()

    def get_from_cache(self, cache_key: str) -> ComputationResult | None:
        """Get result from cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached result or None
        """
        if not self.enable_cache:
            return None

        return self.cache.get(cache_key)

    def put_in_cache(self, cache_key: str, result: ComputationResult) -> None:
        """Put result in cache.

        Args:
            cache_key: Cache key
            result: Computation result
        """
        if self.enable_cache:
            self.cache[cache_key] = result

    def compute_single(
        self,
        feature_name: str,
        compute_fn: Any,
        data: Any,
    ) -> ComputationResult:
        """Compute a single feature.

        Args:
            feature_name: Feature name
            compute_fn: Computation function
            data: Input data

        Returns:
            Computation result
        """
        # Check cache
        cache_key = self.get_cache_key(feature_name, data)
        cached_result = self.get_from_cache(cache_key)

        if cached_result:
            LOGGER.debug(f"Cache hit for {feature_name}")
            cached_result.cached = True
            return cached_result

        # Compute
        start_time = datetime.now()
        try:
            value = compute_fn(data)
        except Exception:
            LOGGER.exception(f"Error computing {feature_name}")
            raise

        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds() * 1000

        # Create result
        result = ComputationResult(
            feature_name=feature_name,
            value=value,
            computed_at=end_time,
            computation_time_ms=computation_time,
            cached=False,
            dependencies=self.computation_graph.get(feature_name, []),
        )

        # Cache result
        self.put_in_cache(cache_key, result)

        LOGGER.debug(f"Computed {feature_name} in {computation_time:.2f}ms")
        return result

    def resolve_dependencies(self, feature_name: str) -> list[str]:
        """Resolve feature dependencies in topological order.

        Args:
            feature_name: Feature name

        Returns:
            List of features in computation order
        """
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return

            # Visit dependencies first
            for dep in self.computation_graph.get(name, []):
                visit(dep)

            visited.add(name)
            order.append(name)

        visit(feature_name)
        return order

    def compute_with_dependencies(
        self,
        feature_name: str,
        compute_fns: dict[str, Any],
        data: Any,
    ) -> dict[str, ComputationResult]:
        """Compute feature and all dependencies.

        Args:
            feature_name: Feature name
            compute_fns: Dictionary of feature_name -> compute_fn
            data: Input data

        Returns:
            Dictionary of feature_name -> ComputationResult
        """
        # Resolve dependencies
        computation_order = self.resolve_dependencies(feature_name)

        # Compute in order
        results = {}
        for name in computation_order:
            if name not in compute_fns:
                msg = f"No computation function for feature {name}"
                raise ValueError(msg)

            result = self.compute_single(name, compute_fns[name], data)
            results[name] = result

        return results

    def compute_batch(
        self,
        feature_names: list[str],
        compute_fns: dict[str, Any],
        data_batch: list[Any],
    ) -> list[dict[str, ComputationResult]]:
        """Compute multiple features for a batch of data.

        Args:
            feature_names: List of feature names
            compute_fns: Dictionary of feature_name -> compute_fn
            data_batch: List of input data

        Returns:
            List of results (one per data item)
        """
        batch_results = []

        for data in data_batch:
            data_results = {}
            for feature_name in feature_names:
                if feature_name not in compute_fns:
                    msg = f"No computation function for feature {feature_name}"
                    raise ValueError(msg)

                result = self.compute_single(
                    feature_name, compute_fns[feature_name], data
                )
                data_results[feature_name] = result

            batch_results.append(data_results)

        return batch_results

    def clear_cache(self, feature_name: str | None = None) -> None:
        """Clear computation cache.

        Args:
            feature_name: Specific feature to clear (None for all)
        """
        if feature_name:
            # Clear only for specific feature
            keys_to_remove = [k for k in self.cache if feature_name in k]
            for key in keys_to_remove:
                del self.cache[key]
            LOGGER.info(f"Cleared cache for {feature_name}")
        else:
            # Clear all
            self.cache.clear()
            LOGGER.info("Cleared all cache")

    def get_statistics(self) -> dict[str, Any]:
        """Get computation statistics.

        Returns:
            Statistics dictionary
        """
        total_computations = len(self.cache)
        cached_computations = sum(1 for r in self.cache.values() if r.cached)

        avg_computation_time = 0.0
        if self.cache:
            avg_computation_time = sum(
                r.computation_time_ms for r in self.cache.values()
            ) / len(self.cache)

        return {
            "total_features": len(self.computation_graph),
            "cached_results": total_computations,
            "cache_hit_rate": (
                cached_computations / total_computations
                if total_computations > 0
                else 0.0
            ),
            "avg_computation_time_ms": avg_computation_time,
        }


def compute_features(
    feature_names: list[str],
    compute_fns: dict[str, Any],
    data: Any,
    enable_cache: bool = True,
) -> dict[str, Any]:
    """Compute features (convenience function).

    Args:
        feature_names: List of feature names
        compute_fns: Dictionary of feature_name -> compute_fn
        data: Input data
        enable_cache: Enable caching

    Returns:
        Dictionary of feature_name -> value
    """
    engine = FeatureComputationEngine(enable_cache=enable_cache)

    results = {}
    for feature_name in feature_names:
        if feature_name not in compute_fns:
            msg = f"No computation function for feature {feature_name}"
            raise ValueError(msg)

        result = engine.compute_single(feature_name, compute_fns[feature_name], data)
        results[feature_name] = result.value

    return results
