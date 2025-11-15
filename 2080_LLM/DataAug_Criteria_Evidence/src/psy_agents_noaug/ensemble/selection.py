"""Diversity-based selection from Pareto fronts for ensemble building.

This module implements intelligent selection strategies to choose diverse,
high-performing models from multi-objective optimization results.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

LOGGER = logging.getLogger(__name__)


class DiversitySelector:
    """Select diverse configurations from Pareto front for ensemble.

    Uses multiple strategies:
    1. Performance-based: Top-K by primary metric
    2. Diversity-based: Maximum config space distance
    3. Cluster-based: Representatives from K clusters
    4. Hybrid: Balance performance and diversity
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        n_models: int = 5,
        diversity_weight: float = 0.3,
        random_state: int = 42,
    ):
        """Initialize diversity selector.

        Args:
            strategy: Selection strategy ("topk", "diversity", "cluster", "hybrid")
            n_models: Number of models to select for ensemble
            diversity_weight: Weight for diversity vs. performance (0-1)
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.n_models = n_models
        self.diversity_weight = diversity_weight
        self.random_state = random_state

    def select(
        self,
        pareto_front: list[dict[str, Any]],
        config_key: str = "config",
        score_key: str = "f1_macro",
    ) -> list[dict[str, Any]]:
        """Select diverse models from Pareto front.

        Args:
            pareto_front: List of Pareto front entries with configs and scores
            config_key: Key for configuration dictionary
            score_key: Key for performance score (for "topk" or "hybrid")

        Returns:
            List of selected configurations for ensemble
        """
        if len(pareto_front) <= self.n_models:
            LOGGER.warning(
                "Pareto front size (%d) <= n_models (%d), selecting all",
                len(pareto_front),
                self.n_models,
            )
            return pareto_front

        if self.strategy == "topk":
            return self._select_topk(pareto_front, score_key)
        if self.strategy == "diversity":
            return self._select_diversity(pareto_front, config_key)
        if self.strategy == "cluster":
            return self._select_cluster(pareto_front, config_key)
        if self.strategy == "hybrid":
            return self._select_hybrid(pareto_front, config_key, score_key)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_topk(
        self,
        pareto_front: list[dict[str, Any]],
        score_key: str,
    ) -> list[dict[str, Any]]:
        """Select top-K by performance score."""
        sorted_front = sorted(
            pareto_front,
            key=lambda x: x.get(score_key, x.get("f1_macro_mean", 0.0)),
            reverse=True,
        )
        selected = sorted_front[: self.n_models]
        LOGGER.info("Selected top-%d models by %s", self.n_models, score_key)
        return selected

    def _select_diversity(
        self,
        pareto_front: list[dict[str, Any]],
        config_key: str,
    ) -> list[dict[str, Any]]:
        """Select models with maximum configuration diversity."""
        # Extract configurations
        configs = [entry[config_key] for entry in pareto_front]

        # Compute pairwise config distances
        dist_matrix = compute_config_distance_matrix(configs)

        # Greedy selection: iteratively pick most diverse model
        selected_indices = [0]  # Start with first model
        for _ in range(self.n_models - 1):
            # For each candidate, compute min distance to selected set
            candidates = [i for i in range(len(configs)) if i not in selected_indices]
            min_dists = []
            for cand in candidates:
                # Min distance to any selected model
                dists_to_selected = [dist_matrix[cand, s] for s in selected_indices]
                min_dists.append(min(dists_to_selected))

            # Pick candidate with maximum min distance (most diverse)
            best_cand_idx = candidates[np.argmax(min_dists)]
            selected_indices.append(best_cand_idx)

        selected = [pareto_front[i] for i in selected_indices]
        LOGGER.info("Selected %d diverse models (max config distance)", self.n_models)
        return selected

    def _select_cluster(
        self,
        pareto_front: list[dict[str, Any]],
        config_key: str,
    ) -> list[dict[str, Any]]:
        """Select representative models from K clusters."""
        # Extract configurations
        configs = [entry[config_key] for entry in pareto_front]

        # Vectorize configs
        config_vectors = vectorize_configs(configs)

        # Cluster into K groups
        kmeans = KMeans(
            n_clusters=self.n_models,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(config_vectors)

        # Select best model from each cluster
        selected = []
        for cluster_id in range(self.n_models):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Pick model closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            cluster_models = [pareto_front[i] for i in cluster_indices]
            cluster_vectors = config_vectors[cluster_indices]

            dists_to_center = np.linalg.norm(cluster_vectors - cluster_center, axis=1)
            best_in_cluster = cluster_models[np.argmin(dists_to_center)]
            selected.append(best_in_cluster)

        LOGGER.info(
            "Selected %d cluster representatives from %d clusters",
            len(selected),
            self.n_models,
        )
        return selected

    def _select_hybrid(
        self,
        pareto_front: list[dict[str, Any]],
        config_key: str,
        score_key: str,
    ) -> list[dict[str, Any]]:
        """Balance performance and diversity using weighted scoring."""
        # Extract configs and scores
        configs = [entry[config_key] for entry in pareto_front]
        scores = np.array(
            [
                entry.get(score_key, entry.get("f1_macro_mean", 0.0))
                for entry in pareto_front
            ]
        )

        # Normalize scores to [0, 1]
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Compute pairwise config distances
        dist_matrix = compute_config_distance_matrix(configs)

        # Greedy selection with hybrid score
        selected_indices = []
        for iteration in range(self.n_models):
            if iteration == 0:
                # Start with best performing model
                best_idx = np.argmax(scores_norm)
                selected_indices.append(best_idx)
            else:
                # For each candidate, compute hybrid score
                candidates = [
                    i for i in range(len(configs)) if i not in selected_indices
                ]
                hybrid_scores = []

                for cand in candidates:
                    # Performance component
                    perf_score = scores_norm[cand]

                    # Diversity component (min distance to selected set)
                    dists_to_selected = [dist_matrix[cand, s] for s in selected_indices]
                    div_score = min(dists_to_selected)

                    # Normalize diversity score
                    if dist_matrix.max() > 0:
                        div_score = div_score / dist_matrix.max()

                    # Hybrid score (weighted combination)
                    hybrid = (
                        1 - self.diversity_weight
                    ) * perf_score + self.diversity_weight * div_score
                    hybrid_scores.append(hybrid)

                # Pick candidate with highest hybrid score
                best_cand_idx = candidates[np.argmax(hybrid_scores)]
                selected_indices.append(best_cand_idx)

        selected = [pareto_front[i] for i in selected_indices]
        LOGGER.info(
            "Selected %d models with hybrid strategy (perf=%.1f%%, div=%.1f%%)",
            self.n_models,
            (1 - self.diversity_weight) * 100,
            self.diversity_weight * 100,
        )
        return selected


def select_diverse_pareto(
    pareto_front: list[dict[str, Any]],
    n_models: int = 5,
    strategy: str = "hybrid",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Convenience function for diversity-based selection.

    Args:
        pareto_front: Pareto front entries
        n_models: Number of models to select
        strategy: Selection strategy
        **kwargs: Additional arguments for DiversitySelector

    Returns:
        Selected configurations for ensemble
    """
    selector = DiversitySelector(
        strategy=strategy,
        n_models=n_models,
        **kwargs,
    )
    return selector.select(pareto_front)


def compute_config_distance(
    config1: dict[str, Any],
    config2: dict[str, Any],
) -> float:
    """Compute distance between two configurations.

    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary

    Returns:
        Distance score (0 = identical, higher = more different)
    """
    # Get union of keys
    all_keys = set(config1.keys()) | set(config2.keys())

    # Count differences
    differences = 0
    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)

        if val1 is None or val2 is None:
            differences += 1
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numeric: normalized difference
            val1_norm = float(val1)
            val2_norm = float(val2)
            differences += abs(val1_norm - val2_norm) / (
                abs(val1_norm) + abs(val2_norm) + 1e-8
            )
        elif val1 != val2:
            # Categorical: binary difference
            differences += 1

    return differences / len(all_keys) if all_keys else 0.0


def compute_config_distance_matrix(
    configs: list[dict[str, Any]],
) -> NDArray[np.floating]:
    """Compute pairwise distance matrix for configurations.

    Args:
        configs: List of configuration dictionaries

    Returns:
        NxN distance matrix
    """
    n = len(configs)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_config_distance(configs[i], configs[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def vectorize_configs(configs: list[dict[str, Any]]) -> NDArray[np.floating]:
    """Convert configuration dictionaries to numeric vectors.

    Args:
        configs: List of configuration dictionaries

    Returns:
        NxD matrix of config vectors
    """
    # Get all keys
    all_keys = sorted(set().union(*[set(c.keys()) for c in configs]))

    # Create mapping for categorical values
    categorical_maps: dict[str, dict[Any, int]] = {}
    for key in all_keys:
        values = [c.get(key) for c in configs if key in c]
        unique_vals = sorted(set(v for v in values if not isinstance(v, (int, float))))
        if unique_vals:
            categorical_maps[key] = {val: i for i, val in enumerate(unique_vals)}

    # Vectorize
    vectors = []
    for config in configs:
        vec = []
        for key in all_keys:
            val = config.get(key, 0)
            if isinstance(val, (int, float)):
                vec.append(float(val))
            elif key in categorical_maps:
                vec.append(float(categorical_maps[key].get(val, -1)))
            else:
                vec.append(0.0)
        vectors.append(vec)

    return np.array(vectors)
