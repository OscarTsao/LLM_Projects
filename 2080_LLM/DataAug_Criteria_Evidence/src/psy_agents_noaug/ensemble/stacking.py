"""Stacking ensemble with meta-learner for advanced combination.

This module implements stacking (stacked generalization) where a meta-learner
is trained on base model predictions to learn optimal combinations.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss

LOGGER = logging.getLogger(__name__)


class StackingEnsemble:
    """Stacking ensemble with trainable meta-learner.

    Uses base model predictions as features for a meta-learner that
    learns to optimally combine them.
    """

    def __init__(
        self,
        meta_learner: Any | None = None,
        meta_learner_type: Literal["logistic", "rf", "custom"] = "logistic",
        use_probs: bool = True,
        use_features: bool = False,
    ):
        """Initialize stacking ensemble.

        Args:
            meta_learner: Pre-initialized meta-learner (if None, will create based on type)
            meta_learner_type: Type of meta-learner ("logistic", "rf", "custom")
            use_probs: Use probability predictions as meta-features
            use_features: Also use original features (requires base model features)
        """
        self.meta_learner_type = meta_learner_type
        self.use_probs = use_probs
        self.use_features = use_features
        self._is_fitted = False

        # Initialize meta-learner
        if meta_learner is not None:
            self.meta_learner = meta_learner
        elif meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42,
            )
        elif meta_learner_type == "rf":
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown meta_learner_type: {meta_learner_type}")

    def fit(
        self,
        model_probs: list[NDArray[np.floating]],
        y_true: NDArray[np.int_],
        original_features: NDArray[np.floating] | None = None,
    ) -> StackingEnsemble:
        """Train meta-learner on base model predictions.

        Args:
            model_probs: List of probability arrays from base models
            y_true: Ground truth labels
            original_features: Original input features (if use_features=True)

        Returns:
            Self
        """
        # Construct meta-features
        meta_features = self._construct_meta_features(model_probs, original_features)

        # Train meta-learner
        LOGGER.info(
            "Training meta-learner (%s) on %d samples with %d features",
            self.meta_learner_type,
            meta_features.shape[0],
            meta_features.shape[1],
        )
        self.meta_learner.fit(meta_features, y_true)

        self._is_fitted = True
        LOGGER.info("Meta-learner training complete")
        return self

    def predict_proba(
        self,
        model_probs: list[NDArray[np.floating]],
        original_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Predict class probabilities using meta-learner.

        Args:
            model_probs: List of probability arrays from base models
            original_features: Original input features (if use_features=True)

        Returns:
            Meta-learner probability predictions
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Construct meta-features
        meta_features = self._construct_meta_features(model_probs, original_features)

        # Predict with meta-learner
        return self.meta_learner.predict_proba(meta_features)

    def predict(
        self,
        model_probs: list[NDArray[np.floating]],
        original_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.int_]:
        """Predict class labels using meta-learner.

        Args:
            model_probs: List of probability arrays from base models
            original_features: Original input features (if use_features=True)

        Returns:
            Meta-learner class predictions
        """
        probs = self.predict_proba(model_probs, original_features)
        return np.argmax(probs, axis=1)

    def evaluate(
        self,
        model_probs: list[NDArray[np.floating]],
        y_true: NDArray[np.int_],
        original_features: NDArray[np.floating] | None = None,
    ) -> dict[str, float]:
        """Evaluate stacking ensemble on test data.

        Args:
            model_probs: List of probability arrays from base models
            y_true: Ground truth labels
            original_features: Original input features (if use_features=True)

        Returns:
            Dictionary of metrics
        """
        probs = self.predict_proba(model_probs, original_features)
        preds = np.argmax(probs, axis=1)

        metrics = {
            "f1_macro": f1_score(y_true, preds, average="macro"),
            "f1_micro": f1_score(y_true, preds, average="micro"),
            "logloss": log_loss(y_true, probs),
        }

        LOGGER.info(
            "Stacking evaluation: F1-macro=%.4f, Logloss=%.4f",
            metrics["f1_macro"],
            metrics["logloss"],
        )
        return metrics

    def _construct_meta_features(
        self,
        model_probs: list[NDArray[np.floating]],
        original_features: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Construct meta-features from base model predictions.

        Args:
            model_probs: List of probability arrays from base models
            original_features: Original input features

        Returns:
            Meta-feature matrix
        """
        features_list = []

        if self.use_probs:
            # Flatten probability predictions from all models
            for probs in model_probs:
                features_list.append(probs)

        if self.use_features and original_features is not None:
            features_list.append(original_features)
        elif self.use_features and original_features is None:
            LOGGER.warning("use_features=True but original_features not provided")

        # Concatenate all features
        meta_features = np.concatenate(features_list, axis=1)
        return meta_features


def train_meta_learner(
    model_probs: list[NDArray[np.floating]],
    y_true: NDArray[np.int_],
    meta_learner_type: str = "logistic",
    **kwargs: Any,
) -> StackingEnsemble:
    """Convenience function to train stacking ensemble.

    Args:
        model_probs: List of probability arrays from base models
        y_true: Ground truth labels
        meta_learner_type: Type of meta-learner
        **kwargs: Additional arguments for StackingEnsemble

    Returns:
        Trained stacking ensemble
    """
    ensemble = StackingEnsemble(meta_learner_type=meta_learner_type, **kwargs)
    ensemble.fit(model_probs, y_true)
    return ensemble


def compare_ensemble_strategies(
    model_probs_train: list[NDArray[np.floating]],
    model_probs_test: list[NDArray[np.floating]],
    y_train: NDArray[np.int_],
    y_test: NDArray[np.int_],
    model_scores: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare different ensemble strategies (voting vs. stacking).

    Args:
        model_probs_train: Training probabilities from base models
        model_probs_test: Test probabilities from base models
        y_train: Training labels
        y_test: Test labels
        model_scores: Individual model F1 scores

    Returns:
        Dictionary mapping strategy name to metrics
    """
    from psy_agents_noaug.ensemble.voting import WeightedVotingEnsemble

    results = {}

    # 1. Uniform voting
    uniform_ensemble = WeightedVotingEnsemble(weight_strategy="uniform")
    uniform_ensemble.fit(model_probs_train, y_train)
    results["uniform_voting"] = uniform_ensemble.evaluate(model_probs_test, y_test)

    # 2. Performance-based voting
    if model_scores is not None:
        perf_ensemble = WeightedVotingEnsemble(weight_strategy="performance")
        perf_ensemble.fit(model_probs_train, y_train, model_scores=model_scores)
        results["performance_voting"] = perf_ensemble.evaluate(model_probs_test, y_test)

    # 3. Optimized voting
    opt_ensemble = WeightedVotingEnsemble(weight_strategy="optimized")
    opt_ensemble.fit(model_probs_train, y_train)
    results["optimized_voting"] = opt_ensemble.evaluate(model_probs_test, y_test)

    # 4. Stacking with logistic regression
    stack_lr = StackingEnsemble(meta_learner_type="logistic")
    stack_lr.fit(model_probs_train, y_train)
    results["stacking_logistic"] = stack_lr.evaluate(model_probs_test, y_test)

    # 5. Stacking with random forest
    stack_rf = StackingEnsemble(meta_learner_type="rf")
    stack_rf.fit(model_probs_train, y_train)
    results["stacking_rf"] = stack_rf.evaluate(model_probs_test, y_test)

    # Log comparison
    LOGGER.info("=" * 80)
    LOGGER.info("Ensemble Strategy Comparison")
    LOGGER.info("=" * 80)
    for strategy, metrics in sorted(results.items(), key=lambda x: -x[1]["f1_macro"]):
        LOGGER.info(
            "%-25s: F1=%.4f, Logloss=%.4f",
            strategy,
            metrics["f1_macro"],
            metrics["logloss"],
        )

    return results
