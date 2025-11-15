#!/usr/bin/env python
"""Statistical analysis for A/B testing (Phase 21).

This module provides:
- Statistical significance testing
- Confidence intervals
- Sample size estimation
- Bayesian analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class StatisticalTest:
    """Statistical test result."""

    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: float
    confidence_interval: tuple[float, float]
    test_name: str


class FrequentistAnalyzer:
    """Frequentist statistical analyzer."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize analyzer.

        Args:
            significance_level: Alpha level for significance
        """
        self.significance_level = significance_level
        LOGGER.info(f"Initialized FrequentistAnalyzer (alpha={significance_level})")

    def t_test(self, values_a: list[float], values_b: list[float]) -> StatisticalTest:
        """Perform two-sample t-test.

        Args:
            values_a: Values for group A
            values_b: Values for group B

        Returns:
            Statistical test result
        """
        if not values_a or not values_b:
            return StatisticalTest(
                p_value=1.0,
                is_significant=False,
                confidence_level=1 - self.significance_level,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                test_name="t-test",
            )

        # Calculate means
        mean_a = sum(values_a) / len(values_a)
        mean_b = sum(values_b) / len(values_b)

        # Calculate variances
        var_a = (
            sum((x - mean_a) ** 2 for x in values_a) / (len(values_a) - 1)
            if len(values_a) > 1
            else 0
        )
        var_b = (
            sum((x - mean_b) ** 2 for x in values_b) / (len(values_b) - 1)
            if len(values_b) > 1
            else 0
        )

        # Pooled standard error
        se = math.sqrt(var_a / len(values_a) + var_b / len(values_b))

        # T-statistic
        t_stat = (mean_b - mean_a) / se if se > 0 else 0.0

        # Approximate p-value (two-tailed)
        # Using normal approximation for large samples
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))

        # Effect size (Cohen's d)
        pooled_std = math.sqrt((var_a + var_b) / 2)
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

        # Confidence interval
        z_score = 1.96  # 95% confidence
        ci_margin = z_score * se
        ci = (mean_b - mean_a - ci_margin, mean_b - mean_a + ci_margin)

        return StatisticalTest(
            p_value=max(0.0, min(1.0, p_value)),
            is_significant=p_value < self.significance_level,
            confidence_level=1 - self.significance_level,
            effect_size=effect_size,
            confidence_interval=ci,
            test_name="t-test",
        )

    def _normal_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal.

        Args:
            x: Input value

        Returns:
            CDF value
        """
        # Approximation of normal CDF
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class BayesianAnalyzer:
    """Bayesian statistical analyzer."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """Initialize Bayesian analyzer.

        Args:
            prior_alpha: Prior alpha parameter (Beta distribution)
            prior_beta: Prior beta parameter (Beta distribution)
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        LOGGER.info("Initialized BayesianAnalyzer")

    def beta_binomial_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
    ) -> dict[str, Any]:
        """Bayesian test for binomial data (e.g., conversion rates).

        Args:
            successes_a: Successes in group A
            trials_a: Trials in group A
            successes_b: Successes in group B
            trials_b: Trials in group B

        Returns:
            Test results with probability B > A
        """
        # Posterior parameters
        alpha_a = self.prior_alpha + successes_a
        beta_a = self.prior_beta + trials_a - successes_a

        alpha_b = self.prior_alpha + successes_b
        beta_b = self.prior_beta + trials_b - successes_b

        # Monte Carlo sampling for P(B > A)
        # For simplicity, use analytical approximation
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)

        var_a = (alpha_a * beta_a) / ((alpha_a + beta_a) ** 2 * (alpha_a + beta_a + 1))
        var_b = (alpha_b * beta_b) / ((alpha_b + beta_b) ** 2 * (alpha_b + beta_b + 1))

        # Approximate P(B > A) using normal approximation
        diff_mean = mean_b - mean_a
        diff_var = var_a + var_b
        diff_std = math.sqrt(diff_var)

        # P(B > A) = P(B - A > 0)
        if diff_std > 0:
            z = diff_mean / diff_std
            prob_b_wins = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            prob_b_wins = 0.5

        return {
            "prob_b_better": prob_b_wins,
            "prob_a_better": 1 - prob_b_wins,
            "expected_lift": diff_mean,
            "posterior_a": {"alpha": alpha_a, "beta": beta_a, "mean": mean_a},
            "posterior_b": {"alpha": alpha_b, "beta": beta_b, "mean": mean_b},
        }


# Convenience functions
def calculate_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    significance_level: float = 0.05,
    power: float = 0.8,
) -> int:
    """Calculate required sample size per variant.

    Args:
        baseline_rate: Baseline conversion rate
        minimum_detectable_effect: Minimum effect to detect (e.g., 0.05 for 5%)
        significance_level: Alpha (Type I error rate)
        power: Statistical power (1 - beta, where beta is Type II error rate)

    Returns:
        Required sample size per variant
    """
    # Z-scores
    z_alpha = 1.96  # For alpha = 0.05 (two-tailed)
    z_beta = 0.84  # For power = 0.8

    # Expected rates
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Sample size calculation
    numerator = (
        z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2

    denominator = (p2 - p1) ** 2

    return int(math.ceil(numerator / denominator))
