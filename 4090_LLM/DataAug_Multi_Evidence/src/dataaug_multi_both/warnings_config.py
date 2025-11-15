"""
Central warnings configuration for the dataaug_multi_both package.

This module configures Python's warning filters to suppress known benign warnings
from dependencies that would otherwise clutter the console output.
"""

import warnings


def configure_warnings() -> None:
    """
    Configure warning filters to suppress known benign warnings.

    This function should be called early in the application startup to suppress:
    - pkg_resources deprecation warnings from jieba
    - Optuna experimental feature warnings for multivariate and group samplers
    - Transformers sentencepiece tokenizer warnings
    - Accelerate layer sharding informational messages
    """
    # Suppress pkg_resources deprecation warning from jieba
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
    )

    # Suppress Optuna experimental warnings for multivariate and group options
    warnings.filterwarnings(
        "ignore",
        message=".*multivariate.*option is an experimental feature",
        category=optuna.exceptions.ExperimentalWarning,  # type: ignore[name-defined]
    )
    warnings.filterwarnings(
        "ignore",
        message=".*group.*option is an experimental feature",
        category=optuna.exceptions.ExperimentalWarning,  # type: ignore[name-defined]
    )

    # Suppress Transformers sentencepiece tokenizer warning
    warnings.filterwarnings(
        "ignore",
        message=".*sentencepiece tokenizer.*byte fallback option",
        category=UserWarning,
    )

    # Note: "The following layers were not sharded" messages from accelerate/transformers
    # are printed directly to stderr and cannot be easily suppressed. These are informational
    # messages that appear when loading models in non-distributed settings and can be safely ignored.


# Try to import optuna to register its warning types
try:
    import optuna.exceptions
except ImportError:
    pass

# Configure warnings immediately upon import
configure_warnings()
