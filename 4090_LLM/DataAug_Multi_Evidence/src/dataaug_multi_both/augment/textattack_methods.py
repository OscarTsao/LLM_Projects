"""Factory for creating textattack-based augmenters.

This module provides 11 text augmentation methods from the textattack library.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

from textattack.augmentation import EmbeddingAugmenter, CharSwapAugmenter, WordNetAugmenter
from textattack.shared import utils, WordEmbedding

try:  # pragma: no cover - optional dependency in unit tests
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    import nltk  # type: ignore
except Exception:  # pragma: no cover
    nltk = None

logger = logging.getLogger(__name__)

_EMBEDDING_CACHE: WordEmbedding | None = None


class TextAttackFactory:
    """Factory for creating textattack augmenters with lazy initialization."""

    # Define the supported textattack methods
    TEXTATTACK_METHODS = [
        "TextFoolerJin2019",
        "PWWSRen2019",
        "BAEGarg2019",
        "DeepWordBugGao2018",
        "HotFlipEbrahimi2017",
        "IGAWang2019",
        "Kuleshov2017",
        "Alzantot2018",
        "BERTAttack",
    ]

    def __init__(self, pct_words_to_swap: float = 0.1, transformations_per_example: int = 1):
        """Initialize the factory.

        Args:
            pct_words_to_swap: Percentage of words to modify
            transformations_per_example: Number of augmented examples to generate
        """
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        self._augmenters: dict[str, Any] = {}
        try:
            flair_logger = logging.getLogger("flair")
            flair_logger.setLevel(logging.ERROR)
            flair_logger.propagate = False
        except Exception:  # pragma: no cover - flair may not be installed
            pass
        if nltk is not None:
            original_download = getattr(nltk, "download", None)
            if original_download is not None and not hasattr(nltk, "_datadaug_download_wrapped"):
                def _quiet_download(*args: Any, **kwargs: Any) -> Any:
                    kwargs.setdefault("quiet", True)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        return original_download(*args, **kwargs)

                nltk.download = _quiet_download  # type: ignore[assignment]
                nltk._datadaug_download_wrapped = True  # type: ignore[attr-defined]
        if torch is not None:
            try:
                if utils.device.type != "cpu":
                    utils.device = torch.device("cpu")  # type: ignore[assignment]
            except AttributeError:  # pragma: no cover - defensive
                utils.device = "cpu"

    def _get_augmenter(self, method: str) -> Any:
        """Get or create an augmenter for the given method (lazy initialization)."""
        if method not in self._augmenters:
            try:
                self._augmenters[method] = self._create_augmenter(method)
            except Exception as exc:
                logger.warning(
                    "textattack %s unavailable (%s); disabling this augmentation method.",
                    method,
                    exc,
                )
                self._augmenters[method] = None
        augmenter = self._augmenters.get(method)
        if augmenter is None:
            raise RuntimeError(f"textattack augmenter '{method}' is unavailable")
        return augmenter

    def _create_augmenter(self, method: str) -> Any:
        """Create a textattack augmenter for the given method."""
        pct = min(1.0, max(0.0, self.pct_words_to_swap))
        tpe = max(1, int(self.transformations_per_example))

        def embedding_augmenter(**kwargs: Any) -> EmbeddingAugmenter:
            global _EMBEDDING_CACHE
            if _EMBEDDING_CACHE is None:
                # Load the heavyweight embedding once per process and reuse it for all augmenters.
                _EMBEDDING_CACHE = WordEmbedding.counterfitted_GLOVE_embedding()

            # Filter out unsupported parameters (fast_augment is not supported in some versions)
            # Try with embedding parameter first (newer versions support it)
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != "fast_augment"}

            try:
                return EmbeddingAugmenter(
                    pct_words_to_swap=pct,
                    transformations_per_example=tpe,
                    embedding=_EMBEDDING_CACHE,
                    **kwargs_filtered,
                )
            except TypeError as e:
                # Remove embedding parameter if not supported (older textattack versions)
                if "embedding" in str(e):
                    return EmbeddingAugmenter(
                        pct_words_to_swap=pct,
                        transformations_per_example=tpe,
                        **kwargs_filtered,
                    )
                else:
                    raise

        def wordnet_augmenter(**kwargs: Any) -> WordNetAugmenter:
            return WordNetAugmenter(
                pct_words_to_swap=pct,
                transformations_per_example=tpe,
                **kwargs,
            )

        def charswap_augmenter(**kwargs: Any) -> CharSwapAugmenter:
            return CharSwapAugmenter(
                pct_words_to_swap=pct,
                transformations_per_example=tpe,
                **kwargs,
            )

        factory_map = {
            "TextFoolerJin2019": lambda: embedding_augmenter(),
            "PWWSRen2019": lambda: wordnet_augmenter(),
            "BAEGarg2019": lambda: embedding_augmenter(high_yield=True),
            "DeepWordBugGao2018": lambda: charswap_augmenter(),
            "HotFlipEbrahimi2017": lambda: charswap_augmenter(high_yield=True),
            "IGAWang2019": lambda: embedding_augmenter(high_yield=True, fast_augment=True),
            "Kuleshov2017": lambda: wordnet_augmenter(high_yield=True),
            "Alzantot2018": lambda: embedding_augmenter(high_yield=True),
            "BERTAttack": lambda: embedding_augmenter(high_yield=True, fast_augment=True),
        }

        factory = factory_map.get(method)
        if not factory:
            raise ValueError(f"Unknown textattack method: {method}")

        augmenter = factory()
        # Ensure TextAttack operates purely on CPU to avoid CUDA issues in workers.
        if hasattr(augmenter, "device"):
            try:
                augmenter.device = "cpu"  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass
        return augmenter

    def augment(self, text: str, method: str) -> str:
        """Augment text using the specified textattack method.

        Args:
            text: Input text to augment
            method: Name of the augmentation method

        Returns:
            Augmented text
        """
        try:
            augmenter = self._get_augmenter(method)
        except RuntimeError:
            return text

        try:
            augmented = augmenter.augment(text)
            # textattack returns a list, take the first result
            if isinstance(augmented, list):
                return augmented[0] if augmented else text
            return augmented
        except Exception as e:
            # If augmentation fails, return original text
            logger.warning("textattack %s failed: %s", method, e)
            return text


def create_textattack_augmenter(
    pct_words_to_swap: float = 0.1,
    transformations_per_example: int = 1,
) -> TextAttackFactory:
    """Create a textattack factory instance.

    Args:
        pct_words_to_swap: Percentage of words to modify
        transformations_per_example: Number of augmented examples to generate

    Returns:
        TextAttackFactory instance
    """
    return TextAttackFactory(
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )


__all__ = ["TextAttackFactory", "create_textattack_augmenter"]
