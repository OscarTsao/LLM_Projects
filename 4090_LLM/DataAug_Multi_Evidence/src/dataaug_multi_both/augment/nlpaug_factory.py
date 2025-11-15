"""Factory for creating nlpaug augmenters.

This module provides 17 text augmentation methods from the nlpaug library.
"""

from __future__ import annotations

import contextlib
import io
import logging
import random
from typing import Any

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

try:  # Lightweight import guard for environments without NLTK preinstalled.
    import nltk
    from nltk import data as nltk_data
except Exception:  # pragma: no cover - NLTK is an optional dependency in tests.
    nltk = None
    nltk_data = None

logger = logging.getLogger(__name__)


class NLPAugFactory:
    """Factory for creating nlpaug augmenters with lazy initialization."""

    # Define the 17 nlpaug methods
    NLPAUG_METHODS = [
        "ContextualWordEmbedding",
        "Synonym",
        "Antonym",
        "RandomWord",
        "Spelling",
        "Keyboard",
        "Ocr",
        "BackTranslation",
        "TfIdf",
        "Split",
        "Reserved",
        "AbstSumm",
        "RandomChar",
        "WordEmbedding",
        "ContextualSentence",
        "Lambada",
        "CharSwap",
    ]

    def __init__(self, aug_prob: float = 0.1, seed: int | None = None):
        """Initialize the factory.

        Args:
            aug_prob: Probability of augmenting a token/word
            seed: Random seed for reproducibility
        """
        self.aug_prob = aug_prob
        self.seed = seed
        self._augmenters: dict[str, Any] = {}
        self._nltk_checked = False
        self._nltk_available = True
        if nltk is not None:
            original_download = getattr(nltk, "download", None)
            if original_download is not None and not hasattr(nltk, "_datadaug_download_wrapped"):
                def _quiet_download(*args: Any, **kwargs: Any) -> Any:
                    kwargs.setdefault("quiet", True)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        return original_download(*args, **kwargs)

                nltk.download = _quiet_download  # type: ignore[assignment]
                nltk._datadaug_download_wrapped = True  # type: ignore[attr-defined]

    def _ensure_nltk_resources(self) -> None:
        if nltk is None or nltk_data is None or self._nltk_checked:
            return self._nltk_available
        resources = {
            "corpora/wordnet": "wordnet",
            "corpora/omw-1.4": "omw-1.4",
            "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
            "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        }
        missing: list[str] = []
        for locator, name in resources.items():
            try:
                nltk_data.find(locator)
            except LookupError:
                missing.append(name)
        if missing:
            self._nltk_available = False
            logger.warning(
                "Missing NLTK resources (%s). nlpaug synonym/antonym augmenters will pass through unchanged.",
                ", ".join(sorted(set(missing))),
            )
        self._nltk_checked = True
        return self._nltk_available

    def _get_augmenter(self, method: str) -> Any:
        """Get or create an augmenter for the given method (lazy initialization)."""
        if method not in self._augmenters:
            self._augmenters[method] = self._create_augmenter(method)
        return self._augmenters[method]

    def _create_augmenter(self, method: str) -> Any:
        """Create an nlpaug augmenter for the given method."""
        aug_p = self.aug_prob

        if method in {"Synonym", "Antonym"}:
            self._ensure_nltk_resources()

        if method == "ContextualWordEmbedding":
            # BERT-based contextual word replacement
            return naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="substitute",
                aug_p=aug_p,
            )

        elif method == "Synonym":
            # WordNet synonym replacement
            if not self._ensure_nltk_resources():
                raise RuntimeError("NLTK resources not available for Synonym augmentation")
            return naw.SynonymAug(aug_src="wordnet", aug_p=aug_p)

        elif method == "Antonym":
            # WordNet antonym replacement
            if not self._ensure_nltk_resources():
                raise RuntimeError("NLTK resources not available for Antonym augmentation")
            return naw.AntonymAug(aug_p=aug_p)

        elif method == "RandomWord":
            # Random word operations (insert/substitute/swap/delete)
            return naw.RandomWordAug(
                action="substitute",
                aug_p=aug_p,
            )

        elif method == "Spelling":
            # Introduce spelling errors
            return naw.SpellingAug(aug_p=aug_p)

        elif method == "Keyboard":
            # Keyboard typo simulation
            return nac.KeyboardAug(aug_char_p=aug_p)

        elif method == "Ocr":
            # OCR error simulation
            return nac.OcrAug(aug_char_p=aug_p)

        elif method == "BackTranslation":
            # Back-translation (English -> German -> English)
            # Note: This requires internet connection
            return naw.BackTranslationAug(
                from_model_name="facebook/wmt19-en-de",
                to_model_name="facebook/wmt19-de-en",
                device="cpu",
            )

        elif method == "TfIdf":
            # TF-IDF based word replacement
            return naw.TfIdfAug(action="substitute", aug_p=aug_p)

        elif method == "Split":
            # Word splitting augmentation
            return naw.SplitAug(aug_p=aug_p)

        elif method == "RandomChar":
            # Random character operations (insert/substitute/swap/delete)
            return nac.RandomCharAug(
                action="substitute",
                aug_char_p=aug_p,
            )

        elif method == "WordEmbedding":
            # Word2Vec/GloVe/FastText word replacement
            return naw.WordEmbsAug(
                model_type="word2vec",
                action="substitute",
                aug_p=aug_p,
            )

        elif method == "ContextualSentence":
            # Contextual sentence-level augmentation
            return naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased",
                action="insert",
                aug_p=aug_p,
            )

        elif method == "Lambada":
            # Lambada-based augmentation (masked language model)
            return naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased",
                action="substitute",
                top_k=100,
                aug_p=aug_p,
            )

        elif method == "CharSwap":
            # Character swap augmentation
            return nac.RandomCharAug(
                action="swap",
                aug_char_p=aug_p,
            )

        else:
            raise ValueError(f"Unknown nlpaug method: {method}")

    def augment(self, text: str, method: str) -> str:
        """Augment text using the specified nlpaug method.

        Args:
            text: Input text to augment
            method: Name of the augmentation method

        Returns:
            Augmented text
        """
        try:
            augmenter = self._get_augmenter(method)
            augmented = augmenter.augment(text)
            # nlpaug returns a list, take the first result
            if isinstance(augmented, list):
                return augmented[0] if augmented else text
            return augmented
        except Exception as e:
            # If augmentation fails, return original text
            logger.warning("nlpaug %s failed: %s", method, e)
            return text


def create_nlpaug_augmenter(aug_prob: float = 0.1, seed: int | None = None) -> NLPAugFactory:
    """Create an nlpaug factory instance.

    Args:
        aug_prob: Probability of augmenting a token/word
        seed: Random seed for reproducibility

    Returns:
        NLPAugFactory instance
    """
    return NLPAugFactory(aug_prob=aug_prob, seed=seed)


__all__ = ["NLPAugFactory", "create_nlpaug_augmenter"]
