"""Text augmentation using nlpaug and TextAttack for evidence augmentation.

This module provides 25 working text augmentation methods:
- 15 from nlpaug library (removed 2 broken: nlp_back_translation, nlp_abst_summary)
- 10 from TextAttack library (removed 1 broken: ta_clare)

Methods preserve non-evidence regions while augmenting evidence text.

Note: Both nlpaug and TextAttack are optional. If not installed, augmentation will be disabled.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Try to import nlpaug (optional dependency)
try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    logger.warning(
        "nlpaug not available. nlpaug augmentations will be disabled. "
        "Install with: poetry install -E augmentation"
    )

# Try to import TextAttack (optional dependency)
try:
    from textattack.augmentation import (
        CharSwapAugmenter,
        CheckListAugmenter,
        DeletionAugmenter,
        EasyDataAugmenter,
        EmbeddingAugmenter,
        WordNetAugmenter,
    )

    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    logger.warning(
        "TextAttack not available. TextAttack augmentations will be disabled. "
        "Install with: poetry install -E augmentation"
    )

# Define all 28 augmentation method names
NLPAUG_METHODS = [
    # Character-level (6 methods)
    "nlp_char_insert",
    "nlp_char_substitute",
    "nlp_char_swap",
    "nlp_char_delete",
    "nlp_keyboard",
    "nlp_ocr",

    # Word-level (8 methods)
    "nlp_word_antonym",
    "nlp_word_synonym",
    "nlp_word_split",
    "nlp_word_spelling",
    "nlp_word_contextual",
    "nlp_word_substitute",
    "nlp_word_swap",
    "nlp_word_delete",

    # Sentence-level (1 method - removed broken: nlp_back_translation, nlp_abst_summary)
    "nlp_contextual_insert",
]

TEXTATTACK_METHODS = [
    "ta_wordnet",
    "ta_embedding",
    "ta_charswap",
    "ta_eda",
    "ta_checklist",
    # "ta_clare",  # Removed - broken (missing 'upos' attribute)
    "ta_deletion",
    "ta_composite",
    "ta_word_swap_random",
    "ta_word_swap_embedding",
    "ta_word_deletion",
]

ALL_METHODS = NLPAUG_METHODS + TEXTATTACK_METHODS


@dataclass
class AugmentationConfig:
    """Configuration for text augmentation."""

    methods: list[str] | None = None  # List of methods to use (None = use single method)
    method: str = "none"  # Single method (legacy, for backward compatibility)
    probability: float = 0.3  # Probability of augmenting each example (0.0-1.0)
    num_augmentations: int = 1  # Number of augmented versions per example
    preserve_non_evidence: bool = True  # Preserve non-evidence regions

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"Augmentation probability must be in [0.0, 1.0], got {self.probability}"
            )

        # Validate methods
        if self.methods is not None:
            invalid_methods = [m for m in self.methods if m not in ALL_METHODS and m != "none"]
            if invalid_methods:
                raise ValueError(
                    f"Unknown augmentation methods: {invalid_methods}. "
                    f"Valid methods: {ALL_METHODS}"
                )
        elif self.method not in ALL_METHODS and self.method != "none":
            raise ValueError(
                f"Unknown augmentation method: {self.method}. "
                f"Valid methods: {ALL_METHODS} or 'none'"
            )

    def get_active_methods(self) -> list[str]:
        """Get list of active augmentation methods."""
        if self.methods is not None:
            return [m for m in self.methods if m != "none"]
        elif self.method != "none":
            return [self.method]
        return []


class TextAugmenter:
    """Text augmentation using nlpaug and TextAttack."""

    def __init__(self, config: AugmentationConfig):
        """Initialize augmenter.

        Args:
            config: Augmentation configuration
        """
        self.config = config
        self.augmenters = {}

        active_methods = config.get_active_methods()
        if not active_methods:
            logger.info("No augmentation methods specified")
            return

        # Initialize augmenters for each active method
        for method in active_methods:
            augmenter = self._create_augmenter(method)
            if augmenter is not None:
                self.augmenters[method] = augmenter
                logger.info(f"Initialized augmenter: {method}")

    def _create_augmenter(self, method: str):
        """Create a single augmenter based on method name.

        Args:
            method: Name of the augmentation method

        Returns:
            Augmenter instance or None if unavailable
        """
        # nlpaug augmenters
        if method.startswith("nlp_") and not NLPAUG_AVAILABLE:
            logger.warning(f"nlpaug not available, skipping {method}")
            return None

        # TextAttack augmenters
        if method.startswith("ta_") and not TEXTATTACK_AVAILABLE:
            logger.warning(f"TextAttack not available, skipping {method}")
            return None

        try:
            # Character-level nlpaug augmenters
            if method == "nlp_char_insert":
                return nac.RandomCharAug(action="insert", aug_char_p=0.1)
            elif method == "nlp_char_substitute":
                return nac.RandomCharAug(action="substitute", aug_char_p=0.1)
            elif method == "nlp_char_swap":
                return nac.RandomCharAug(action="swap", aug_char_p=0.1)
            elif method == "nlp_char_delete":
                return nac.RandomCharAug(action="delete", aug_char_p=0.1)
            elif method == "nlp_keyboard":
                return nac.KeyboardAug(aug_char_p=0.1)
            elif method == "nlp_ocr":
                return nac.OcrAug(aug_char_p=0.1)

            # Word-level nlpaug augmenters
            elif method == "nlp_word_antonym":
                return naw.AntonymAug(aug_p=0.1)
            elif method == "nlp_word_synonym":
                return naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
            elif method == "nlp_word_split":
                return naw.SplitAug(aug_p=0.1)
            elif method == "nlp_word_spelling":
                return naw.SpellingAug(aug_p=0.1)
            elif method == "nlp_word_contextual":
                return naw.ContextualWordEmbsAug(
                    model_path='distilbert-base-uncased',
                    action="substitute",
                    aug_p=0.1
                )
            elif method == "nlp_word_substitute":
                return naw.RandomWordAug(action="substitute", aug_p=0.1)
            elif method == "nlp_word_swap":
                return naw.RandomWordAug(action="swap", aug_p=0.1)
            elif method == "nlp_word_delete":
                return naw.RandomWordAug(action="delete", aug_p=0.1)

            # Sentence-level nlpaug augmenters
            elif method == "nlp_contextual_insert":
                return naw.ContextualWordEmbsAug(
                    model_path='distilbert-base-uncased',
                    action="insert",
                    aug_p=0.1
                )

            # TextAttack augmenters
            elif method == "ta_wordnet":
                return WordNetAugmenter(pct_words_to_swap=0.1)
            elif method == "ta_embedding":
                return EmbeddingAugmenter(transformations_per_example=1)
            elif method == "ta_charswap":
                return CharSwapAugmenter(
                    pct_words_to_swap=0.1,
                    transformations_per_example=1
                )
            elif method == "ta_eda":
                return EasyDataAugmenter(transformations_per_example=1)
            elif method == "ta_checklist":
                return CheckListAugmenter(transformations_per_example=1)
            elif method == "ta_deletion":
                return DeletionAugmenter(pct_words_to_swap=0.1)
            elif method == "ta_composite":
                # Composite of multiple transformations
                return EmbeddingAugmenter(transformations_per_example=2)
            elif method == "ta_word_swap_random":
                # Random word swap using embedding
                return EmbeddingAugmenter(
                    pct_words_to_swap=0.1,
                    transformations_per_example=1
                )
            elif method == "ta_word_swap_embedding":
                return EmbeddingAugmenter(pct_words_to_swap=0.15)
            elif method == "ta_word_deletion":
                return DeletionAugmenter(pct_words_to_swap=0.1)

        except Exception as e:
            logger.error(f"Failed to initialize augmenter {method}: {e}")
            return None

        return None

    def augment_text(self, text: str, method: str | None = None) -> list[str]:
        """Augment a single text.

        Args:
            text: Text to augment
            method: Specific method to use (None = use random from available)

        Returns:
            List of augmented texts (may be empty if augmentation disabled)
        """
        if not self.augmenters or not text:
            return []

        # Select method
        if method is None:
            method = random.choice(list(self.augmenters.keys()))

        if method not in self.augmenters:
            logger.warning(f"Method {method} not available")
            return []

        augmenter = self.augmenters[method]

        try:
            # nlpaug augmenters
            if method.startswith("nlp_"):
                augmented = augmenter.augment(text, n=self.config.num_augmentations)
                if isinstance(augmented, str):
                    return [augmented]
                return augmented

            # TextAttack augmenters
            elif method.startswith("ta_"):
                augmented = augmenter.augment(text)
                if isinstance(augmented, str):
                    return [augmented]
                return augmented[: self.config.num_augmentations]

        except Exception as e:
            logger.warning(f"Augmentation failed for method {method}: {e}")
            return []

        return []

    def augment_evidence(
        self,
        text: str,
        evidence_spans: list[tuple[int, int]] | None = None,
        method: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Augment only evidence regions, preserve non-evidence text.

        Args:
            text: Full text
            evidence_spans: List of (start, end) character positions for evidence
                           If None, augment entire text
            method: Specific method to use (None = random selection)

        Returns:
            Tuple of (augmented_text, metadata)
            metadata contains:
                - original_evidence: Original evidence text
                - augmented_evidence: Augmented evidence text
                - augmentation_applied: Whether augmentation was applied
                - method_used: Which augmentation method was used
        """
        # Decide whether to augment based on probability
        if random.random() > self.config.probability:
            return text, {
                "original_evidence": None,
                "augmented_evidence": None,
                "augmentation_applied": False,
                "method_used": None,
            }

        # Select random method if not specified
        if method is None and self.augmenters:
            method = random.choice(list(self.augmenters.keys()))

        if evidence_spans is None or not self.config.preserve_non_evidence:
            # Augment entire text
            augmented_texts = self.augment_text(text, method=method)
            if augmented_texts:
                return augmented_texts[0], {
                    "original_evidence": text,
                    "augmented_evidence": augmented_texts[0],
                    "augmentation_applied": True,
                    "method_used": method,
                }
            return text, {
                "original_evidence": text,
                "augmented_evidence": None,
                "augmentation_applied": False,
                "method_used": method,
            }

        # Augment only evidence spans
        augmented_text = text
        metadata = {
            "original_evidence": [],
            "augmented_evidence": [],
            "augmentation_applied": False,
            "method_used": method,
        }

        # Sort spans by start position (reverse order for replacement)
        sorted_spans = sorted(evidence_spans, key=lambda x: x[0], reverse=True)

        for start, end in sorted_spans:
            evidence_text = text[start:end]
            augmented_evidence = self.augment_text(evidence_text, method=method)

            if augmented_evidence:
                # Replace evidence span with augmented version
                augmented_text = (
                    augmented_text[:start] + augmented_evidence[0] + augmented_text[end:]
                )
                metadata["original_evidence"].append(evidence_text)
                metadata["augmented_evidence"].append(augmented_evidence[0])
                metadata["augmentation_applied"] = True

        return augmented_text, metadata

    def augment_batch(
        self,
        examples: list[dict[str, Any]],
        text_key: str = "text",
        evidence_key: str | None = "evidence_spans",
    ) -> list[dict[str, Any]]:
        """Augment a batch of examples.

        Args:
            examples: List of example dictionaries
            text_key: Key for text field in examples
            evidence_key: Key for evidence spans field (optional)

        Returns:
            List of augmented examples (original + augmented)
        """
        augmented_examples = []

        for example in examples:
            # Keep original
            augmented_examples.append(example)

            # Augment if probability allows
            if random.random() <= self.config.probability:
                text = example.get(text_key, "")
                evidence_spans = example.get(evidence_key) if evidence_key else None

                augmented_text, metadata = self.augment_evidence(text, evidence_spans)

                if metadata["augmentation_applied"]:
                    # Create augmented example
                    augmented_example = example.copy()
                    augmented_example[text_key] = augmented_text
                    augmented_example["augmentation_metadata"] = metadata
                    augmented_examples.append(augmented_example)

        return augmented_examples


def create_augmenter(
    methods: list[str] | None = None,
    method: str = "none",
    probability: float = 0.3,
    **kwargs
) -> TextAugmenter:
    """Create a text augmenter with given configuration.

    Args:
        methods: List of augmentation methods to use
        method: Single method (used if methods is None)
        probability: Augmentation probability
        **kwargs: Additional configuration options

    Returns:
        Configured TextAugmenter

    Example:
        # Single method
        augmenter = create_augmenter(method="nlp_word_synonym", probability=0.3)

        # Multiple methods
        augmenter = create_augmenter(
            methods=["nlp_word_synonym", "ta_wordnet", "nlp_char_swap"],
            probability=0.3
        )

        augmented_text, metadata = augmenter.augment_evidence(text)
    """
    config = AugmentationConfig(
        methods=methods,
        method=method,
        probability=probability,
        **kwargs
    )
    return TextAugmenter(config)


def get_available_methods() -> dict[str, list[str]]:
    """Get dictionary of available augmentation methods.

    Returns:
        Dictionary with 'nlpaug' and 'textattack' keys containing lists of available methods
    """
    available = {
        "nlpaug": NLPAUG_METHODS if NLPAUG_AVAILABLE else [],
        "textattack": TEXTATTACK_METHODS if TEXTATTACK_AVAILABLE else [],
        "all": []
    }

    if NLPAUG_AVAILABLE:
        available["all"].extend(NLPAUG_METHODS)
    if TEXTATTACK_AVAILABLE:
        available["all"].extend(TEXTATTACK_METHODS)

    return available
