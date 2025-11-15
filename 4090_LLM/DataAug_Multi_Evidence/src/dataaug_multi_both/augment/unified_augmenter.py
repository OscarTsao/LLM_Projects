"""Unified augmentation interface supporting both nlpaug and textattack methods."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from dataaug_multi_both.augment.nlpaug_factory import NLPAugFactory
from dataaug_multi_both.augment.textattack_methods import TextAttackFactory


# All 28 augmentation methods (17 nlpaug + 11 textattack)
ALL_AUG_METHODS = [
    # 17 nlpaug methods
    "nlp_ContextualWordEmbedding",
    "nlp_Synonym",
    "nlp_Antonym",
    "nlp_RandomWord",
    "nlp_Spelling",
    "nlp_Keyboard",
    "nlp_Ocr",
    "nlp_BackTranslation",
    "nlp_TfIdf",
    "nlp_Split",
    "nlp_Reserved",
    "nlp_AbstSumm",
    "nlp_RandomChar",
    "nlp_WordEmbedding",
    "nlp_ContextualSentence",
    "nlp_Lambada",
    "nlp_CharSwap",
    # 11 textattack methods
    "ta_TextFoolerJin2019",
    "ta_PWWSRen2019",
    "ta_BAEGarg2019",
    "ta_DeepWordBugGao2018",
    "ta_HotFlipEbrahimi2017",
    "ta_IGAWang2019",
    "ta_Kuleshov2017",
    "ta_CheckList2020",
    "ta_Alzantot2018",
    "ta_CLARE",
    "ta_BERTAttack",
]


@dataclass
class AugConfig:
    """Configuration for a single augmentation method."""

    name: str
    prob: float


class UnifiedAugmenter:
    """Unified augmenter that supports both nlpaug and textattack methods."""

    def __init__(
        self,
        aug_methods: Sequence[str],
        aug_prob: float = 0.1,
        compose_mode: str = "sequential",
        seed: int | None = None,
    ):
        """Initialize the unified augmenter.

        Args:
            aug_methods: List of augmentation method names (from ALL_AUG_METHODS)
            aug_prob: Probability of applying each augmentation
            compose_mode: How to compose multiple augmentations ("sequential" or "random_one")
            seed: Random seed for reproducibility
        """
        self.aug_methods = list(aug_methods)
        self.aug_prob = aug_prob
        self.compose_mode = compose_mode
        self.seed = seed
        self.rng = random.Random(seed)

        # Lazy initialization of factories
        self._nlpaug_factory: NLPAugFactory | None = None
        self._textattack_factory: TextAttackFactory | None = None

    @property
    def nlpaug_factory(self) -> NLPAugFactory:
        """Get or create nlpaug factory (lazy initialization)."""
        if self._nlpaug_factory is None:
            self._nlpaug_factory = NLPAugFactory(aug_prob=self.aug_prob, seed=self.seed)
        return self._nlpaug_factory

    @property
    def textattack_factory(self) -> TextAttackFactory:
        """Get or create textattack factory (lazy initialization)."""
        if self._textattack_factory is None:
            self._textattack_factory = TextAttackFactory(
                pct_words_to_swap=self.aug_prob,
                transformations_per_example=1,
            )
        return self._textattack_factory

    def _apply_single_augmentation(self, text: str, method: str) -> str:
        """Apply a single augmentation method to the text.

        Args:
            text: Input text
            method: Augmentation method name

        Returns:
            Augmented text
        """
        if method.startswith("nlp_"):
            # nlpaug method
            nlp_method = method[4:]  # Remove "nlp_" prefix
            return self.nlpaug_factory.augment(text, nlp_method)
        elif method.startswith("ta_"):
            # textattack method
            ta_method = method[3:]  # Remove "ta_" prefix
            return self.textattack_factory.augment(text, ta_method)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    def __call__(self, text: str) -> str:
        """Apply augmentation(s) to the input text.

        Args:
            text: Input text to augment

        Returns:
            Augmented text
        """
        if not self.aug_methods:
            return text

        if self.compose_mode == "random_one":
            # Apply only one randomly selected augmentation
            method = self.rng.choice(self.aug_methods)
            return self._apply_single_augmentation(text, method)
        else:  # sequential
            # Apply all augmentations sequentially
            augmented = text
            for method in self.aug_methods:
                augmented = self._apply_single_augmentation(augmented, method)
            return augmented


class AugmentedDataset:
    """Dataset wrapper that applies augmentation on-the-fly."""

    def __init__(
        self,
        dataset: Any,
        augmenter: UnifiedAugmenter,
        field: str = "sentence_text",
    ):
        """Initialize the augmented dataset.

        Args:
            dataset: Base dataset to augment
            augmenter: UnifiedAugmenter instance
            field: Field name to augment
        """
        self.dataset = dataset
        self.augmenter = augmenter
        self.field = field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, dict) and self.field in sample:
            sample = dict(sample)
            sample[self.field] = self.augmenter(sample[self.field])
        return sample


def create_augmenter(
    params: dict[str, Any],
    rng: random.Random,
) -> UnifiedAugmenter | None:
    """Create a UnifiedAugmenter from HPO parameters.

    Args:
        params: HPO parameters dictionary
        rng: Random number generator

    Returns:
        UnifiedAugmenter instance or None if no augmentations selected
    """
    # Get the number of augmentations to use
    num_augs = params.get("num_augmentations", 0)
    if num_augs <= 0:
        return None

    # Get which specific augmentation methods to use
    selected_methods = []
    for i in range(num_augs):
        method_key = f"aug_method_{i}"
        if method_key in params:
            selected_methods.append(params[method_key])

    if not selected_methods:
        return None

    # Get augmentation probability and composition mode
    aug_prob = params.get("aug_prob", 0.1)
    compose_mode = params.get("aug_compose_mode", "sequential")

    # Get seed from rng
    seed = rng.randint(0, 2**31 - 1)

    return UnifiedAugmenter(
        aug_methods=selected_methods,
        aug_prob=aug_prob,
        compose_mode=compose_mode,
        seed=seed,
    )


__all__ = [
    "UnifiedAugmenter",
    "AugmentedDataset",
    "create_augmenter",
    "ALL_AUG_METHODS",
]
