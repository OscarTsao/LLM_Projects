"""Augmentation exports."""

from dataaug_multi_both.augment.nlpaug_factory import (
    NLPAugFactory,
    create_nlpaug_augmenter,
)
from dataaug_multi_both.augment.textattack_methods import (
    TextAttackFactory,
    create_textattack_augmenter,
)
from dataaug_multi_both.augment.unified_augmenter import (
    ALL_AUG_METHODS,
    AugmentedDataset,
    UnifiedAugmenter,
    create_augmenter,
)

# Legacy imports for backward compatibility
from dataaug_multi_both.augment.textattack_factory import EvidenceAugmenter

__all__ = [
    "create_augmenter",
    "AugmentedDataset",
    "UnifiedAugmenter",
    "NLPAugFactory",
    "TextAttackFactory",
    "create_nlpaug_augmenter",
    "create_textattack_augmenter",
    "ALL_AUG_METHODS",
    "EvidenceAugmenter",  # Legacy
]

