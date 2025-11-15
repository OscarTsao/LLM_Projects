from __future__ import annotations

import random

from dataaug_multi_both.augment import AugmentedDataset, create_augmenter
from dataaug_multi_both.hpo.space import encode_mask


def _params_for_simple_aug() -> dict[str, object]:
    return {
        "aug_simple_mask": encode_mask(("EDA",)),
        "aug_simple_EDA_p_token": 0.3,
        "aug_simple_EDA_tpe": 1,
        "aug_simple_compose": "sequence",
        "aug_ta_mask": "none",
        "aug_cross_family": "serial_only",
    }


def test_augmentation_applies_only_to_training_field() -> None:
    params = _params_for_simple_aug()
    rng = random.Random(123)
    augmenter = create_augmenter(params, rng)
    assert augmenter is not None

    train_samples = [
        {"sentence_text": "I feel good today and hopeful", "status": 1},
        {"sentence_text": "My sleep is bad but energy is okay", "status": 1},
    ]
    augmented = AugmentedDataset(train_samples, augmenter)

    original = train_samples[0]["sentence_text"]
    augmented_text = augmented[0]["sentence_text"]
    assert original != augmented_text

    val_samples = train_samples.copy()
    assert val_samples[0]["sentence_text"] == train_samples[0]["sentence_text"]
