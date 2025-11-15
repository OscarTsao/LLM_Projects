"""
Tests for augmentation pipeline behaviour.
"""

import sys
from pathlib import Path
import random

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aug.compose import AugmentationPipeline
from aug.registry import AugmenterRegistry


def test_pipeline_reproducibility():
    """Augmentation should be reproducible for the same example index."""
    registry = AugmenterRegistry()
    pipeline = AugmentationPipeline(
        combo=["punctuation_noise"],
        params={"punctuation_noise": {"aug_p": 0.4}},
        registry=registry,
        seed=21,
    )

    text = "Pipeline reproducibility!"
    first = pipeline.augment_text(text, example_idx=5)
    random.seed(999)
    second = pipeline.augment_text(text, example_idx=5)

    assert first == second


def test_pipeline_with_textattack_adapter():
    """TextAttack augmenters should integrate with the pipeline."""
    pytest.importorskip("textattack")
    registry = AugmenterRegistry()
    pipeline = AugmentationPipeline(combo=["add_typos"], registry=registry, seed=13)

    text = "Typo candidate"
    augmented = pipeline.augment_text(text, example_idx=0)

    assert isinstance(augmented, str)
    assert augmented
