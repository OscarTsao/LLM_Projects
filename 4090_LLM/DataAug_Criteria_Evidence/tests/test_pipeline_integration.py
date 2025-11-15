"""Integration tests for augmentation-aware classification dataloaders."""

import pandas as pd
import torch

from psy_agents_noaug.augmentation.pipeline import AugConfig
from psy_agents_noaug.data.classification_loader import (
    ClassificationLoaders,
    build_evidence_classification_loaders,
)


class DummyTokenizer:
    """Minimal tokenizer returning padded tensors for testing."""

    def __call__(
        self,
        texts,
        text_pair=None,
        *,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        input_ids = (
            torch.arange(batch_size * max_length).reshape(batch_size, max_length) % 100
        )
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_text": [
                "Patient reports persistent sadness.",
                "Patient denies hallucinations or delusions.",
                "Sleep quality has deteriorated recently.",
            ],
            "criterion_text": [
                "Major depressive episode",
                "Psychotic symptoms",
                "Sleep disturbance",
            ],
            "label": [1, 0, 1],
        }
    )


def test_build_classification_loaders_with_augmentation():
    tokenizer = DummyTokenizer()
    train_df = _make_dataframe()
    val_df = _make_dataframe()
    test_df = _make_dataframe()

    aug_cfg = AugConfig(
        lib="nlpaug",
        methods=["nlpaug/char/RandomCharAug"],
        p_apply=1.0,
        ops_per_sample=1,
        max_replace_ratio=0.2,
        seed=123,
    )

    loaders: ClassificationLoaders = build_evidence_classification_loaders(
        train_df,
        val_df,
        test_df,
        tokenizer=tokenizer,
        max_length=16,
        batch_size=2,
        eval_batch_size=2,
        augment_config=aug_cfg,
        seed=777,
        num_workers=0,
        pin_memory=False,
    )

    assert isinstance(loaders.train, torch.utils.data.DataLoader)
    assert loaders.augmentation is not None
    assert "nlpaug/char/RandomCharAug" in loaders.augmentation.methods

    batch = next(iter(loaders.train))
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == 2

    stats = loaders.augmentation.pipeline.stats()
    assert stats["total"] >= 2
    assert stats["applied"] >= 1

    loaders_second = build_evidence_classification_loaders(
        train_df,
        val_df,
        test_df,
        tokenizer=tokenizer,
        max_length=16,
        batch_size=2,
        eval_batch_size=2,
        augment_config=aug_cfg,
        seed=777,
        num_workers=0,
        pin_memory=False,
    )
    batch_second = next(iter(loaders_second.train))
    torch.testing.assert_close(batch["input_ids"], batch_second["input_ids"])
