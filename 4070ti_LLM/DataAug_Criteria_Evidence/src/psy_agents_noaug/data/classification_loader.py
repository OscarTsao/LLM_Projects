"""Helpers to construct classification DataLoaders with optional augmentation.

This module wires together:
  - a tokenised dataset (lazy/eager)
  - a collate function that can augment raw texts pre-tokenisation
  - DataLoader worker seeding so augmentation remains deterministic with workers
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline, worker_init
from psy_agents_noaug.data.datasets import (
    ClassificationDataset,
    create_classification_collate,
)

from .augmentation_utils import AugmentationArtifacts, build_evidence_augmenter

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ClassificationLoaders:
    """Bundle containing DataLoaders and augmentation context."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    augmentation: AugmentationArtifacts | None

    @property
    def augmenter_pipeline(self) -> AugmenterPipeline | None:
        return self.augmentation.pipeline if self.augmentation else None


def _as_aug_config(config: AugConfig | dict[str, Any] | None) -> AugConfig | None:
    """Accept either an AugConfig instance or a plain dict for convenience."""
    if config is None or isinstance(config, AugConfig):
        return config
    cfg_dict = dict(config)
    if "max_replace" not in cfg_dict and "max_replace_ratio" in cfg_dict:
        cfg_dict["max_replace"] = cfg_dict.pop("max_replace_ratio")
    if "enabled" not in cfg_dict:
        cfg_dict["enabled"] = bool(cfg_dict.get("methods")) or cfg_dict.get(
            "lib"
        ) not in (
            None,
            "none",
        )
    return AugConfig(**cfg_dict)


def build_evidence_classification_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    augment_config: AugConfig | dict[str, Any] | None = None,
    seed: int = 42,
    num_workers: int | None = None,
    prefetch_factor: int | None = 4,
    pin_memory: bool = True,
    persistent_workers: bool | None = True,
    text_column: str = "criterion_text",  # First sequence (criterion)
    text_pair_column: str | None = "input_text",  # Second sequence (post/input)
    label_column: str = "label",
    tfidf_dir: str | Path = "_artifacts/tfidf/evidence",
) -> ClassificationLoaders:
    """Create train/val/test DataLoaders for evidence classification.

    Input format: [CLS] criterion_text [SEP] input_text [SEP]
    This matches the order used in span extraction tasks for consistency.

    When augmentation is enabled, the train dataset defers tokenisation to the
    collate function (``lazy_encode=True``) so transformations operate on raw
    strings before tokenisation.
    """

    aug_cfg = _as_aug_config(augment_config)
    augmentation: AugmentationArtifacts | None = None

    if aug_cfg is not None:
        train_texts = train_df[text_column].astype(str).tolist()
        augmentation = build_evidence_augmenter(
            aug_cfg,
            train_texts,
            tfidf_dir=tfidf_dir,
        )

    has_text_pair = text_pair_column is not None

    train_dataset = ClassificationDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column,
        text_pair_column=text_pair_column,
        label_column=label_column,
        label_dtype="int",
        augmenter=augmentation.pipeline if augmentation else None,
        lazy_encode=augmentation is not None,
    )

    val_dataset = ClassificationDataset(
        val_df,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column,
        text_pair_column=text_pair_column,
        label_column=label_column,
        label_dtype="int",
    )

    test_dataset = ClassificationDataset(
        test_df,
        tokenizer=tokenizer,
        max_length=max_length,
        text_column=text_column,
        text_pair_column=text_pair_column,
        label_column=label_column,
        label_dtype="int",
    )

    train_collate = create_classification_collate(
        tokenizer,
        max_length,
        has_text_pair=has_text_pair,
        augmenter=augmentation.pipeline if augmentation else None,
    )
    eval_collate = create_classification_collate(
        tokenizer,
        max_length,
        has_text_pair=has_text_pair,
    )

    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = max(2, cpu_count // 2)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    effective_prefetch = (
        None if num_workers == 0 else prefetch_factor if prefetch_factor else 2
    )

    def _worker_init_fn(worker_id: int) -> None:
        """Derive a unique seed per worker and propagate it to the pipeline."""
        derived_seed = worker_init(worker_id, seed)
        if augmentation:
            augmentation.pipeline.set_seed(derived_seed)

    common_loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
    if effective_prefetch and num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = effective_prefetch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        **common_loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        **common_loader_kwargs,
    )

    return ClassificationLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        augmentation=augmentation,
    )


__all__ = ["ClassificationLoaders", "build_evidence_classification_loaders"]
