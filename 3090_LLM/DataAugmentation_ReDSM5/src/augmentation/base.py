"""Base utilities for dataset augmentation pipelines."""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.timestamp import utc_timestamp
from src.data.redsm5_loader import (
    DEFAULT_ANNOTATIONS_PATH,
    DEFAULT_POSTS_PATH,
    get_positive_evidence,
    load_annotations,
    load_posts,
)

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    output_dir: Path = Path("Data/Augmentation")
    posts_path: Path = DEFAULT_POSTS_PATH
    annotations_path: Path = DEFAULT_ANNOTATIONS_PATH
    random_seed: int = 13
    variants_per_example: int = 3
    include_original: bool = True


@dataclass
class AugmentedRecord:
    post_id: str
    criterion: str
    post_text: str
    augmented_post: str
    evidence_augmented: str
    label: int = 1
    evidence_original: str | None = None
    source: str = ""
    metadata: dict | None = None


class BaseAugmenter(ABC):
    """Common logic for all augmentation strategies."""

    name: str = "base"

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._posts = load_posts(self.config.posts_path)
        self._annotations = load_annotations(self.config.annotations_path)
        self._positives = get_positive_evidence(self._annotations)

    @abstractmethod
    def _augment_evidence(self, evidence: str, num_variants: int) -> Sequence[str]:
        """Generate augmented variants of evidence text.

        Args:
            evidence: Original evidence text to augment
            num_variants: Number of variants to generate

        Returns:
            Sequence of augmented text variants
        """
        raise NotImplementedError

    # --- public api ------------------------------------------------------------
    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _normalise_for_match(text: str) -> str:
        text = text.lower()
        text = text.replace("’", "'")
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _best_sentence_match(cls, post_text: str, target: str) -> str | None:
        sentences = re.split(r"(?<=[.!?])\s+", post_text)
        if not sentences:
            sentences = [post_text]
        norm_target = cls._normalise_for_match(target)
        best_sentence = None
        best_score = 0.0
        for sentence in sentences:
            norm_sentence = cls._normalise_for_match(sentence)
            if not norm_sentence:
                continue
            score = SequenceMatcher(None, norm_target, norm_sentence).ratio()
            if score > best_score:
                best_score = score
                best_sentence = sentence
        if best_score >= 0.6:
            return best_sentence
        return None

    @classmethod
    def _replace_evidence(cls, post_text: str, original: str, replacement: str) -> str:
        if original in post_text:
            return post_text.replace(original, replacement, 1)
        pattern = re.compile(r"\s+".join(map(re.escape, original.split())), flags=re.MULTILINE)
        match = pattern.search(post_text)
        if match:
            return post_text[: match.start()] + replacement + post_text[match.end() :]
        candidate = cls._best_sentence_match(post_text, original)
        if candidate and candidate in post_text:
            return post_text.replace(candidate, replacement, 1)
        raise ValueError("Evidence sentence not found in post text")

    def generate(self) -> pd.DataFrame:
        records: list[AugmentedRecord] = []
        failed_replacements = 0

        for row in self._positives.itertuples(index=False):
            post_id = row.post_id
            criterion = row.DSM5_symptom
            evidence = row.evidence
            post_text = self._posts.loc[post_id, "text"]
            variants = list(self._augment_evidence(evidence, self.config.variants_per_example))
            if self.config.include_original:
                variants.insert(0, evidence)
            for idx, variant in enumerate(variants):
                try:
                    augmented_post = self._replace_evidence(post_text, evidence, variant)
                except ValueError as e:
                    logger.warning(
                        f"Failed to replace evidence in post {post_id} for {criterion}: {e}. "
                        f"Skipping this variant."
                    )
                    failed_replacements += 1
                    continue

                records.append(
                    AugmentedRecord(
                        post_id=post_id,
                        criterion=criterion,
                        post_text=post_text,
                        augmented_post=augmented_post,
                        evidence_augmented=variant,
                        evidence_original=evidence,
                        source=f"{self.name}:{idx}",
                    )
                )

        if failed_replacements > 0:
            logger.info(f"Skipped {failed_replacements} variants due to evidence replacement failures")

        return pd.DataFrame([record.__dict__ for record in records])

    def save(self, df: pd.DataFrame) -> Path:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_timestamp()
        path = output_dir / f"{self.name}_dataset_{timestamp}.csv"
        df.to_csv(path, index=False)
        return path

    def run(self) -> Path:
        df = self.generate()
        return self.save(df)
