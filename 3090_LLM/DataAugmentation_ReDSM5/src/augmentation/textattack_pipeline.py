"""Dataset generation using TextAttack augmenters."""
from __future__ import annotations

from typing import Sequence

from .base import AugmentationConfig, BaseAugmenter


class TextAttackAugmenter(BaseAugmenter):
    name = "textattack"

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        super().__init__(config)
        try:
            from textattack.augmentation import EmbeddingAugmenter  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "textattack is required for TextAttackAugmenter. Install it via `pip install textattack`."
            ) from exc
        # Embedding-based augmentation provides lightweight paraphrases suited for evidence sentences.
        self._augmenter = EmbeddingAugmenter(transformations_per_example=3)

    def _augment_evidence(self, evidence: str, num_variants: int) -> Sequence[str]:
        seen = set()
        results: list[str] = []
        attempts = 0
        while len(results) < num_variants and attempts < num_variants * 4:
            augmented_list = self._augmenter.augment(evidence)
            attempts += 1
            if not augmented_list:
                continue
            for candidate in augmented_list:
                candidate = candidate.strip()
                if not candidate or candidate.lower() == evidence.lower():
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                results.append(candidate)
                if len(results) >= num_variants:
                    break
        return results

