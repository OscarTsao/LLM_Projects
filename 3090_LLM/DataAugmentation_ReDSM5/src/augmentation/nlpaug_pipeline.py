"""Dataset generation using NLPAug augmenters."""
from __future__ import annotations

from typing import Sequence

from .base import AugmentationConfig, BaseAugmenter


class NLPAugAugmenter(BaseAugmenter):
    name = "nlpaug"

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        super().__init__(config)
        try:
            import nlpaug.augmenter.word as naw  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "nlpaug is required for NLPAugAugmenter. Install it via `pip install nlpaug`."
            ) from exc
        # Use synonym augmentation as default; opt for GPU contextual if available when subclassing.
        self._augmenter = naw.SynonymAug(aug_src="wordnet", aug_min=1, aug_max=3)

    def _augment_evidence(self, evidence: str, num_variants: int) -> Sequence[str]:
        seen = set()
        results: list[str] = []
        attempts = 0
        while len(results) < num_variants and attempts < num_variants * 4:
            augmented = self._augmenter.augment(evidence)
            if not augmented:
                attempts += 1
                continue
            candidate = augmented if isinstance(augmented, str) else augmented[0]
            candidate = candidate.strip()
            if not candidate or candidate.lower() == evidence.lower():
                attempts += 1
                continue
            if candidate in seen:
                attempts += 1
                continue
            seen.add(candidate)
            results.append(candidate)
        return results

