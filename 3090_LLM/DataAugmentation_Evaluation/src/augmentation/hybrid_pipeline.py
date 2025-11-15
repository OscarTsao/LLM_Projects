"""Dataset generation combining NLPAug and TextAttack."""
from __future__ import annotations

from typing import Sequence

from .base import AugmentationConfig, BaseAugmenter


class HybridAugmenter(BaseAugmenter):
    name = "hybrid"

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        super().__init__(config)
        try:
            import nlpaug.augmenter.word as naw  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "nlpaug is required for HybridAugmenter. Install it via `pip install nlpaug`."
            ) from exc
        try:
            from textattack.augmentation import EmbeddingAugmenter  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "textattack is required for HybridAugmenter. Install it via `pip install textattack`."
            ) from exc
        self._nlpaug = naw.SynonymAug(aug_src="wordnet", aug_min=1, aug_max=3)
        self._textattack = EmbeddingAugmenter(transformations_per_example=3)

    def _augment_evidence(self, evidence: str, num_variants: int) -> Sequence[str]:
        results: list[str] = []
        seen = {evidence.strip().lower()}

        # Step 1: generate NLPAug variants.
        nlpaug_candidates = self._nlpaug.augment(evidence, n=num_variants)
        if isinstance(nlpaug_candidates, str):
            nlpaug_candidates = [nlpaug_candidates]
        for candidate in nlpaug_candidates or []:
            cand = candidate.strip()
            if cand and cand.lower() not in seen:
                seen.add(cand.lower())
                results.append(cand)
                if len(results) >= num_variants:
                    return results

        # Step 2: generate TextAttack variants.
        textattack_candidates = self._textattack.augment(evidence)
        for candidate in textattack_candidates:
            cand = candidate.strip()
            if cand and cand.lower() not in seen:
                seen.add(cand.lower())
                results.append(cand)
                if len(results) >= num_variants:
                    return results

        # Step 3: cascade TextAttack on NLPAug outputs for additional diversity.
        for seed in list(results):
            cascaded = self._textattack.augment(seed)
            for candidate in cascaded:
                cand = candidate.strip()
                if cand and cand.lower() not in seen:
                    seen.add(cand.lower())
                    results.append(cand)
                    if len(results) >= num_variants:
                        return results
        return results

