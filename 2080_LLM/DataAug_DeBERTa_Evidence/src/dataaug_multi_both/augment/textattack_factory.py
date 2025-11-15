from __future__ import annotations

import inspect
import json
import logging
import random
from pathlib import Path
from typing import Callable, Iterable, Mapping, Tuple

import numpy as np
try:
    from textattack import augmentation as ta_augmentation
    from textattack.augmentation import Augmenter
    from textattack.augmentation import recipes as ta_recipes
    from textattack.transformations.word_swaps.word_swap_random_character_substitute import (
        WordSwapRandomCharacterSubstitute,
    )
except ImportError:  # pragma: no cover - optional dependency
    ta_augmentation = None
    ta_recipes = None

    class Augmenter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def augment(self, text: str, **kwargs) -> list[str]:
            return [text]

    class WordSwapRandomCharacterSubstitute:  # pragma: no cover - fallback
        pass

from dataaug_multi_both.mlflow_buffer import log_artifact_safe

logger = logging.getLogger(__name__)

MAX_SIMPLE_ACTIVE = 2
MAX_ATTACK_ACTIVE = 1
MAX_SEQUENCE_STEPS = 2

_SIMPLE_ALIASES = {
    "EDA": "EasyDataAugmenter",
    "CharSwap": "CharSwapAugmenter",
    "Embedding": "EmbeddingAugmenter",
    "BackTranslation": "BackTranslationAugmenter",
    "CheckList": "CheckListAugmenter",
    "CLARE": "CLAREAugmenter",
}


def _candidate_names(name: str) -> list[str]:
    alias = _SIMPLE_ALIASES.get(name, name)
    candidates = [
        alias,
        f"{alias}Augmenter",
        alias.replace("Recipe", "") + "Augmenter",
        name,
    ]
    # Preserve order but remove duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def _resolve_augmenter_class(name: str) -> type[Augmenter] | None:
    if ta_recipes is None:
        return None
    for candidate in _candidate_names(name):
        cls = getattr(ta_recipes, candidate, None)
        if cls:
            return cls
    if ta_augmentation is None:
        return None
    for candidate in _candidate_names(name):
        cls = getattr(ta_augmentation, candidate, None)
        if cls:
            return cls
    return None


def _fallback_augmenter(seed: int) -> Augmenter:
    class _Fallback(Augmenter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._swap = WordSwapRandomCharacterSubstitute()

        def augment(self, text: str, **kwargs) -> list[str]:
            if not text:
                return []
            tokens = list(text)
            if not tokens:
                return []
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = tokens[idx].upper()
            return ["".join(tokens)]

    return _Fallback(transformation_prob=0.1, transformations_per_example=1, seed=seed)


def _instantiate_augmenter(
    recipe_name: str,
    *,
    seed: int,
    transformations_per_example: int,
    probability: float,
) -> Augmenter:
    cls = _resolve_augmenter_class(recipe_name)
    if cls is None:
        logger.warning("Falling back to simple augmenter for recipe=%s", recipe_name)
        return _fallback_augmenter(seed)

    kwargs: dict[str, object] = {}
    signature = inspect.signature(cls.__init__)
    if "transformations_per_example" in signature.parameters:
        kwargs["transformations_per_example"] = max(0, transformations_per_example)
    if "pct_words_to_swap" in signature.parameters:
        kwargs["pct_words_to_swap"] = probability
    if "pct_change" in signature.parameters:
        kwargs["pct_change"] = probability
    if "probability" in signature.parameters:
        kwargs["probability"] = probability
    if "p" in signature.parameters:
        kwargs["p"] = probability
    if "seed" in signature.parameters:
        kwargs["seed"] = seed

    try:
        augmenter = cls(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to instantiate TextAttack augmenter %s (%s), falling back to simple augmenter",
            recipe_name,
            exc,
        )
        return _fallback_augmenter(seed)

    if hasattr(augmenter, "seed"):
        try:
            augmenter.seed = seed
        except Exception:  # pragma: no cover
            pass

    return augmenter


def _apply_augmenter(
    augmenter: Augmenter,
    text: str,
    rng: random.Random,
    steps: int,
) -> str:
    result = text
    for _ in range(max(1, steps)):
        augmented = augmenter.augment(result)
        if not augmented:
            break
        result = rng.choice(augmented)
    return result


def _prepare_active_methods(mask: Mapping[str, bool], limit: int) -> list[str]:
    active = [name for name, enabled in mask.items() if enabled]
    if len(active) > limit:
        logger.warning(
            "Too many augmentation methods active (%d > %d). Trimming deterministically.",
            len(active),
            limit,
        )
    return active[:limit]


def build_augmenter(cfg: dict) -> Callable[[str], str]:
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    np.random.seed(seed)

    simple_cfg = cfg.get("simple", {})
    ta_cfg = cfg.get("ta", {})

    simple_methods = _prepare_active_methods(
        simple_cfg.get("enabled_mask", {}), MAX_SIMPLE_ACTIVE
    )
    ta_methods = _prepare_active_methods(ta_cfg.get("enabled_mask", {}), MAX_ATTACK_ACTIVE)

    simple_params = simple_cfg.get("params", {})
    ta_params = ta_cfg.get("params", {})

    simple_augmenters: list[Tuple[Augmenter, int]] = []
    for method in simple_methods:
        params = simple_params.get(method, {})
        augmenter = _instantiate_augmenter(
            method,
            seed=seed,
            transformations_per_example=int(params.get("tpe", 1)),
            probability=float(params.get("p_token", 0.1)),
        )
        steps = min(MAX_SEQUENCE_STEPS, max(1, int(params.get("tpe", 1))))
        simple_augmenters.append((augmenter, steps))

    attack_augmenter: Tuple[Augmenter, int] | None = None
    if ta_methods:
        method = ta_methods[0]
        params = ta_params.get(method, {})
        augmenter = _instantiate_augmenter(
            method,
            seed=seed,
            transformations_per_example=int(params.get("tpe", 1)),
            probability=float(params.get("p_token", 0.1)),
        )
        steps = min(MAX_SEQUENCE_STEPS, max(1, int(params.get("tpe", 1))))
        attack_augmenter = (augmenter, steps)
    elif ta_cfg.get("enabled_mask"):
        logger.debug("No adversarial augmentation enabled for this trial.")

    compose_simple = simple_cfg.get("compose", "one_of")
    cross_family = cfg.get("compose_cross_family", "attack_only")

    if not simple_augmenters:
        if cross_family in {"serial_then_attack", "attack_then_serial", "serial_only"}:
            cross_family = "attack_only" if attack_augmenter else "serial_only"
    if attack_augmenter is None and cross_family in {
        "serial_then_attack",
        "attack_then_serial",
        "attack_only",
    }:
        cross_family = "serial_only"

    if not simple_augmenters and attack_augmenter is None:
        logger.info("Augmentation disabled – no active methods.")
        return lambda text: text

    def apply_simple(text: str) -> str:
        if not simple_augmenters:
            return text
        if compose_simple == "sequence":
            result = text
            for augmenter, steps in simple_augmenters:
                result = _apply_augmenter(augmenter, result, rng, steps)
            return result
        augmenter, steps = rng.choice(simple_augmenters)
        return _apply_augmenter(augmenter, text, rng, steps)

    def apply_attack(text: str) -> str:
        if attack_augmenter is None:
            return text
        augmenter, steps = attack_augmenter
        return _apply_augmenter(augmenter, text, rng, steps)

    def pipeline(text: str) -> str:
        if cross_family == "serial_then_attack":
            return apply_attack(apply_simple(text))
        if cross_family == "attack_then_serial":
            return apply_simple(apply_attack(text))
        if cross_family == "serial_only":
            return apply_simple(text)
        if cross_family == "attack_only":
            return apply_attack(text)
        logger.warning("Unknown compose_cross_family=%s – returning original text", cross_family)
        return text

    return pipeline


def log_augmentation_samples(
    before_after_pairs: Iterable[tuple[str, str]],
    artifact_dir: Path,
    limit: int = 10,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    samples_path = artifact_dir / "augmentation_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for idx, (before, after) in enumerate(before_after_pairs):
            if idx >= limit:
                break
            payload = {"before": before, "after": after}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    log_artifact_safe(samples_path, artifact_path="augmentation")
