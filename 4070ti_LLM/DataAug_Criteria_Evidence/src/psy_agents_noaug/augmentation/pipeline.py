"""Augmentation pipeline orchestration and deterministic seeding helpers."""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .registry import (
    ALL_METHODS,
    LEGACY_NAME_MAP,
    NLPAUG_METHODS,
    REGISTRY,
    TEXTATTACK_METHODS,
    AugmenterWrapper,
    RegistryEntry,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ensure_sequence(methods: Sequence[str] | str | None) -> list[str]:
    if methods is None:
        return []
    if isinstance(methods, str):
        return [methods]
    return list(methods)


def _resolve_methods(methods: Sequence[str] | str | None) -> list[str]:
    declared = [m.strip() for m in _ensure_sequence(methods) if str(m).strip()]
    if not declared:
        declared = ["all"]

    resolved: list[str] = []
    for method in declared:
        lowered = method.lower()
        if lowered in {"all"}:
            resolved.extend(ALL_METHODS)
            continue
        if lowered in {"nlpaug", "nlpaug/all"}:
            resolved.extend(NLPAUG_METHODS)
            continue
        if lowered in {"textattack", "textattack/all"}:
            resolved.extend(TEXTATTACK_METHODS)
            continue
        mapped = LEGACY_NAME_MAP.get(method, method)
        if mapped not in REGISTRY:
            raise KeyError(f"Unknown augmentation method: {method}")
        resolved.append(mapped)

    unique: list[str] = []
    seen: set[str] = set()
    for method in resolved:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def _ratio_kwargs(name: str, ratio: float) -> dict[str, Any]:
    ratio = _clamp(ratio, 0.0, 1.0)
    if name.startswith("nlpaug/char/"):
        return {"aug_char_p": ratio}
    if name.startswith("nlpaug/word/"):
        return {"aug_p": ratio}
    mapping = {
        "textattack/CharSwapAugmenter": "pct_characters_to_swap",
        "textattack/DeletionAugmenter": "pct_words_to_delete",
        "textattack/SwapAugmenter": "pct_words_to_swap",
        "textattack/SynonymInsertionAugmenter": "pct_words_to_swap",
        "textattack/EasyDataAugmenter": "pct_words_to_swap",
        "textattack/CheckListAugmenter": "pct_words_to_swap",
        "textattack/WordNetAugmenter": "pct_words_to_swap",
    }
    param = mapping.get(name)
    return {param: ratio} if param else {}


def _merge_kwargs(
    base: dict[str, Any], override: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


@dataclass
class AugConfig:
    lib: str | None = None
    enabled: bool = False
    methods: Sequence[str] | str = field(default_factory=lambda: ("all",))
    p_apply: float = 0.15
    ops_per_sample: int = 1
    max_replace: float = 0.30
    max_replace_ratio: float | None = None
    tfidf_model_path: str | None = None
    reserved_map_path: str | None = None
    seed: int = 42
    method_weights: dict[str, float] | None = None
    method_kwargs: dict[str, dict[str, Any]] | None = None
    example_limit: int = 32
    allow_antonym: bool = True
    antonym_guard: str = "off"
    method_subset_size: int | None = None
    method_subset_seed: int | None = None
    tfidf_top_k: int | None = None

    def __post_init__(self) -> None:
        methods_seq = _ensure_sequence(self.methods)

        lib_lower = (self.lib or "").lower()
        if not methods_seq:
            if lib_lower in {"nlpaug", "textattack"}:
                methods_seq = [f"{lib_lower}/all"]
            else:
                methods_seq = ["all"]

        if not self.enabled and lib_lower not in {"", "none"}:
            self.enabled = True
        if not self.enabled and methods_seq != ["all"]:
            self.enabled = True

        mapped_methods = tuple(LEGACY_NAME_MAP.get(m, m) for m in methods_seq)
        object.__setattr__(self, "methods", mapped_methods)

        object.__setattr__(
            self, "ops_per_sample", max(1, min(3, int(self.ops_per_sample)))
        )
        object.__setattr__(self, "p_apply", float(self.p_apply))
        max_replace = self.max_replace
        if self.max_replace_ratio is not None:
            max_replace = float(self.max_replace_ratio)
        object.__setattr__(self, "max_replace", float(max_replace))
        object.__setattr__(self, "max_replace_ratio", float(max_replace))

        weights = {
            LEGACY_NAME_MAP.get(str(k), str(k)): float(v)
            for k, v in (self.method_weights or {}).items()
        }
        object.__setattr__(self, "method_weights", weights)

        kwargs = {
            LEGACY_NAME_MAP.get(str(k), str(k)): dict(v)
            for k, v in (self.method_kwargs or {}).items()
        }
        object.__setattr__(self, "method_kwargs", kwargs)

        object.__setattr__(self, "allow_antonym", bool(self.allow_antonym))

        guard = (self.antonym_guard or "off").lower()
        if not self.allow_antonym and guard == "off":
            guard = "on_low_weight"
        if guard not in {"off", "on_low_weight"}:
            raise ValueError("antonym_guard must be 'off' or 'on_low_weight'")
        object.__setattr__(self, "antonym_guard", guard)

        subset_size = self.method_subset_size
        if subset_size is not None:
            subset_size = max(1, int(subset_size))
        object.__setattr__(self, "method_subset_size", subset_size)

        if self.tfidf_top_k is not None:
            object.__setattr__(self, "tfidf_top_k", max(1, int(self.tfidf_top_k)))


@dataclass
class AugResources:
    tfidf_model_path: str | None = None
    reserved_map_path: str | None = None


class AugmenterPipeline:
    """Deterministic augmentation pipeline for evidence text."""

    def __init__(self, cfg: AugConfig, resources: AugResources | None = None) -> None:
        if not cfg.enabled:
            raise ValueError("Cannot instantiate AugmenterPipeline when enabled=False")

        methods = _resolve_methods(cfg.methods)
        if cfg.method_subset_size and cfg.method_subset_size < len(methods):
            subset_rng = random.Random(
                cfg.method_subset_seed
                if cfg.method_subset_seed is not None
                else cfg.seed
            )
            methods = subset_rng.sample(methods, cfg.method_subset_size)
        if not methods:
            raise ValueError("AugmenterPipeline requires at least one method")

        self.cfg = cfg
        self.methods = tuple(methods)
        self.resources = resources or AugResources()
        self.p_apply = _clamp(cfg.p_apply, 0.0, 1.0)
        self.ops_per_sample = max(1, min(3, int(cfg.ops_per_sample)))
        self.max_replace = _clamp(cfg.max_replace, 0.0, 1.0)
        self.example_limit = max(0, int(cfg.example_limit))

        self._rng = random.Random(cfg.seed)
        self._augmenters: list[AugmenterWrapper] = []
        self._weights: list[float] = []

        for name in self.methods:
            entry: RegistryEntry = REGISTRY[name]
            kwargs = _builder_kwargs(name, cfg, self.resources)
            try:
                wrapper = entry.factory(**kwargs)
            except TypeError as exc:
                LOGGER.debug(
                    "Retrying augmenter %s without ratio kwargs: %s", name, exc
                )
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"aug_p", "aug_char_p"} and not k.startswith("pct_")
                }
                wrapper = entry.factory(**clean_kwargs)
            self._augmenters.append(wrapper)
            weight = cfg.method_weights.get(name, 0.0) if cfg.method_weights else 0.0
            if (
                not cfg.method_weights
                and cfg.antonym_guard == "on_low_weight"
                and "AntonymAug" in name
            ):
                weight = -4.0
            self._weights.append(float(weight))

        if self._weights and any(w != 0.0 for w in self._weights):
            max_logit = max(self._weights)
            exp_vals = [math.exp(w - max_logit) for w in self._weights]
            total = sum(exp_vals) or 1.0
            self._weights = [val / total for val in exp_vals]
        else:
            count = len(self._augmenters)
            self._weights = [1.0 / count] * count

        self.method_counts: Counter[str] = Counter()
        self.applied_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.examples: list[dict[str, Any]] = []

    def set_seed(self, seed: int) -> None:
        self._rng.seed(int(seed))

    def close(self) -> None:
        if hasattr(self, "_augmenters"):
            self._augmenters.clear()

    def __enter__(self) -> AugmenterPipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __call__(self, text: str) -> str:
        self.total_count += 1
        if not self._augmenters or self.p_apply <= 0.0:
            self.skipped_count += 1
            return text

        if self._rng.random() > self.p_apply:
            self.skipped_count += 1
            return text

        augmented = text
        applied_methods: list[str] = []

        for _ in range(self.ops_per_sample):
            idx = self._rng.choices(
                range(len(self._augmenters)), weights=self._weights, k=1
            )[0]
            method_name = self.methods[idx]
            candidate = self._augmenters[idx].augment_one(augmented)
            if isinstance(candidate, str) and candidate:
                augmented = candidate
                applied_methods.append(method_name)

        if not applied_methods:
            self.skipped_count += 1
            return text

        self.applied_count += 1
        for method_name in applied_methods:
            self.method_counts[method_name] += 1

        if len(self.examples) < self.example_limit:
            self.examples.append(
                {
                    "original": text,
                    "augmented": augmented,
                    "methods": applied_methods,
                    "timestamp": time.time(),
                }
            )

        return augmented

    def stats(self) -> dict[str, Any]:
        return {
            "total": self.total_count,
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "method_counts": dict(self.method_counts),
        }

    def drain_examples(self) -> list[dict[str, Any]]:
        data = list(self.examples)
        self.examples.clear()
        return data


def _builder_kwargs(
    name: str, cfg: AugConfig, resources: AugResources
) -> dict[str, Any]:
    overrides = cfg.method_kwargs.get(name, {}) if cfg.method_kwargs else {}
    ratio_kwargs = _ratio_kwargs(name, cfg.max_replace)
    kwargs = _merge_kwargs(ratio_kwargs, overrides)

    if name == "nlpaug/char/RandomCharAug":
        kwargs.setdefault("action", "substitute")
    if name == "nlpaug/word/RandomWordAug":
        kwargs.setdefault("action", "swap")

    if name == "nlpaug/word/TfIdfAug":
        model_path = (
            kwargs.get("model_path")
            or cfg.tfidf_model_path
            or resources.tfidf_model_path
        )
        if model_path is None:
            raise ValueError("TfIdfAug requires tfidf_model_path")
        model_dir = Path(model_path)
        if model_dir.suffix:
            model_dir = model_dir.parent
        kwargs["model_path"] = str(model_dir)
        kwargs.setdefault("action", "substitute")
        if cfg.tfidf_top_k:
            kwargs.setdefault("top_k", int(cfg.tfidf_top_k))

    if name == "nlpaug/word/ReservedAug":
        reserved_path = (
            kwargs.get("reserved_map_path")
            or cfg.reserved_map_path
            or resources.reserved_map_path
        )
        if reserved_path is None:
            raise ValueError("ReservedAug requires reserved_map_path")
        kwargs["reserved_map_path"] = reserved_path
        kwargs.setdefault("action", "substitute")

    return kwargs


def worker_init(
    worker_id: int, base_seed: int, rank: int = 0, num_workers_per_rank: int = 1
) -> int:
    seed = int(base_seed) + (rank * num_workers_per_rank) + worker_id + 1
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:  # pragma: no cover
        import torch

        torch.manual_seed(seed)
    except Exception:
        LOGGER.debug("Torch not available for worker seeding", exc_info=True)
    return seed


def is_enabled(cfg: AugConfig) -> bool:
    if not cfg.enabled:
        return False
    try:
        return bool(_resolve_methods(cfg.methods))
    except KeyError:
        return False


__all__ = [
    "AugConfig",
    "AugResources",
    "AugmenterPipeline",
    "is_enabled",
    "worker_init",
]
