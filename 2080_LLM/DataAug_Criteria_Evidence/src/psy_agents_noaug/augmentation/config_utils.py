"""Utilities for converting augmentation sections into AugConfig objects."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import AugConfig
from .registry import LEGACY_NAME_MAP

LOGGER = logging.getLogger(__name__)


def _coerce_methods(value: Sequence[str] | str | None) -> Sequence[str] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return LEGACY_NAME_MAP.get(value, value)
    return [LEGACY_NAME_MAP.get(v, v) for v in value]


def _coerce_method_weights(raw: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse aug.method_weights JSON string")
            return None
    if isinstance(raw, Mapping):
        return {LEGACY_NAME_MAP.get(str(k), str(k)): float(v) for k, v in raw.items()}
    if isinstance(raw, Sequence):
        weights: dict[str, float] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            if "name" in item and "weight" in item:
                name = LEGACY_NAME_MAP.get(str(item["name"]), str(item["name"]))
                try:
                    weights[name] = float(item["weight"])
                except (TypeError, ValueError):
                    continue
        return weights or None
    return None


def hpo_config_to_aug_config(
    config: Mapping[str, Any],
    global_seed: int = 42,
) -> AugConfig | None:
    aug_dict = config.get("augmentation")
    if not isinstance(aug_dict, Mapping):
        return None

    enabled = bool(aug_dict.get("enabled", False))
    if not enabled:
        return None

    methods = aug_dict.get("methods")
    if methods is None:
        legacy_lib = aug_dict.get("lib")
        if isinstance(legacy_lib, str) and legacy_lib.lower() != "none":
            methods = legacy_lib

    p_apply = float(aug_dict.get("p_apply", 0.15))
    ops_per_sample = int(aug_dict.get("ops_per_sample", 1))
    max_replace = float(
        aug_dict.get("max_replace", aug_dict.get("max_replace_ratio", 0.3))
    )
    seed = aug_dict.get("seed", global_seed)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = global_seed

    tfidf_model_path = aug_dict.get("tfidf_model_path")
    reserved_map_path = aug_dict.get("reserved_map_path")
    allow_antonym = bool(aug_dict.get("allow_antonym", True))

    method_kwargs = aug_dict.get("method_kwargs")
    if isinstance(method_kwargs, str):
        try:
            method_kwargs = json.loads(method_kwargs)
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse aug.method_kwargs JSON string")
            method_kwargs = None

    method_weights = _coerce_method_weights(aug_dict.get("method_weights"))

    antonym_guard = str(aug_dict.get("antonym_guard", "off")).lower()
    subset_size = aug_dict.get("method_subset_size")
    subset_seed = aug_dict.get("method_subset_seed")
    tfidf_top_k = aug_dict.get("tfidf_top_k")

    if subset_size is not None:
        try:
            subset_size = int(subset_size)
        except (TypeError, ValueError):
            subset_size = None

    if subset_seed is not None:
        try:
            subset_seed = int(subset_seed)
        except (TypeError, ValueError):
            subset_seed = None

    if tfidf_top_k is not None:
        try:
            tfidf_top_k = int(tfidf_top_k)
        except (TypeError, ValueError):
            tfidf_top_k = None

    aug_cfg = AugConfig(
        enabled=True,
        methods=_coerce_methods(methods),
        p_apply=p_apply,
        ops_per_sample=ops_per_sample,
        max_replace=max_replace,
        tfidf_model_path=tfidf_model_path,
        reserved_map_path=reserved_map_path,
        seed=seed,
        method_weights=method_weights,
        method_kwargs=method_kwargs,
        example_limit=int(aug_dict.get("example_limit", 32)),
        allow_antonym=allow_antonym,
        antonym_guard=antonym_guard,
        method_subset_size=subset_size,
        method_subset_seed=subset_seed,
        tfidf_top_k=tfidf_top_k,
    )

    LOGGER.info(
        "Augmentation enabled: methods=%s p_apply=%.3f ops=%d max_replace=%.3f seed=%d",
        aug_cfg.methods,
        aug_cfg.p_apply,
        aug_cfg.ops_per_sample,
        aug_cfg.max_replace,
        aug_cfg.seed,
    )
    return aug_cfg


__all__ = ["hpo_config_to_aug_config"]
