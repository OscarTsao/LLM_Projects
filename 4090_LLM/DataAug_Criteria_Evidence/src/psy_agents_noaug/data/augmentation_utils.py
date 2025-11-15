"""Utilities for constructing augmentation pipelines for the evidence task."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from psy_agents_noaug.augmentation import (
    ALLOWED_METHODS,
    LEGACY_NAME_MAP,
    AugConfig,
    AugmenterPipeline,
    AugResources,
    TfidfResource,
    is_enabled,
    load_or_fit_tfidf,
)


@dataclass(frozen=True)
class AugmentationArtifacts:
    pipeline: AugmenterPipeline
    config: AugConfig
    resources: AugResources
    tfidf: TfidfResource | None
    methods: tuple[str, ...]


def _ensure_sequence(methods: Sequence[str] | str | None) -> list[str]:
    if methods is None:
        return []
    if isinstance(methods, str):
        return [methods]
    return list(methods)


def _resolve_methods(methods: Sequence[str] | str | None) -> list[str]:
    declared = [
        LEGACY_NAME_MAP.get(m.strip(), m.strip()) for m in _ensure_sequence(methods)
    ]
    if not declared:
        declared = ["all"]

    resolved: list[str] = []
    for method in declared:
        lowered = method.lower()
        if lowered in {"all"}:
            resolved.extend(ALLOWED_METHODS)
            continue
        if lowered in {"nlpaug", "nlpaug/all"}:
            resolved.extend([m for m in ALLOWED_METHODS if m.startswith("nlpaug/")])
            continue
        if lowered in {"textattack", "textattack/all"}:
            resolved.extend([m for m in ALLOWED_METHODS if m.startswith("textattack/")])
            continue
        if method not in ALLOWED_METHODS:
            raise KeyError(f"Unknown augmentation method: {method}")
        resolved.append(method)

    unique: list[str] = []
    seen: set[str] = set()
    for method in resolved:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def resolve_methods(lib: str, methods: Sequence[str] | str | None) -> list[str]:
    """Public helper that filters resolved methods by library name."""
    lib_lower = (lib or "").lower()
    if lib_lower in {"none", ""}:
        return []
    if methods in (None, [], (), ""):
        return []
    resolved = _resolve_methods(methods)
    if lib_lower == "nlpaug":
        return [m for m in resolved if m.startswith("nlpaug/")]
    if lib_lower == "textattack":
        return [m for m in resolved if m.startswith("textattack/")]
    return resolved


def build_evidence_augmenter(
    cfg: AugConfig,
    train_texts: Iterable[str],
    *,
    tfidf_dir: str | Path = "_artifacts/tfidf/evidence",
) -> AugmentationArtifacts | None:
    if not is_enabled(cfg):
        return None

    cfg_copy = replace(cfg)
    resolved_methods = _resolve_methods(cfg_copy.methods)

    resources = AugResources(
        tfidf_model_path=cfg_copy.tfidf_model_path,
        reserved_map_path=cfg_copy.reserved_map_path,
    )
    tfidf_resource: TfidfResource | None = None

    filtered_methods: list[str] = []
    for method in resolved_methods:
        if method == "nlpaug/word/ReservedAug" and not (
            cfg_copy.reserved_map_path or resources.reserved_map_path
        ):
            continue
        filtered_methods.append(method)

    resolved_methods = filtered_methods
    cfg_copy = replace(cfg_copy, methods=tuple(resolved_methods))

    texts = [str(text) for text in train_texts]

    if any(method.endswith("TfIdfAug") for method in resolved_methods):
        target_dir = Path(tfidf_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = (
            Path(cfg_copy.tfidf_model_path)
            if cfg_copy.tfidf_model_path
            else target_dir / "tfidf.pkl"
        )
        tfidf_resource = load_or_fit_tfidf(texts, model_path)
        resources.tfidf_model_path = str(tfidf_resource.path)
        cfg_copy = replace(cfg_copy, tfidf_model_path=str(tfidf_resource.path))

    pipeline = AugmenterPipeline(cfg_copy, resources=resources)
    pipeline.set_seed(cfg_copy.seed)

    return AugmentationArtifacts(
        pipeline=pipeline,
        config=cfg_copy,
        resources=resources,
        tfidf=tfidf_resource,
        methods=tuple(pipeline.methods),
    )


__all__ = ["AugmentationArtifacts", "build_evidence_augmenter", "resolve_methods"]
