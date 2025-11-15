from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml  # type: ignore

from dataaug_multi_both.cache.dataset_cache import (
    CACHE_ROOT_DEFAULT,
    CacheIndex,
    CacheKey,
    compute_cache_key,
    enforce_size_budget,
    save_tokenized_cache,
)
from dataaug_multi_both.data import DatasetLoader, build_dataset_config_from_dict, create_collator
from dataaug_multi_both.augment.unified_augmenter import (
    UnifiedAugmenter,
    ALL_AUG_METHODS,
    AugmentedDataset,
)


def _collect_dataset_files(dataset_cfg_path: Path) -> list[Path]:
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve data_files from config if present; otherwise use path
    files: list[Path] = []
    ds = cfg.get("dataset", {})
    data_files = ds.get("data_files")
    base_dir = dataset_cfg_path.parent
    if isinstance(data_files, dict):
        for v in data_files.values():
            if isinstance(v, str):
                files.append((base_dir / v).resolve())
            elif isinstance(v, list):
                for it in v:
                    files.append((base_dir / it).resolve())
    elif isinstance(data_files, list):
        for it in data_files:
            files.append((base_dir / it).resolve())
    else:
        # fallback: single file at dataset path
        path_str = ds.get("path") or ds.get("local_path")
        if path_str:
            files.append((base_dir / path_str).resolve())
    return files


def _load_dataset(dataset_cfg_path: Path):
    with open(dataset_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds_cfg = build_dataset_config_from_dict(cfg.get("dataset", {}), dataset_cfg_path.parent)
    loader = DatasetLoader()
    splits = loader.load(ds_cfg)
    # Normalize to python lists/dicts if huggingface-like objects
    def _as_py(ds):
        return ds.with_format("python") if hasattr(ds, "with_format") else ds
    return {k: _as_py(v) for k, v in splits.items()}


def _concat_tensors(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.cat([c[k] for c in chunks], dim=0)
    return out


def _iter_chunk(dataset, chunk_size: int):
    n = len(dataset)
    for i in range(0, n, chunk_size):
        yield [dataset[j] for j in range(i, min(n, i + chunk_size))]


def precompute_tokenized(
    hpo_config: Path,
    cache_dir: Path,
    max_cache_gb: float,
    splits: Iterable[str] = ("train", "validation"),
    chunk_size: int = 512,
    augment_all: bool = False,
) -> None:
    with open(hpo_config, "r", encoding="utf-8") as f:
        study_cfg = yaml.safe_load(f)

    dataset_cfg_path = Path(study_cfg["dataset"]["config_path"]).resolve()
    dataset_files = _collect_dataset_files(dataset_cfg_path)
    datasets = _load_dataset(dataset_cfg_path)

    # Model choices
    search_space = study_cfg.get("search_space", {})
    model_space = search_space.get("model_id", {})
    model_choices = model_space.get("choices", [])
    if not model_choices:
        # fallback to a single model if not specified
        model_choices = [study_cfg.get("fixed", {}).get("model_id", "microsoft/deberta-v3-base")]

    max_length = 512
    training_cfg = study_cfg.get("training", {})
    if "max_length" in training_cfg:
        max_length = int(training_cfg["max_length"])  # type: ignore[arg-type]

    index = CacheIndex(cache_dir)
    budget_bytes = int(max_cache_gb * (1024**3))

    for model_id in model_choices:
        collator = create_collator(model_name_or_path=str(model_id), max_length=max_length)
        # Base (non-augmented) caches for requested splits
        base_key: CacheKey = compute_cache_key(dataset_files, str(model_id), max_length, aug_params=None)
        for split in splits:
            if split not in datasets:
                continue
            ds = datasets[split]
            chunks: list[dict[str, torch.Tensor]] = []
            for batch in _iter_chunk(ds, chunk_size):
                batch_tensors = collator(batch)
                chunks.append(batch_tensors)
            merged = _concat_tensors(chunks)
            save_tokenized_cache(cache_dir, base_key, split, merged, index=index)
            enforce_size_budget(cache_dir, index, budget_bytes)

        # Augmented caches (exhaustive policy) — do TRAIN split only
        if augment_all and "train" in datasets:
            for method in ALL_AUG_METHODS:
                aug = UnifiedAugmenter(methods=[method], aug_prob=1.0, compose_mode="sequential", seed=0)
                ds_aug = AugmentedDataset(datasets["train"], augmenter=aug)
                aug_key: CacheKey = compute_cache_key(
                    dataset_files,
                    str(model_id),
                    max_length,
                    aug_params={"methods": [method], "policy": "single"},
                )
                chunks_aug: list[dict[str, torch.Tensor]] = []
                for batch in _iter_chunk(ds_aug, chunk_size):
                    batch_tensors = collator(batch)
                    chunks_aug.append(batch_tensors)
                merged_aug = _concat_tensors(chunks_aug)
                save_tokenized_cache(cache_dir, aug_key, "train", merged_aug, index=index)
                enforce_size_budget(cache_dir, index, budget_bytes)

    # Emit summary
    total_gb = index.total_size_bytes() / (1024**3)
    manifest = {
        "cache_dir": str(cache_dir),
        "total_size_gb": total_gb,
        "entries": len(index.entries),
        "models": model_choices,
        "splits": list(splits),
        "dataset_config": str(dataset_cfg_path),
        "augmented_all": bool(augment_all),
        "augmented_methods": list(ALL_AUG_METHODS) if augment_all else [],
    }
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Precompute tokenized dataset caches for HPO")
    p.add_argument("--hpo-config", required=True, help="Path to HPO study YAML")
    p.add_argument("--cache-dir", default=str(CACHE_ROOT_DEFAULT), help="Cache directory")
    p.add_argument("--max-cache-gb", type=float, default=50.0, help="Max cache size budget in GB")
    p.add_argument("--splits", default="train,validation", help="Comma-separated splits to cache")
    p.add_argument("--chunk-size", type=int, default=512, help="Chunk size for collate passes")
    p.add_argument("--augment-all", action="store_true", help="Also precompute all augmentation methods for train split")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    precompute_tokenized(
        hpo_config=Path(args.hpo_config),
        cache_dir=cache_dir,
        max_cache_gb=float(args.max_cache_gb),
        splits=splits,
        chunk_size=int(args.chunk_size),
        augment_all=bool(args.augment_all),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

