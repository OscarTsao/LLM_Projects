#!/usr/bin/env python3
"""
CLI to generate evidence-only augmented datasets for Reddit DSM-5.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import sqlite3
import sys
import time
from collections import OrderedDict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from tqdm.auto import tqdm

from src.augment.combinator import ComboDescriptor, ComboGenerator
from src.augment.evidence import EvidenceReplacer
from src.augment.methods import (
    MethodRegistry,
    MethodUnavailableError,
    list_missing_methods,
)
logger = logging.getLogger("augment")

MATCH_MIN_SIM = 0.55


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


class AugmentationCache:
    """In-memory LRU with optional disk-backed persistence."""

    def __init__(self, maxsize: int = 2048, disk_path: Optional[Path] = None) -> None:
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, Tuple[str, ...]]" = OrderedDict()
        self.disk_path = disk_path
        if disk_path:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_db()

    def _ensure_db(self) -> None:
        with sqlite3.connect(self.disk_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )

    def _memory_get(self, key: str) -> Optional[Tuple[str, ...]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _memory_set(self, key: str, value: Sequence[str]) -> None:
        self._cache[key] = tuple(value)
        self._cache.move_to_end(key)
        while len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def _disk_get(self, key: str) -> Optional[Tuple[str, ...]]:
        if not self.disk_path:
            return None
        with sqlite3.connect(self.disk_path) as conn:
            row = conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        value = json.loads(row[0])
        return tuple(value)

    def _disk_set(self, key: str, value: Sequence[str]) -> None:
        if not self.disk_path:
            return
        payload = json.dumps(list(value))
        with sqlite3.connect(self.disk_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, payload),
            )

    def get(self, method_id: str, text: str) -> Optional[Tuple[str, ...]]:
        key = self._key(method_id, text)
        cached = self._memory_get(key)
        if cached is not None:
            return cached
        cached = self._disk_get(key)
        if cached is not None:
            self._memory_set(key, cached)
            return cached
        return None

    def set(self, method_id: str, text: str, values: Sequence[str]) -> None:
        key = self._key(method_id, text)
        self._memory_set(key, values)
        self._disk_set(key, values)

    @staticmethod
    def _key(method_id: str, text: str) -> str:
        digest = sha1(f"{method_id}\x1e{text}".encode("utf-8"), usedforsecurity=False)
        return digest.hexdigest()


# ---------------------------------------------------------------------------
# Argument parsing and top-level orchestration
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--text-col", required=True, help="Column with full post text.")
    parser.add_argument("--evidence-col", required=True, help="Column with evidence.")
    parser.add_argument("--criterion-col", help="Optional criterion passthrough column.")
    parser.add_argument("--label-col", help="Optional label passthrough column.")
    parser.add_argument("--id-col", help="Optional post_id passthrough column.")
    parser.add_argument(
        "--methods-yaml",
        default="conf/augment_methods.yaml",
        help="Path to augmentation method registry YAML.",
    )
    parser.add_argument(
        "--combo-mode",
        choices=("singletons", "bounded_k", "all"),
        default="singletons",
    )
    parser.add_argument(
        "--max-combo-size",
        type=int,
        default=3,
        help="Maximum combo size when combo-mode=bounded_k.",
    )
    parser.add_argument(
        "--confirm-powerset",
        action="store_true",
        help="Acknowledge full powerset generation when combo-mode=all.",
    )
    parser.add_argument(
        "--variants-per-sample",
        type=int,
        default=2,
        help="Number of augmented evidences to keep per row.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global deterministic seed.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/augsets",
        help="Root folder for augmented datasets.",
    )
    parser.add_argument(
        "--save-format",
        choices=("parquet", "csv"),
        default="parquet",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Worker processes for CPU-only combos.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index for distributed execution.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate combos even if outputs already exist.",
    )
    parser.add_argument(
        "--disk-cache",
        help="Optional SQLite cache file for augmenter outputs.",
    )
    parser.add_argument(
        "--quality-min-sim",
        type=float,
        default=0.40,
        help="Minimum similarity threshold between steps.",
    )
    parser.add_argument(
        "--quality-max-sim",
        type=float,
        default=0.98,
        help="Maximum similarity threshold between steps.",
    )
    parser.add_argument(
        "--global-dedupe",
        action="store_true",
        help="Enable global dedupe based on final post_text hash.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def derive_row_seed(global_seed: int, combo_id: str, row_index: int) -> int:
    digest = sha1(f"{global_seed}:{combo_id}:{row_index}".encode("utf-8"), usedforsecurity=False)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Worker context
# ---------------------------------------------------------------------------


_WORKER_STATE: Dict[str, Any] = {}


def _worker_initializer(config: Dict[str, Any]) -> None:
    global _WORKER_STATE
    registry_path = config["registry_path"]
    combo_methods = config["combo_methods"]

    _WORKER_STATE = {
        "registry": MethodRegistry(registry_path),
        "combo_methods": combo_methods,
        "combo_id": config["combo_id"],
        "variants": config["variants"],
        "quality_min": config["quality_min"],
        "quality_max": config["quality_max"],
        "global_seed": config["global_seed"],
        "max_candidates": config["max_candidates"],
        "replacer": EvidenceReplacer(MATCH_MIN_SIM),
        "cache": AugmentationCache(
            maxsize=config["cache_size"], disk_path=config["disk_cache"]
        ),
    }


def _worker_process(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]], bool]:
    row_index, row = task
    context = _WORKER_STATE
    registry: MethodRegistry = context["registry"]

    try:
        augmenters = [
            registry.instantiate(method_id) for method_id in context["combo_methods"]
        ]
    except MethodUnavailableError as exc:
        logger.warning("Skipping row %s due to unavailable method: %s", row_index, exc)
        return row_index, [], False

    row_seed = derive_row_seed(context["global_seed"], context["combo_id"], row_index)
    seed_everything(row_seed)
    rng = random.Random(row_seed)
    return apply_combo_to_row(
        row_index=row_index,
        row=row,
        augmenters=augmenters,
        method_ids=context["combo_methods"],
        variants=context["variants"],
        quality_min=context["quality_min"],
        quality_max=context["quality_max"],
        max_candidates=context["max_candidates"],
        cache=context["cache"],
        replacer=context["replacer"],
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Core augmentation logic
# ---------------------------------------------------------------------------


def apply_combo_to_row(
    row_index: int,
    row: Dict[str, Any],
    augmenters: Sequence[Any],
    method_ids: Sequence[str],
    variants: int,
    quality_min: float,
    quality_max: float,
    max_candidates: int,
    cache: AugmentationCache,
    replacer: EvidenceReplacer,
    rng: random.Random,
) -> Tuple[int, List[Dict[str, Any]], bool]:
    post_text = row["post_text"]
    evidence = row["evidence"]
    passthrough = row["passthrough"]
    match = replacer.locate(post_text, evidence)
    if match is None:
        return row_index, [], False

    augmented_evidences = generate_augmented_evidence(
        evidence,
        augmenters,
        method_ids,
        quality_min=quality_min,
        quality_max=quality_max,
        max_candidates=max_candidates,
        cache=cache,
        rng=rng,
    )
    results: List[Dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for aug_evidence in augmented_evidences:
        key = aug_evidence.casefold()
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        new_post = replacer.replace(post_text, match, aug_evidence)
        record = {
            "post_text": new_post,
            "evidence": aug_evidence,
            "evidence_original": evidence,
            "source_combo": "+".join(method_ids),
        }
        record.update(passthrough)
        results.append(record)
        if len(results) >= variants:
            break
    return row_index, results, True


def generate_augmented_evidence(
    seed_evidence: str,
    augmenters: Sequence[Any],
    method_ids: Sequence[str],
    *,
    quality_min: float,
    quality_max: float,
    max_candidates: int,
    cache: AugmentationCache,
    rng: random.Random,
) -> List[str]:
    candidates = [seed_evidence]
    original = seed_evidence
    for augmenter, method_id in zip(augmenters, method_ids):
        next_candidates: List[str] = []
        for candidate in candidates:
            minted = produce_candidates(
                augmenter=augmenter,
                method_id=method_id,
                text=candidate,
                cache=cache,
                rng=rng,
                limit=max_candidates,
            )
            filtered = [
                c
                for c in minted
                if _passes_quality(candidate, c, quality_min, quality_max)
            ]
            for text in filtered:
                if text not in next_candidates:
                    next_candidates.append(text)
        if not next_candidates:
            return []
        rng.shuffle(next_candidates)
        candidates = next_candidates[:max_candidates]

    final = [
        c
        for c in candidates
        if _passes_quality(original, c, quality_min, quality_max)
    ]
    deduped: List[str] = []
    seen = set()
    for item in final:
        key = item.casefold().strip()
        if not key or key == original.casefold().strip():
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def produce_candidates(
    augmenter: Any,
    method_id: str,
    text: str,
    cache: AugmentationCache,
    rng: random.Random,
    limit: int,
) -> List[str]:
    cached = cache.get(method_id, text)
    if cached is not None:
        return list(cached)[:limit]

    result: List[str] = []
    attempts = 0
    while len(result) < limit and attempts < limit * 4:
        attempts += 1
        generated = augmenter.augment(text, n=limit)
        if not generated:
            continue
        if isinstance(generated, str):
            generated_list = [generated]
        else:
            generated_list = list(generated)
        for candidate in generated_list:
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if not cleaned:
                continue
            if cleaned.casefold() == text.strip().casefold():
                continue
            result.append(cleaned)
            if len(result) >= limit:
                break
    if result:
        cache.set(method_id, text, result)
    return result[:limit]


def _passes_quality(source: str, candidate: str, minimum: float, maximum: float) -> bool:
    if not candidate or candidate.strip().lower() == source.strip().lower():
        return False
    ratio = SequenceMatcher(
        None, source.strip().lower(), candidate.strip().lower()
    ).ratio()
    return minimum <= ratio <= maximum


# ---------------------------------------------------------------------------
# Combo processing
# ---------------------------------------------------------------------------


def process_combo(
    combo: ComboDescriptor,
    df: pd.DataFrame,
    registry: MethodRegistry,
    args: argparse.Namespace,
    passthrough_columns: Sequence[str],
    available_methods: Sequence[str],
    quality_min: float,
    quality_max: float,
    summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    combo_methods = list(combo.methods)
    if any(method not in available_methods for method in combo_methods):
        logger.warning("Skipping combo %s due to unavailable methods.", combo.source_combo)
        return None

    combo_dir = Path(args.output_root) / f"combo_{combo.size}_{combo.combo_id}"
    combo_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = combo_dir / f"dataset.{args.save_format}"
    meta_path = combo_dir / "meta.json"

    if (
        not args.force
        and dataset_path.exists()
        and meta_path.exists()
    ):
        logger.info("Skipping combo %s (already exists).", combo.source_combo)
        summary["skipped_existing"] += 1
        return {
            "combo_id": combo.combo_id,
            "methods": combo.source_combo,
            "k": combo.size,
            "rows": None,
            "dataset_path": str(dataset_path),
            "status": "skipped",
        }

    requires_gpu = any(registry.get_spec(mid).requires_gpu for mid in combo_methods)
    num_proc = 1 if requires_gpu else max(1, args.num_proc)

    logger.info(
        "Processing combo %s | methods=%s | workers=%d",
        combo.combo_id,
        combo.source_combo,
        num_proc,
    )

    start_time = time.time()

    tasks = []
    for row_index, row in df.iterrows():
        passthrough = {col: row[col] for col in passthrough_columns if col in row}
        tasks.append(
            (
                row_index,
                {
                    "post_text": row[args.text_col],
                    "evidence": row[args.evidence_col],
                    "passthrough": passthrough,
                },
            )
        )

    cache_file = Path(args.disk_cache) if args.disk_cache else None
    worker_config = {
        "registry_path": str(registry.config_path),
        "combo_methods": combo_methods,
        "combo_id": combo.combo_id,
        "variants": args.variants_per_sample,
        "quality_min": quality_min,
        "quality_max": quality_max,
        "global_seed": args.seed,
        "max_candidates": 3,
        "cache_size": 4096,
        "disk_cache": cache_file,
    }

    results: List[Tuple[int, List[Dict[str, Any]], bool]]
    if num_proc > 1:
        with mp.get_context("spawn").Pool(
            processes=num_proc,
            initializer=_worker_initializer,
            initargs=(worker_config,),
            maxtasksperchild=50,
        ) as pool:
            iterator = pool.imap(_worker_process, tasks)
            results = list(tqdm(iterator, total=len(tasks), desc=f"{combo.combo_id}"))
    else:
        _worker_initializer(worker_config)
        results = []
        for task in tqdm(tasks, desc=f"{combo.combo_id}"):
            results.append(_worker_process(task))

    generated_rows: List[Dict[str, Any]] = []
    skip_counter = 0
    global_dedupe = set()
    for _, variants, matched in results:
        if not matched:
            skip_counter += 1
        for record in variants:
            if args.global_dedupe:
                digest = sha1(
                    record["post_text"].encode("utf-8"), usedforsecurity=False
                ).hexdigest()
                if digest in global_dedupe:
                    continue
                global_dedupe.add(digest)
            generated_rows.append(record)

    if not generated_rows:
        logger.warning("No rows generated for combo %s.", combo.source_combo)
        summary["empty"] += 1
        return None

    logger.info(
        "Combo %s generated %d rows (skipped=%d).",
        combo.combo_id,
        len(generated_rows),
        skip_counter,
    )

    out_df = pd.DataFrame(generated_rows)
    if args.save_format == "parquet":
        dataset_path = dataset_path.with_suffix(".parquet")
        out_df.to_parquet(dataset_path, index=False)
    else:
        dataset_path = dataset_path.with_suffix(".csv")
        out_df.to_csv(dataset_path, index=False)

    meta = {
        "combo_methods": combo_methods,
        "combo_id": combo.combo_id,
        "variants_per_sample": args.variants_per_sample,
        "seed": args.seed,
        "rows": int(len(generated_rows)),
        "source_input": str(Path(args.input).resolve()),
        "columns": list(out_df.columns),
        "timing": {
            "duration_seconds": time.time() - start_time,
            "rows_per_second": len(generated_rows) / max(
                1.0, time.time() - start_time
            ),
            "rows_skipped": int(skip_counter),
        },
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    summary["produced"] += 1
    summary["total_rows"] += len(generated_rows)
    return {
        "combo_id": combo.combo_id,
        "methods": combo.source_combo,
        "k": combo.size,
        "rows": len(generated_rows),
        "dataset_path": str(dataset_path),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def update_manifest(
    manifest_path: Path,
    entries: Iterable[Dict[str, Any]],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows = [entry for entry in entries if entry]
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        merged = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["combo_id"], keep="last")
            .sort_values("combo_id")
        )
    else:
        merged = new_df
    merged.to_csv(manifest_path, index=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging()
    start_time = time.time()

    registry_path = Path(args.methods_yaml)
    registry = MethodRegistry(registry_path)
    missing = list_missing_methods(registry)
    if missing:
        logger.warning("Unavailable methods detected:")
        for method_id, reason in missing.items():
            logger.warning("  %s -> %s", method_id, reason)
    available_methods = [
        method_id for method_id in registry.list_methods() if method_id not in missing
    ]
    logger.info("Available methods: %s", ", ".join(available_methods))

    df = pd.read_csv(args.input)
    required_cols = {args.text_col, args.evidence_col}
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    passthrough_cols = [
        col
        for col in (args.criterion_col, args.label_col, args.id_col)
        if col and col in df.columns
    ]

    combo_gen = ComboGenerator(available_methods)
    combos = list(
        combo_gen.iter_combos(
            mode=args.combo_mode,
            max_combo_size=args.max_combo_size,
            confirm_powerset=args.confirm_powerset,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    )
    logger.info(
        "Prepared %d combos for shard %d/%d.",
        len(combos),
        args.shard_index,
        args.num_shards,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {"produced": 0, "skipped_existing": 0, "total_rows": 0, "empty": 0}
    manifest_entries: List[Dict[str, Any]] = []

    for combo in combos:
        try:
            manifest_row = process_combo(
                combo=combo,
                df=df,
                registry=registry,
                args=args,
                passthrough_columns=passthrough_cols,
                available_methods=available_methods,
                quality_min=args.quality_min_sim,
                quality_max=args.quality_max_sim,
                summary=summary,
            )
            if manifest_row:
                manifest_entries.append(manifest_row)
        except Exception as exc:
            logger.exception("Combo %s failed: %s", combo.source_combo, exc)

    manifest_name = f"manifest_shard{args.shard_index}_of_{args.num_shards}.csv"
    manifest_path = output_root / manifest_name
    update_manifest(manifest_path, manifest_entries)

    elapsed = time.time() - start_time
    logger.info(
        "Completed in %.2fs | combos attempted=%d | produced=%d | skipped=%d | "
        "empty=%d | rows=%d",
        elapsed,
        len(combos),
        summary["produced"],
        summary["skipped_existing"],
        summary["empty"],
        summary["total_rows"],
    )


if __name__ == "__main__":
    main()
