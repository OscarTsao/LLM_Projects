from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

CACHE_ROOT_DEFAULT = Path("experiments/cache")
CACHE_VERSION = 1


@dataclass
class CacheKey:
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    augmentation_fingerprint: str

    def to_str(self) -> str:
        return f"{self.dataset_fingerprint}__{self.augmentation_fingerprint}__{self.tokenizer_fingerprint}"


def _sha256_of_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def fingerprint_paths(paths: Iterable[Path]) -> str:
    """Fingerprint a set of file paths using (path, size, mtime_ns).

    Robust to reordering; does not read file contents for speed.
    """
    tuples: list[str] = []
    for p in sorted(map(Path, paths)):
        try:
            st = p.stat()
            tuples.append(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}")
        except FileNotFoundError:
            tuples.append(f"{p.resolve()}:MISSING:0")
    return _sha256_of_str("|".join(tuples))


def fingerprint_tokenizer(model_id: str, max_length: int = 512, padding: str = "max_length", truncation: bool = True) -> str:
    payload = json.dumps({"model_id": model_id, "max_length": max_length, "padding": padding, "truncation": truncation}, sort_keys=True)
    return _sha256_of_str(payload)


def fingerprint_augmentation(params: Mapping[str, Any] | None) -> str:
    if not params:
        return "none"
    # Keep only aug-related keys to avoid mixing in unrelated hparams
    aug_params = {k: v for k, v in params.items() if k.startswith("aug_")}
    if not aug_params:
        return "none"
    return _sha256_of_str(json.dumps(aug_params, sort_keys=True))


@dataclass
class CacheEntry:
    key: str
    split: str
    created_at: float
    last_access: float
    kind: str  # "tokenized" or "text"
    size_bytes: int
    path: str


class CacheIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._index_path = root / "index.json"
        self.entries: dict[str, CacheEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text())
            except Exception:
                data = {}
            for k, v in data.get("entries", {}).items():
                self.entries[k] = CacheEntry(**v)
        else:
            self.entries = {}

    def save(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "entries": {k: vars(v) for k, v in self.entries.items()},
        }
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(payload, indent=2))

    def update_access(self, entry_id: str) -> None:
        if entry_id in self.entries:
            self.entries[entry_id].last_access = time.time()

    def register(self, entry: CacheEntry) -> None:
        self.entries[entry.key] = entry
        self.save()

    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries.values())


def _split_dir(root: Path, key: CacheKey, split: str) -> Path:
    return root / key.dataset_fingerprint / key.augmentation_fingerprint / key.tokenizer_fingerprint / split


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class TokenizedDataset(torch.utils.data.Dataset):
    """A lightweight dataset backed by pre-tokenized tensors saved on disk.

    Expects a single .pt file containing a dict of tensors: input_ids, attention_mask,
    start_positions, end_positions.
    """

    def __init__(self, tensor_path: Path) -> None:
        self.tensor_path = tensor_path
        data = torch.load(tensor_path, map_location="cpu")
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.start_positions = data["start_positions"]
        self.end_positions = data["end_positions"]
        n = self.input_ids.size(0)
        assert all(t.size(0) == n for t in [self.attention_mask, self.start_positions, self.end_positions])
        self._len = n

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "start_positions": self.start_positions[idx],
            "end_positions": self.end_positions[idx],
        }


class MixedTokenizedDataset(torch.utils.data.Dataset):
    """Mix items from multiple tokenized datasets deterministically per-index.

    All child datasets must be the same length and share the same tensor schema.
    """

    def __init__(self, children: list[TokenizedDataset], seed: int | None = None) -> None:
        if not children:
            raise ValueError("MixedTokenizedDataset requires at least one child dataset")
        n = len(children[0])
        for ds in children[1:]:
            if len(ds) != n:
                raise ValueError("All child datasets must have the same length")
        self.children = children
        # Derive a deterministic 32-byte seed
        seed_bytes = hashlib.sha256(str(seed or 0).encode("utf-8")).digest()
        self._seed_bytes = seed_bytes
        self._len = n

    def __len__(self) -> int:
        return self._len

    def _choice(self, idx: int) -> int:
        # Deterministic pseudo-random choice per index based on seed hash and idx
        token = hashlib.sha256(self._seed_bytes + idx.to_bytes(8, "little")).digest()
        val = int.from_bytes(token[:4], "little")
        return val % len(self.children)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        source_idx = self._choice(idx)
        return self.children[source_idx][idx]


def try_load_tokenized_cache(root: Path, key: CacheKey, split: str, index: CacheIndex | None = None) -> TokenizedDataset | None:
    split_dir = _split_dir(root, key, split)
    tensor_path = split_dir / "tokenized.pt"
    if tensor_path.exists():
        if index:
            entry_id = f"{key.to_str()}::{split}::tokenized"
            index.update_access(entry_id)
            index.save()
        return TokenizedDataset(tensor_path)
    return None


def save_tokenized_cache(root: Path, key: CacheKey, split: str, batch_tensors: Mapping[str, torch.Tensor], index: CacheIndex | None = None) -> Path:
    split_dir = _split_dir(root, key, split)
    tensor_path = split_dir / "tokenized.pt"
    _ensure_parent(tensor_path)
    torch.save(batch_tensors, tensor_path)
    size = tensor_path.stat().st_size
    if index:
        entry = CacheEntry(
            key=f"{key.to_str()}::{split}::tokenized",
            split=split,
            created_at=time.time(),
            last_access=time.time(),
            kind="tokenized",
            size_bytes=size,
            path=str(tensor_path),
        )
        index.register(entry)
    return tensor_path


def compute_cache_key(dataset_files: Iterable[Path], tokenizer_model: str, max_length: int, aug_params: Mapping[str, Any] | None) -> CacheKey:
    ds_fp = fingerprint_paths(dataset_files)
    tok_fp = fingerprint_tokenizer(tokenizer_model, max_length=max_length)
    aug_fp = fingerprint_augmentation(aug_params)
    return CacheKey(ds_fp, tok_fp, aug_fp)


def enforce_size_budget(root: Path, index: CacheIndex, max_bytes: int) -> None:
    """Evict least-recently-used entries until under budget.

    This simple LRU evicts entire entries (per split).
    """
    total = index.total_size_bytes()
    if total <= max_bytes:
        return
    # Sort by last_access asc
    entries = sorted(index.entries.values(), key=lambda e: e.last_access)
    for e in entries:
        try:
            p = Path(e.path)
            if p.exists():
                p.unlink()
        except Exception:
            pass
        # Remove empty parent directories up to dataset-specific dir
        try:
            Path(e.path).parent.rmdir()
        except Exception:
            pass
        index.entries.pop(e.key, None)
        index.save()
        total = index.total_size_bytes()
        if total <= max_bytes:
            break


__all__ = [
    "CacheKey",
    "TokenizedDataset",
    "MixedTokenizedDataset",
    "CacheIndex",
    "CACHE_ROOT_DEFAULT",
    "compute_cache_key",
    "try_load_tokenized_cache",
    "save_tokenized_cache",
    "enforce_size_budget",
]

