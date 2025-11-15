from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

from datasets import DatasetDict


def build_cache_path(
    base_dir: str | Path,
    *,
    dataset_id: str,
    revision: str | None,
    tokenizer_name: str,
    max_length: int,
    extra: str | None = None,
) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    revision_str = revision or "latest"
    fingerprint = f"{dataset_id}|{revision_str}|{tokenizer_name}|{max_length}"
    if extra:
        fingerprint += f"|{extra}"
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return base_path / digest


def load_cached_dataset(path: Path) -> Optional[DatasetDict]:
    if not path.exists():
        return None
    return DatasetDict.load_from_disk(str(path))


def save_cached_dataset(dataset: DatasetDict, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    if path.exists():
        shutil.rmtree(path)
    shutil.move(str(tmp_path), str(path))
