from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from src.utils.io import read_jsonl


def load_split_ids(path: Path) -> List[str]:
    path = Path(path)
    if not path.is_file():
        return []
    return [row.get("post_id") for row in read_jsonl(path) if row.get("post_id") is not None]


def filter_dataset_by_ids(dataset: Sequence[dict], allowed_ids: Iterable[str]) -> List[dict]:
    allowed = {str(pid) for pid in allowed_ids}
    return [item for item in dataset if str(item.get("post_id")) in allowed]
