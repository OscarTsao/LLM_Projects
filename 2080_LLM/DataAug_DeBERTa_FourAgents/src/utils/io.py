from __future__ import annotations

import json
import os
from typing import Iterable, Dict, Any, Tuple


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def persist_splits(train_ids, dev_ids, test_ids, out_dir: str) -> Tuple[str, str, str]:
    """Persist split ID lists to the configured directory as JSONL files.

    Each file is one ID per line encoded as JSON string or object with {"post_id": id}.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_p = os.path.join(out_dir, 'train.jsonl')
    dev_p = os.path.join(out_dir, 'dev.jsonl')
    test_p = os.path.join(out_dir, 'test.jsonl')

    def _norm(ids):
        for _id in ids:
            if isinstance(_id, dict):
                yield _id
            else:
                yield {"post_id": str(_id)}

    write_jsonl(train_p, _norm(train_ids))
    write_jsonl(dev_p, _norm(dev_ids))
    write_jsonl(test_p, _norm(test_ids))
    return train_p, dev_p, test_p


def try_generate_groupkfold_splits(posts, groups, n_splits=5, seed=42):
    """Optional helper to generate GroupKFold splits if scikit-learn is available.

    Returns one (train_idx, dev_idx, test_idx) triple. If sklearn not available,
    raises ImportError.
    """
    from sklearn.model_selection import GroupKFold  # type: ignore
    import numpy as np  # type: ignore

    rng = np.random.default_rng(seed)
    gkf = GroupKFold(n_splits=n_splits)
    # Use first fold as dev/test split from held-out groups; rest train
    indices = np.arange(len(posts))
    splits = list(gkf.split(indices, groups=groups))
    (train_idx, rest_idx) = splits[0]
    # Split rest in half for dev/test
    half = len(rest_idx) // 2
    dev_idx = rest_idx[:half]
    test_idx = rest_idx[half:]
    return train_idx, dev_idx, test_idx

