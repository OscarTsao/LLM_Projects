from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Union


def _load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _load_csv(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {
                "post_id": row.get("post_id") or row.get("id"),
                "sentence_id": row.get("sentence_id"),
                "text": row.get("text"),
                "symptom": row.get("symptom"),
                "status": int(row.get("status", 0)),
            }


def _csv_to_records(rows: Iterable[Dict]) -> List[Dict]:
    posts: Dict[str, Dict] = {}
    for row in rows:
        pid = row["post_id"]
        posts.setdefault(pid, {"post_id": pid, "sentences": [], "labels": []})
        sentence_id = row.get("sentence_id") or f"S{len(posts[pid]['sentences'])+1}"
        posts[pid]["sentences"].append({"sentence_id": sentence_id, "text": row.get("text", "")})
        posts[pid]["labels"].append(
            {
                "sentence_id": sentence_id,
                "symptom": row.get("symptom", "UNKNOWN"),
                "status": row.get("status", 0),
            }
        )
    return list(posts.values())


def load_dataset(path: Union[str, Path]) -> Iterable[Dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        yield from _load_jsonl(path)
    elif path.suffix.lower() == ".csv":
        yield from _csv_to_records(_load_csv(path))
    else:
        raise ValueError(f"Unsupported dataset format: {path}")

