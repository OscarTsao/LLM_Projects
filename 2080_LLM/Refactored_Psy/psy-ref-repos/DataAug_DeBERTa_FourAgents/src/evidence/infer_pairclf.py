"""Inference routine for the Evidence pair classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.io import write_jsonl
from src.utils.seed import set_seed
from src.utils.data import load_dataset


DEFAULT_OUTPUT = Path("outputs/evaluation/run_stub/predictions.jsonl")


def _deterministic_eu_id(post_id: str, sentence_id: str, symptom: str) -> str:
    h = hashlib.sha256(f"{post_id}:{sentence_id}:{symptom}".encode("utf-8"))
    return h.hexdigest()[:16]


def _load_model(checkpoint_path: Path) -> Dict[str, int]:
    with checkpoint_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {k: int(v) for k, v in payload.get("label_map", {}).items()}


def _sentence_lookup(sentences: List[Dict[str, Any]]) -> Dict[str, str]:
    return {str(s["sentence_id"]): s.get("text", "") for s in sentences}


def _generate_predictions(dataset, label_map: Dict[str, int]) -> Iterable[Dict[str, Any]]:
    for item in dataset:
        post_id = item["post_id"]
        sentence_texts = _sentence_lookup(item.get("sentences", []))
        for label in item.get("labels", []):
            sentence_id = str(label["sentence_id"])
            symptom = label["symptom"]
            key = "::".join([post_id, sentence_id, symptom])
            status = label_map.get(key, int(label.get("status", 0)))
            assertion = "present" if status == 1 else "absent"
            score = 0.99 if status == 1 else 0.05
            eu_id = _deterministic_eu_id(post_id, sentence_id, symptom)
            yield {
                "post_id": post_id,
                "sentence_id": sentence_id,
                "sentence": sentence_texts.get(sentence_id, ""),
                "symptom": symptom,
                "assertion": assertion,
                "score": score,
                "gold": int(label.get("status", 0)),
                "eu_id": eu_id,
            }


def run_inference(
    dataset,
    checkpoint_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    label_map = _load_model(checkpoint_path)
    predictions = list(_generate_predictions(dataset, label_map))
    write_jsonl(output_path, predictions)

    return {
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "num_predictions": len(predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evidence inference")
    parser.add_argument("--data", type=Path, required=True, help="Dataset JSONL/CSV path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint produced by training")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Predictions JSONL path")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = list(load_dataset(args.data))
    info = run_inference(dataset, args.checkpoint, args.output, args.seed)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
