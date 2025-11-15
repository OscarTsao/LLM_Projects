"""Convert Gemini judgments into DPO preference records."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable, Set, Tuple

import hydra
from omegaconf import DictConfig

from ..schemas import DPORecord, Judgment
from ..io_utils import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)


def _load_judgments(path: Path) -> list[Judgment]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [Judgment.model_validate(row) for row in read_jsonl(path)]


def _prompt_from_judgment(judgment: Judgment) -> str:
    parts = [
        f"Criterion ID: {judgment.criterion_id}",
        f"Criterion Definition: {judgment.criterion_text}",
        "Instructions: identify whether the note supports the criterion.",
        "--- Note ---",
        judgment.note_text.strip(),
        "---------------",
    ]
    return "\n".join(parts)


def _records_from_judgment(judgment: Judgment) -> Iterable[DPORecord]:
    prompt = _prompt_from_judgment(judgment)
    candidates = [candidate.text for candidate in judgment.candidates]
    seen_pairs: Set[Tuple[str, str]] = set()
    for pref in judgment.preferences:
        try:
            chosen = candidates[pref.winner_idx]
            rejected = candidates[pref.loser_idx]
        except IndexError:
            continue
        pair_key = (chosen, rejected)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        yield DPORecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={
                "job_id": judgment.job_id,
                "criterion_id": judgment.criterion_id,
                "provider": judgment.provider,
            },
        )


def _subsample(records: list[DPORecord], sample_rate: float, max_pairs: int | None, seed: int) -> list[DPORecord]:
    if sample_rate >= 1.0 and not max_pairs:
        return records
    rng = random.Random(seed)
    filtered = [record for record in records if rng.random() <= sample_rate]
    if max_pairs is not None:
        filtered = filtered[:max_pairs]
    return filtered


@hydra.main(config_path="../../../conf", config_name="prefs_to_dpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    input_path = Path(cfg.input_path)
    output_path = Path(cfg.output_path)
    sample_rate = float(cfg.get("sample_rate", 1.0))
    max_pairs = cfg.get("max_pairs")
    shuffle = bool(cfg.get("shuffle", False))
    seed = int(cfg.get("seed", 42))

    judgments = _load_judgments(input_path)
    records: list[DPORecord] = []
    for judgment in judgments:
        records.extend(_records_from_judgment(judgment))
    LOGGER.info("Derived %d raw preference pairs", len(records))

    records = _subsample(records, sample_rate, max_pairs, seed)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, [record.to_dict() for record in records])
    LOGGER.info("Wrote %d DPO records to %s", len(records), output_path)


if __name__ == "__main__":
    main()
