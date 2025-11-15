#!/usr/bin/env python3
"""
Download Hugging Face models required for deterministic augmentation runs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

logger = logging.getLogger("prefetch")


MASKED_LM_MODELS = [
    "bert-base-uncased",
    "roberta-base",
]

SEQ2SEQ_MODELS = [
    "facebook/wmt19-en-de",
    "facebook/wmt19-de-en",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory to reuse/download models.",
    )
    return parser.parse_args()


def download_masked_lm(model_name: str, cache_dir: Path | None) -> None:
    logger.info("Prefetching masked LM model %s", model_name)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)


def download_seq2seq(model_name: str, cache_dir: Path | None) -> None:
    logger.info("Prefetching seq2seq model %s", model_name)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    configure_logging()

    cache_dir = args.cache_dir
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using cache dir %s", cache_dir.resolve())

    for model_name in MASKED_LM_MODELS:
        download_masked_lm(model_name, cache_dir)

    for model_name in SEQ2SEQ_MODELS:
        download_seq2seq(model_name, cache_dir)

    logger.info("Model prefetch complete.")


if __name__ == "__main__":
    main()
