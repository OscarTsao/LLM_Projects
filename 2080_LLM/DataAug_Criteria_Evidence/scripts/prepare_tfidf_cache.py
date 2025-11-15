#!/usr/bin/env python
"""Pre-fit TF-IDF resources for augmentation HPO.

This script loads training data and fits TF-IDF vectorizers for use with
nlpaug's TfIdfAug augmenter during HPO. Pre-fitting saves ~30-60s per trial.

Usage:
    python scripts/prepare_tfidf_cache.py --task criteria
    python scripts/prepare_tfidf_cache.py --task evidence --max-features 50000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from psy_agents_noaug.augmentation.tfidf_cache import load_or_fit_tfidf
from psy_agents_noaug.data.loaders import load_groundtruth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-fit TF-IDF cache for augmentation"
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["criteria", "evidence"],
        help="Task to fit TF-IDF for (determines which text field to use)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=40000,
        help="Maximum vocabulary size for TF-IDF vectorizer",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        default="1,2",
        help="N-gram range as comma-separated min,max (e.g., '1,2' for unigrams+bigrams)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/augmentation_cache/tfidf"),
        help="Directory to store TF-IDF cache files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse ngram range
    ngram_parts = args.ngram_range.split(",")
    if len(ngram_parts) != 2:
        raise ValueError(
            f"Invalid ngram-range: {args.ngram_range} (expected 'min,max')"
        )
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))

    LOGGER.info("Loading groundtruth for task=%s to extract training texts", args.task)

    # Load groundtruth
    gt = load_groundtruth(args.task)
    if not gt or "data" not in gt:
        raise ValueError(f"Failed to load groundtruth for task {args.task}")

    # Extract texts based on task
    texts: list[str] = []
    for row in gt["data"]:
        if args.task == "criteria":
            # Criteria uses post_text field
            text = row.get("post_text", "")
        elif args.task == "evidence":
            # Evidence uses evidence text spans
            if "evidence_texts" in row:
                # If pre-extracted evidence texts
                texts.extend(row["evidence_texts"])
                continue
            if "post_text" in row:
                # Fallback: use full post text
                text = row["post_text"]
            else:
                continue
        else:
            raise ValueError(f"Unknown task: {args.task}")

        if text and isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if not texts:
        raise ValueError(f"No texts extracted for task {args.task}")

    LOGGER.info("Extracted %d texts for TF-IDF fitting", len(texts))

    # Ensure cache directory exists
    cache_dir = args.cache_dir / args.task
    cache_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Fitting TF-IDF with max_features=%d, ngram_range=%s",
        args.max_features,
        ngram_range,
    )

    # Fit TF-IDF
    resource = load_or_fit_tfidf(
        train_texts=texts,
        model_path=cache_dir,
        max_features=args.max_features,
        ngram_range=ngram_range,
    )

    if resource.fitted:
        LOGGER.info(
            "✅ TF-IDF fitted in %.2fs and cached to: %s",
            resource.build_time_sec or 0.0,
            resource.path,
        )
    else:
        LOGGER.info(
            "✅ TF-IDF loaded from existing cache: %s",
            resource.path,
        )

    # Verify cache files exist
    idf_file = resource.path / "tfidfaug_w2idf.txt"
    tfidf_file = resource.path / "tfidfaug_w2tfidf.txt"
    vectorizer_file = resource.vectorizer_path

    if not idf_file.exists():
        raise FileNotFoundError(f"IDF file missing: {idf_file}")
    if not tfidf_file.exists():
        raise FileNotFoundError(f"TF-IDF file missing: {tfidf_file}")
    if not vectorizer_file or not vectorizer_file.exists():
        LOGGER.warning("Vectorizer pickle missing: %s", vectorizer_file)

    LOGGER.info("Cache validation passed:")
    LOGGER.info("  - IDF file: %s (%d bytes)", idf_file, idf_file.stat().st_size)
    LOGGER.info("  - TF-IDF file: %s (%d bytes)", tfidf_file, tfidf_file.stat().st_size)
    if vectorizer_file and vectorizer_file.exists():
        LOGGER.info(
            "  - Vectorizer: %s (%d bytes)",
            vectorizer_file,
            vectorizer_file.stat().st_size,
        )

    LOGGER.info(
        "Ready for augmentation HPO! Use --aug-tfidf-model %s in CLI",
        cache_dir,
    )


if __name__ == "__main__":
    main()
