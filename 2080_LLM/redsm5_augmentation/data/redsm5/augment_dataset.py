"""
Augment the REDSM5 dataset by applying NLP augmentation methods to evidence sentences.

This script loads the REDSM5 posts and annotations, applies every non-empty
combination of 28 augmentation methods (11 from nlpaug and 17 from textattack)
to each evidence sentence, and writes the augmented posts to a CSV file located
in ``data/augmented``. The resulting CSV contains the columns:
``post_id, original_post, augmented_post, evidence_sentence, status``.

Usage (from the repository root or this directory):

    python augment_dataset.py --max-combinations 10

The ``--max-combinations`` flag is optional and limits the number of method
combinations attempted per evidence sentence, which is helpful for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import random
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nltk
from textattack.augmentation import Augmenter as TextAttackAugmenter
from textattack.transformations import (
    WordDeletion,
    WordInnerSwapRandom,
    WordInsertionRandomSynonym,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
    WordSwapHomoglyphSwap,
    WordSwapInflections,
    WordSwapMaskedLM,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapWordNet,
)

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configure basic logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for python, numpy, torch, and nlpaug."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are missing."""
    resources = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            LOGGER.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)


def _first_non_empty(output: Iterable[str]) -> Optional[str]:
    """Return the first non-empty stripped string from an iterable."""
    for candidate in output:
        if not isinstance(candidate, str):
            continue
        stripped = candidate.strip()
        if stripped:
            return stripped
    return None


def _normalize_output(output) -> Optional[str]:
    """Normalize augmenter outputs to a single string or None."""
    if output is None:
        return None
    if isinstance(output, str):
        stripped = output.strip()
        return stripped if stripped else None
    if isinstance(output, (list, tuple)):
        return _first_non_empty(output)
    return str(output).strip() or None


@dataclass(frozen=True)
class AugmentationMethod:
    """Callable augmentation method with metadata."""

    name: str
    provider: str
    apply_fn: Callable[[str], Optional[str]]

    def apply(self, text: str) -> Optional[str]:
        """Apply the augmentation and normalize its output."""
        try:
            result = self.apply_fn(text)
        except Exception as exc:
            LOGGER.warning("Augmenter %s failed: %s", self.name, exc)
            return None
        return _normalize_output(result)


def build_nlpaug_methods() -> List[AugmentationMethod]:
    """Return the 11 nlpaug augmentation methods."""
    synonym_aug = naw.SynonymAug(aug_src="wordnet")
    antonym_aug = naw.AntonymAug()
    random_swap_aug = naw.RandomWordAug(action="swap")
    random_delete_aug = naw.RandomWordAug(action="delete")
    spelling_aug = naw.SpellingAug()
    split_aug = naw.SplitAug()

    methods = [
        AugmentationMethod(
            name="nlpaug_random_char_insert",
            provider="nlpaug",
            apply_fn=lambda text, aug=nac.RandomCharAug(action="insert"): aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_random_char_delete",
            provider="nlpaug",
            apply_fn=lambda text, aug=nac.RandomCharAug(action="delete"): aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_random_char_substitute",
            provider="nlpaug",
            apply_fn=lambda text, aug=nac.RandomCharAug(action="substitute"): aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_keyboard",
            provider="nlpaug",
            apply_fn=lambda text, aug=nac.KeyboardAug(): aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_ocr",
            provider="nlpaug",
            apply_fn=lambda text, aug=nac.OcrAug(): aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_synonym_wordnet",
            provider="nlpaug",
            apply_fn=lambda text, aug=synonym_aug: aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_antonym_wordnet",
            provider="nlpaug",
            apply_fn=lambda text, aug=antonym_aug: aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_random_word_swap",
            provider="nlpaug",
            apply_fn=lambda text, aug=random_swap_aug: aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_random_word_delete",
            provider="nlpaug",
            apply_fn=lambda text, aug=random_delete_aug: aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_spelling",
            provider="nlpaug",
            apply_fn=lambda text, aug=spelling_aug: aug.augment(text),
        ),
        AugmentationMethod(
            name="nlpaug_split",
            provider="nlpaug",
            apply_fn=lambda text, aug=split_aug: aug.augment(text),
        ),
    ]
    return methods


def build_textattack_methods() -> List[AugmentationMethod]:
    """Return the 17 textattack augmentation methods."""
    transformations = [
        ("textattack_word_swap_neighboring_character", WordSwapNeighboringCharacterSwap()),
        ("textattack_word_swap_random_char_deletion", WordSwapRandomCharacterDeletion()),
        ("textattack_word_swap_random_char_insertion", WordSwapRandomCharacterInsertion()),
        ("textattack_word_swap_random_char_substitution", WordSwapRandomCharacterSubstitution()),
        ("textattack_word_swap_qwerty", WordSwapQWERTY()),
        ("textattack_word_swap_homoglyph", WordSwapHomoglyphSwap()),
        ("textattack_word_swap_wordnet", WordSwapWordNet()),
        ("textattack_word_swap_inflections", WordSwapInflections()),
        ("textattack_word_swap_change_number", WordSwapChangeNumber()),
        ("textattack_word_swap_change_name", WordSwapChangeName()),
        ("textattack_word_swap_change_location", WordSwapChangeLocation()),
        ("textattack_word_swap_extend", WordSwapExtend()),
        ("textattack_word_swap_contract", WordSwapContract()),
        ("textattack_word_deletion", WordDeletion()),
        (
            "textattack_word_swap_masked_lm",
            WordSwapMaskedLM(max_candidates=1),
        ),
        ("textattack_word_inner_swap_random", WordInnerSwapRandom()),
        ("textattack_word_insertion_random_synonym", WordInsertionRandomSynonym()),
    ]

    methods: List[AugmentationMethod] = []
    for name, transformation in transformations:
        augmenter = TextAttackAugmenter(
            transformation=transformation,
            transformations_per_example=1,
        )
        methods.append(
            AugmentationMethod(
                name=name,
                provider="textattack",
                apply_fn=lambda text, aug=augmenter: aug.augment(text),
            )
        )
    return methods


def generate_method_combinations(
    methods: Sequence[AugmentationMethod],
    max_combinations: Optional[int] = None,
) -> Iterator[Sequence[AugmentationMethod]]:
    """Yield every non-empty combination of methods up to ``max_combinations``."""
    produced = 0
    for size in range(1, len(methods) + 1):
        for combo in itertools.combinations(methods, size):
            yield combo
            produced += 1
            if max_combinations is not None and produced >= max_combinations:
                return


def apply_method_chain(
    sentence: str, methods: Sequence[AugmentationMethod]
) -> Optional[str]:
    """Apply a sequence of methods to a sentence in order."""
    augmented = sentence
    for method in methods:
        output = method.apply(augmented)
        if output is None:
            LOGGER.debug("Method %s returned None; stopping chain.", method.name)
            return None
        augmented = output
    return augmented


def replace_evidence_sentence(
    post_text: str, evidence_sentence: str, augmented_sentence: str
) -> Optional[str]:
    """Replace the first occurrence of the evidence sentence in the post."""
    if not evidence_sentence:
        LOGGER.debug("Empty evidence sentence encountered.")
        return None

    if evidence_sentence in post_text:
        return post_text.replace(evidence_sentence, augmented_sentence, 1)

    pattern_tokens = evidence_sentence.strip().split()
    if pattern_tokens:
        escaped_tokens = [re.escape(token) for token in pattern_tokens]
        pattern = r"\s+".join(escaped_tokens)
        match = re.search(pattern, post_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            start, end = match.span()
            return f"{post_text[:start]}{augmented_sentence}{post_text[end:]}"

    matcher = SequenceMatcher(
        None, post_text.lower(), evidence_sentence.lower()
    )
    match = matcher.find_longest_match(
        0, len(post_text), 0, len(evidence_sentence)
    )
    if match.size:
        overlap_ratio = match.size / max(len(evidence_sentence), 1)
        if overlap_ratio >= 0.6:
            start = match.a
            end = start + match.size
            return f"{post_text[:start]}{augmented_sentence}{post_text[end:]}"

    LOGGER.warning(
        "Evidence sentence not found in post even after normalization; skipping."
    )
    return None


def compute_total_combinations(total_methods: int) -> int:
    """Compute the number of non-empty subsets for informational logging."""
    return sum(comb(total_methods, r) for r in range(1, total_methods + 1))


def augment_dataset(
    annotations_path: Path,
    posts_path: Path,
    output_path: Path,
    max_combinations: Optional[int],
    max_annotations: Optional[int],
) -> None:
    """Load the dataset, perform augmentations, and write the output CSV."""
    LOGGER.info("Loading annotations from %s", annotations_path)
    annotations_df = pd.read_csv(annotations_path)
    LOGGER.info("Loading posts from %s", posts_path)
    posts_df = pd.read_csv(posts_path)
    posts_map = dict(zip(posts_df["post_id"], posts_df["text"]))

    nlpaug_methods = build_nlpaug_methods()
    textattack_methods = build_textattack_methods()
    all_methods = nlpaug_methods + textattack_methods
    total_combos = compute_total_combinations(len(all_methods))
    LOGGER.info(
        "Prepared %d augmentation methods (nlpaug=%d, textattack=%d).",
        len(all_methods),
        len(nlpaug_methods),
        len(textattack_methods),
    )
    LOGGER.info("Total combinations available per evidence sentence: %d", total_combos)
    if max_combinations is not None:
        LOGGER.info(
            "Limiting to %d combinations per evidence sentence for this run.",
            max_combinations,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "post_id",
        "original_post",
        "augmented_post",
        "evidence_sentence",
        "status",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        annotation_iter: Iterator = annotations_df.itertuples(index=False)
        if max_annotations is not None:
            annotation_iter = itertools.islice(annotation_iter, max_annotations)

        for annotation in annotation_iter:
            post_id = getattr(annotation, "post_id")
            evidence_sentence = getattr(annotation, "sentence_text")
            status = getattr(annotation, "status")

            post_text = posts_map.get(post_id)
            if not post_text:
                LOGGER.warning("Post text missing for post_id=%s; skipping.", post_id)
                continue

            LOGGER.info("Augmenting post_id=%s", post_id)

            combination_iter = generate_method_combinations(
                all_methods, max_combinations=max_combinations
            )
            combinations_attempted = 0
            combinations_saved = 0

            for methods in combination_iter:
                augmented_sentence = apply_method_chain(evidence_sentence, methods)
                combinations_attempted += 1
                if not augmented_sentence or augmented_sentence == evidence_sentence:
                    continue

                augmented_post = replace_evidence_sentence(
                    post_text, evidence_sentence, augmented_sentence
                )
                if not augmented_post:
                    continue

                writer.writerow(
                    {
                        "post_id": post_id,
                        "original_post": post_text,
                        "augmented_post": augmented_post,
                        "evidence_sentence": evidence_sentence,
                        "status": status,
                    }
                )
                combinations_saved += 1

            LOGGER.info(
                "Finished post_id=%s: attempted %d combinations, saved %d augmentations.",
                post_id,
                combinations_attempted,
                combinations_saved,
            )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(__file__).resolve().parent / "redsm5_annotations.csv",
        help="Path to redsm5_annotations.csv file.",
    )
    parser.add_argument(
        "--posts",
        type=Path,
        default=Path(__file__).resolve().parent / "redsm5_posts.csv",
        help="Path to redsm5_posts.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__).resolve().parent.parent / "augmented" / "redsm5_augmented.csv"),
        help="Output CSV path (default: data/augmented/redsm5_augmented.csv).",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of method combinations attempted per evidence sentence. "
            "If omitted, all 2^28 - 1 combinations are attempted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=None,
        help="Process at most this many annotations (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_global_seed(args.seed)
    ensure_nltk_resources()
    augment_dataset(
        annotations_path=args.annotations,
        posts_path=args.posts,
        output_path=args.output,
        max_combinations=args.max_combinations,
        max_annotations=args.max_annotations,
    )


if __name__ == "__main__":
    main()
