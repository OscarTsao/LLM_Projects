#!/usr/bin/env python3
"""Regenerate augmented_positive_pairs.csv with inline paraphrases."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from difflib import SequenceMatcher

import pandas as pd

CRITERIA = [
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
]

MAX_VARIANTS = 3
MAX_ATTEMPTS = 12  # guardrail against infinite loops when generating variants
PREVIEW_ROWS = 20


@dataclass
class SynonymRule:
    pattern: str
    alternatives: Sequence[str]
    regex: re.Pattern[str]

    @classmethod
    def from_row(cls, pattern: str, alternatives: Sequence[str]) -> "SynonymRule":
        clean_alts = [alt.strip() for alt in alternatives if alt.strip()]
        return cls(pattern=pattern, alternatives=clean_alts, regex=re.compile(pattern, flags=re.IGNORECASE))


FALLBACK_PREFIXES = [
    "Honestly, ",
    "Truthfully, ",
    "To be honest, ",
]

FALLBACK_INTENSIFIERS = [
    (re.compile(r"\bfeel\b", re.IGNORECASE), ["feel really", "feel truly", "feel deeply"]),
    (re.compile(r"\bfeel like\b", re.IGNORECASE), ["feel very much like", "feel as though", "feel as if"]),
    (re.compile(r"\bam\b", re.IGNORECASE), ["am really", "am truly", "am completely"]),
    (re.compile(r"\bare\b", re.IGNORECASE), ["are really", "are truly", "are completely"]),
]


def _literal_pattern(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.startswith("\\b") or text.endswith("\\b"):
        return text
    return rf"\\b{re.escape(text)}\\b"


def _dedupe_rules(rules: Sequence[SynonymRule]) -> List[SynonymRule]:
    unique: dict[tuple[str, tuple[str, ...]], SynonymRule] = {}
    for rule in rules:
        key = (rule.regex.pattern, tuple(sorted(alt.lower() for alt in rule.alternatives)))
        unique[key] = rule
    return list(unique.values())


def load_synonym_rules(path: Path) -> List[SynonymRule]:
    rules: List[SynonymRule] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pattern = row.get("pattern", "").strip()
            alternatives = [alt.strip() for alt in row.get("alternatives", "").split("|") if alt.strip()]
            if not pattern:
                continue
            rules.append(SynonymRule.from_row(pattern, alternatives))
            # Expand to cover scenarios where an alternative already appears in the text.
            for alt in alternatives:
                literal = _literal_pattern(alt)
                if not literal:
                    continue
                reverse_alts = [candidate for candidate in {pattern, *alternatives} if candidate.strip() and candidate.strip().lower() != alt.lower()]
                if not reverse_alts:
                    continue
                rules.append(SynonymRule.from_row(literal, reverse_alts))
    return _dedupe_rules(rules)


def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement


def apply_synonym_rules(sentence: str, rules: Sequence[SynonymRule], seed: int) -> Tuple[str, List[str]] | None:
    updated = sentence
    replacements: List[str] = []
    for idx, rule in enumerate(rules):
        matches = list(rule.regex.finditer(updated))
        if not matches:
            continue
        match = matches[min(seed + len(replacements), len(matches) - 1)]
        original_text = match.group(0)
        viable_alts = [alt for alt in rule.alternatives if alt.lower() != original_text.lower()]
        if not viable_alts:
            continue
        alt_index = (seed + len(replacements)) % len(viable_alts)
        replacement = preserve_case(original_text, viable_alts[alt_index])
        start, end = match.span()
        updated = updated[:start] + replacement + updated[end:]
        replacements.append(f"{original_text}→{replacement}")
        if len(replacements) >= 1:
            break
    if replacements:
        return updated, replacements
    return None


def apply_fallback(sentence: str, seed: int) -> Tuple[str, List[str]] | None:
    # Try intensifier substitutions first
    for pattern, options in FALLBACK_INTENSIFIERS:
        match = pattern.search(sentence)
        if not match:
            continue
        original = match.group(0)
        viable = [opt for opt in options if opt.lower() != original.lower()]
        if not viable:
            continue
        replacement = preserve_case(original, viable[seed % len(viable)])
        start, end = match.span()
        updated = sentence[:start] + replacement + sentence[end:]
        return updated, [f"{original}→{replacement}"]

    # As last resort, add a softener prefix
    prefix = FALLBACK_PREFIXES[seed % len(FALLBACK_PREFIXES)]
    updated = prefix + sentence.lstrip()
    return updated, [f"prefix→{prefix.strip()}"]


def normalise_for_match(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("－", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def best_sentence_match(post_text: str, target: str) -> str | None:
    sentences = re.split(r"(?<=[.!?])\s+", post_text)
    if not sentences:
        sentences = [post_text]
    norm_target = normalise_for_match(target)
    best_sentence = None
    best_score = 0.0
    for sentence in sentences:
        norm_sentence = normalise_for_match(sentence)
        if not norm_sentence:
            continue
        score = SequenceMatcher(None, norm_target, norm_sentence).ratio()
        if score > best_score:
            best_score = score
            best_sentence = sentence
    if best_score >= 0.6:
        return best_sentence
    return None


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def replace_evidence(post_text: str, original: str, replacement: str) -> str:
    if original in post_text:
        return post_text.replace(original, replacement, 1)
    pattern = re.compile(r"\s+".join(map(re.escape, original.split())), flags=re.MULTILINE)
    match = pattern.search(post_text)
    if match:
        return post_text[:match.start()] + replacement + post_text[match.end():]
    candidate = best_sentence_match(post_text, original)
    if candidate and candidate in post_text:
        return post_text.replace(candidate, replacement, 1)
    raise ValueError("Evidence sentence not found in post text")


def unique_variants(sentence: str, rules: Sequence[SynonymRule], max_variants: int) -> List[Tuple[str, List[str]]]:
    variants: List[Tuple[str, List[str]]] = []
    seen = {normalise_whitespace(sentence)}
    attempts = 0
    seed = 0
    while len(variants) < max_variants and attempts < MAX_ATTEMPTS:
        produced = False
        attempt = apply_synonym_rules(sentence, rules, seed)
        if attempt is not None:
            candidate, replacements = attempt
            signature = normalise_whitespace(candidate)
            if signature not in seen:
                seen.add(signature)
                variants.append((candidate, replacements))
                produced = True
        if not produced:
            attempt = apply_fallback(sentence, seed)
            if attempt is not None:
                candidate, replacements = attempt
                signature = normalise_whitespace(candidate)
                if signature not in seen:
                    seen.add(signature)
                    variants.append((candidate, replacements))
                    produced = True
        seed += 1
        attempts += 1
        if not produced:
            continue
    if len(variants) < max_variants:
        filler_seed = seed
        while len(variants) < max_variants:
            prefix = FALLBACK_PREFIXES[filler_seed % len(FALLBACK_PREFIXES)]
            candidate = prefix + sentence.lstrip()
            signature = normalise_whitespace(candidate)
            filler_seed += 1
            if signature in seen:
                continue
            variants.append((candidate, [f"prefix→{prefix.strip()}"]))
            seen.add(signature)
    return variants


def build_positive_pairs(annotations: pd.DataFrame) -> pd.DataFrame:
    cleaned = annotations.copy()
    cleaned["DSM5_symptom"] = cleaned["DSM5_symptom"].replace({"LEEP_ISSUES": "SLEEP_ISSUES"})
    filtered = cleaned[(cleaned["status"] == 1) & (cleaned["DSM5_symptom"].isin(CRITERIA))]
    filtered = filtered.sort_values(["post_id", "DSM5_symptom", "sentence_id"])
    grouped = (
        filtered.groupby(["post_id", "DSM5_symptom"], as_index=False)
        .first()
        .rename(columns={"sentence_text": "evidence"})
    )
    return grouped[["post_id", "DSM5_symptom", "evidence"]]


def regenerate(
    posts_path: Path,
    annotations_path: Path,
    synonym_path: Path,
    output_path: Path,
    preview_path: Path,
) -> None:
    synonym_rules = load_synonym_rules(synonym_path)
    posts = pd.read_csv(posts_path).set_index("post_id")
    annotations = pd.read_csv(annotations_path)
    positives = build_positive_pairs(annotations)

    records = []
    for row in positives.itertuples(index=False):
        post_id = row.post_id
        criterion = row.DSM5_symptom
        evidence = row.evidence
        post_text = posts.loc[post_id, "text"]
        variants = unique_variants(evidence, synonym_rules, MAX_VARIANTS)
        if not variants:
            # retain original with explicit note if no transformation possible
            variants = [(evidence + " (rephrased)", ["note→rephrased"]) ]
        for idx, (aug_sentence, replacements) in enumerate(variants, start=1):
            try:
                post_augmented = replace_evidence(post_text, evidence, aug_sentence)
            except ValueError as exc:
                raise ValueError(f'Cannot locate evidence for {post_id} / {criterion}: {exc}') from exc
            records.append(
                {
                    "post_id": post_id,
                    "criterion": criterion,
                    "status": 1,
                    "evidence_original": evidence,
                    "evidence_augmented": aug_sentence,
                    "replacements": "; ".join(replacements),
                    "post_augmented": post_augmented,
                    "augmentation_method": "synonym_inline",
                    "aug_index": idx,
                }
            )

    df = pd.DataFrame(records)
    df.sort_values(["post_id", "criterion", "aug_index"], inplace=True)
    df.to_csv(output_path, index=False)

    preview = df.head(min(PREVIEW_ROWS, len(df)))
    preview.to_csv(preview_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate augmented dataset")
    parser.add_argument("--posts", default="Data/ReDSM5/redsm5_posts.csv", type=Path)
    parser.add_argument("--annotations", default="Data/ReDSM5/redsm5_annotations.csv", type=Path)
    parser.add_argument("--synonyms", default="Data/Augmentation/synonym_bank.tsv", type=Path)
    parser.add_argument("--output", default="Data/Augmentation/augmented_positive_pairs.csv", type=Path)
    parser.add_argument("--preview", default="Data/Augmentation/sample_aug_preview.csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regenerate(args.posts, args.annotations, args.synonyms, args.output, args.preview)


if __name__ == "__main__":
    main()
