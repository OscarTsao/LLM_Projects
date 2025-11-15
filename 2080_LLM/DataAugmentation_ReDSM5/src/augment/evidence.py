"""
Evidence span detection and replacement helpers.

This module locates evidence substrings within Reddit posts using an
exact-match heuristic followed by a fuzzy sentence-level fallback. The
replacement routine ensures that the rest of the post remains byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Optional, Tuple


_DELIMITERS = {".", "!", "?", "。"}


@dataclass(frozen=True)
class EvidenceMatch:
    """Represents a resolved evidence span within the post text."""

    start: int
    end: int
    span_text: str
    similarity: float
    match_type: str


class EvidenceReplacer:
    """
    Helper responsible for finding and replacing evidence spans.

    Parameters
    ----------
    min_similarity:
        Minimum similarity ratio (0-1) required to accept a fuzzy match.
    """

    def __init__(self, min_similarity: float = 0.55) -> None:
        self.min_similarity = min_similarity

    # ------------------------------------------------------------------ public
    def locate(self, post_text: str, evidence: str) -> Optional[EvidenceMatch]:
        """
        Locate the evidence span within ``post_text``.

        Returns ``None`` if no acceptable match is found.
        """
        if not evidence:
            return None

        exact_idx = post_text.find(evidence)
        if exact_idx != -1:
            return EvidenceMatch(
                start=exact_idx,
                end=exact_idx + len(evidence),
                span_text=evidence,
                similarity=1.0,
                match_type="exact",
            )

        best_match: Optional[EvidenceMatch] = None
        evidence_norm = evidence.strip().lower()
        sentences = list(_enumerate_sentences(post_text))
        for sent_text, start, end in sentences:
            candidate = sent_text.strip()
            if not candidate:
                continue
            ratio = SequenceMatcher(
                None, candidate.lower(), evidence_norm
            ).ratio()
            if ratio < self.min_similarity:
                continue
            if best_match is None or ratio > best_match.similarity:
                best_match = EvidenceMatch(
                    start=start,
                    end=end,
                    span_text=post_text[start:end],
                    similarity=ratio,
                    match_type="fuzzy",
                )
        return best_match

    def replace(self, post_text: str, match: EvidenceMatch, new_evidence: str) -> str:
        """Replace the matched span with the augmented evidence."""
        return post_text[: match.start] + new_evidence + post_text[match.end :]


def _enumerate_sentences(text: str) -> Iterable[Tuple[str, int, int]]:
    """
    Yield ``(sentence, start, end)`` tuples for the provided text.

    Splits on ., !, ?, 。 and newline characters while preserving byte offsets.
    """
    start = 0
    length = len(text)
    for idx, ch in enumerate(text):
        is_delimiter = ch in _DELIMITERS
        is_newline = ch == "\n"
        take = False
        if is_delimiter:
            take = True
        elif is_newline:
            take = True
        if take:
            end = idx + 1
            yield text[start:end], start, end
            start = end
    if start < length:
        yield text[start:], start, length
