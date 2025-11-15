"""Test-only augmentation stubs used to avoid heavy dependencies."""

from __future__ import annotations

import random
from typing import List


class PrefixAugmenter:
    """Prefixes the text with a short token to ensure quality thresholds."""

    def __init__(self, prefix: str = "[AUG] ") -> None:
        self.prefix = prefix

    def augment(self, text: str, n: int = 1) -> List[str]:
        return [f"{self.prefix}{text}" for _ in range(max(1, n))]


class RandomSuffixAugmenter:
    """Appends deterministic suffixes based on global randomness."""

    def __init__(self, suffixes: list[str] | None = None) -> None:
        self.suffixes = suffixes or ["!", "?", "..."]

    def augment(self, text: str, n: int = 1) -> List[str]:
        results: List[str] = []
        attempts = 0
        limit = max(1, n)
        while len(results) < limit and attempts < limit * 3:
            attempts += 1
            suffix = random.choice(self.suffixes)
            random_token = random.randint(0, 9999)
            candidate = f"{text}{suffix}{random_token:04d}"
            if candidate not in results:
                results.append(candidate)
        return results or [text]


class NoisyPassthroughAugmenter:
    """Produces repeated candidates including invalid ones to test filtering."""

    def augment(self, text: str, n: int = 1) -> List[str]:
        return [text, text + " ", text + "***", text.lower()]
