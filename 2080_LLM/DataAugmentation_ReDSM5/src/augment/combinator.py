"""
Deterministic combination generator with sharding support.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, Sequence, Tuple


@dataclass(frozen=True)
class ComboDescriptor:
    """Description for a single augmentation combo."""

    methods: Tuple[str, ...]
    combo_id: str

    @property
    def size(self) -> int:
        return len(self.methods)

    @property
    def source_combo(self) -> str:
        return "+".join(self.methods)


class ComboGenerator:
    """
    Generate augmentation method combinations in a deterministic order.

    Combos are enumerated lexicographically by method ids. Optional sharding
    allows distributed execution where shard S of N retains combos whose index
    satisfies ``index % N == S``.
    """

    def __init__(self, method_ids: Sequence[str]) -> None:
        if not method_ids:
            raise ValueError("At least one method id is required.")
        self._method_ids: Tuple[str, ...] = tuple(sorted(method_ids))

    # ------------------------------------------------------------------ public
    def iter_combos(
        self,
        mode: str,
        max_combo_size: int | None = None,
        *,
        confirm_powerset: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> Iterator[ComboDescriptor]:
        """Yield combos according to the requested mode."""
        if num_shards <= 0:
            raise ValueError("num_shards must be >= 1.")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError("Shard index must satisfy 0 <= shard < num_shards.")

        combos: Iterable[Tuple[str, ...]]
        if mode == "singletons":
            combos = ((mid,) for mid in self._method_ids)
        elif mode == "bounded_k":
            if not max_combo_size or max_combo_size < 1:
                raise ValueError("max_combo_size must be >= 1 for bounded_k mode.")
            combos = _combinations_up_to(self._method_ids, max_combo_size)
        elif mode == "all":
            if not confirm_powerset:
                raise ValueError(
                    "combo-mode=all requires --confirm-powerset to acknowledge "
                    "the 2^N-1 combinations."
                )
            combos = _combinations_up_to(self._method_ids, len(self._method_ids))
        else:
            raise ValueError(f"Unsupported combo mode: {mode}")

        for index, methods in enumerate(combos):
            if index % num_shards != shard_index:
                continue
            yield ComboDescriptor(methods=methods, combo_id=_hash_combo(methods))


def _combinations_up_to(
    items: Sequence[str], max_size: int
) -> Iterable[Tuple[str, ...]]:
    for k in range(1, max_size + 1):
        for combo in combinations(items, k):
            yield combo


def _hash_combo(methods: Sequence[str]) -> str:
    joined = "+".join(methods)
    digest = hashlib.sha1(joined.encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:10]
