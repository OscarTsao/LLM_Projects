"""Augmentation helpers for evidence sentences."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from dataaug_multi_both.hpo.space import decode_mask


def _clamp_prob(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _compute_change_count(tokens: list[str], prob: float, tpe: int, rng: random.Random) -> int:
    base = int(_clamp_prob(prob) * len(tokens))
    if tpe > 0:
        base += rng.randint(0, tpe)
    return max(1, min(len(tokens), base)) if tokens else 0


def _remove_random_tokens(tokens: list[str], count: int, rng: random.Random) -> list[str]:
    if count <= 0 or len(tokens) <= 1:
        return tokens
    indices = rng.sample(range(len(tokens)), min(count, len(tokens) - 1))
    return [tok for idx, tok in enumerate(tokens) if idx not in indices]


def _swap_characters(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    idx = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def _replace_token(tokens: list[str], idx: int, replacement: str) -> None:
    tokens[idx] = replacement


def _simple_synonym(token: str) -> str:
    mapping = {
        "good": "great",
        "bad": "poor",
        "happy": "joyful",
        "sad": "sorrowful",
        "anxiety": "worry",
        "sleep": "rest",
        "energy": "vitality",
        "hope": "optimism",
    }
    return mapping.get(token.lower(), token)


def _apply_simple(method: str, text: str, rng: random.Random, prob: float, tpe: int) -> str:
    tokens = text.split()
    if not tokens:
        return text

    count = _compute_change_count(tokens, prob, tpe, rng)

    if method == "EDA":
        augmented = _remove_random_tokens(tokens, count, rng)
        if not augmented:
            augmented = tokens
        return " ".join(augmented)

    if method == "CharSwap":
        for _ in range(count):
            idx = rng.randrange(len(tokens))
            tokens[idx] = _swap_characters(tokens[idx], rng)
        return " ".join(tokens)

    if method in {"Embedding", "CLARE"}:
        for _ in range(count):
            idx = rng.randrange(len(tokens))
            synonym = _simple_synonym(tokens[idx])
            _replace_token(tokens, idx, synonym)
        return " ".join(tokens)

    if method == "BackTranslation":
        if len(tokens) > 3:
            start = rng.randrange(1, len(tokens) // 2)
            end = min(len(tokens), start + count)
            segment = list(reversed(tokens[start:end]))
            tokens[start:end] = segment
        return " ".join(tokens)

    return text


def _apply_attack(method: str, text: str, rng: random.Random, prob: float, tpe: int) -> str:
    tokens = text.split()
    if not tokens:
        return text
    count = _compute_change_count(tokens, prob, tpe, rng)
    for _ in range(count):
        idx = rng.randrange(len(tokens))
        token = tokens[idx]
        if method in {"TextFoolerJin2019", "PWWSRen2019", "BAEGarg2019"}:
            tokens[idx] = token[::-1]
        elif method in {"DeepWordBugGao2018", "HotFlipEbrahimi2017"}:
            tokens[idx] = token.upper()
        elif method in {"IGAWang2019", "Kuleshov2017"}:
            tokens[idx] = f"{token}{rng.choice(['!', '?', '...'])}"
        else:
            tokens[idx] = token
    return " ".join(tokens)


@dataclass
class SimpleAugConfig:
    name: str
    prob: float
    tpe: int


class EvidenceAugmenter:
    def __init__(
        self,
        simple_methods: Sequence[SimpleAugConfig],
        simple_mode: str,
        attack_method: SimpleAugConfig | None,
        cross_compose: str,
        rng: random.Random,
    ) -> None:
        self.simple_methods = list(simple_methods)
        self.simple_mode = simple_mode
        self.attack_method = attack_method
        self.cross_compose = cross_compose
        self.rng = rng

    def __call__(self, text: str) -> str:
        simple_applied = text
        if self.simple_methods:
            if self.simple_mode == "one_of":
                cfg = self.rng.choice(self.simple_methods)
                simple_applied = _apply_simple(cfg.name, text, self.rng, cfg.prob, cfg.tpe)
            else:  # sequence
                simple_applied = text
                for cfg in self.simple_methods:
                    simple_applied = _apply_simple(cfg.name, simple_applied, self.rng, cfg.prob, cfg.tpe)
        attack_applied = simple_applied
        if self.attack_method:
            attack_applied = _apply_attack(
                self.attack_method.name,
                simple_applied if self.cross_compose != "attack_only" else text,
                self.rng,
                self.attack_method.prob,
                self.attack_method.tpe,
            )

        mode = self.cross_compose
        if not self.simple_methods and mode in {"serial_then_attack", "serial_only"}:
            mode = "attack_only" if self.attack_method else "serial_only"
        if not self.attack_method and mode in {"attack_then_serial", "attack_only"}:
            mode = "serial_only"

        if mode == "serial_then_attack":
            return attack_applied
        if mode == "attack_then_serial" and self.attack_method:
            attacked = _apply_attack(
                self.attack_method.name,
                text,
                self.rng,
                self.attack_method.prob,
                self.attack_method.tpe,
            )
            if self.simple_methods:
                return self.__class__(self.simple_methods, self.simple_mode, None, "serial_only", self.rng)(attacked)
            return attacked
        if mode == "attack_only" and self.attack_method:
            return attack_applied
        return simple_applied


def create_augmenter(params: dict[str, Any], rng: random.Random) -> EvidenceAugmenter | None:  # type: ignore[name-defined]
    simple_mask = decode_mask(params.get("aug_simple_mask", "none"))
    ta_mask = decode_mask(params.get("aug_ta_mask", "none"))

    simple_methods: list[SimpleAugConfig] = []
    for name in simple_mask:
        simple_methods.append(
            SimpleAugConfig(
                name=name,
                prob=float(params.get(f"aug_simple_{name}_p_token", 0.1)),
                tpe=int(params.get(f"aug_simple_{name}_tpe", 0)),
            )
        )

    attack_cfg: SimpleAugConfig | None = None
    if ta_mask:
        attack_name = ta_mask[0]
        attack_cfg = SimpleAugConfig(
            name=attack_name,
            prob=float(params.get("aug_ta_p_token", 0.1)),
            tpe=int(params.get("aug_ta_tpe", 0)),
        )

    if not simple_methods and not attack_cfg:
        return None

    simple_mode = params.get("aug_simple_compose", "one_of")
    cross_mode = params.get("aug_cross_family", "serial_only")

    return EvidenceAugmenter(simple_methods, simple_mode, attack_cfg, cross_mode, rng)


class AugmentedDataset:
    def __init__(self, dataset, augmenter: EvidenceAugmenter, field: str = "sentence_text") -> None:
        self.dataset = dataset
        self.augmenter = augmenter
        self.field = field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if isinstance(sample, dict) and self.field in sample:
            sample = dict(sample)
            sample[self.field] = self.augmenter(sample[self.field])
        return sample


__all__ = ["create_augmenter", "AugmentedDataset", "EvidenceAugmenter"]
