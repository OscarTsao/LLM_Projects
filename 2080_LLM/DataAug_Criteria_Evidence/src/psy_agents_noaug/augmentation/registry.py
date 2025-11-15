"""Allowlisted registry of lightweight, on-the-fly augmentation methods."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from textattack.augmentation.recipes import (
    CharSwapAugmenter,
    CheckListAugmenter,
    DeletionAugmenter,
    EasyDataAugmenter,
    SwapAugmenter,
    SynonymInsertionAugmenter,
    WordNetAugmenter,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_METHODS: tuple[str, ...] = (
    # nlpaug (char)
    "nlpaug/char/KeyboardAug",
    "nlpaug/char/OcrAug",
    "nlpaug/char/RandomCharAug",
    # nlpaug (word)
    "nlpaug/word/RandomWordAug",
    "nlpaug/word/ReservedAug",
    "nlpaug/word/SpellingAug",
    "nlpaug/word/SplitAug",
    "nlpaug/word/SynonymAug(wordnet)",
    "nlpaug/word/AntonymAug(wordnet)",
    "nlpaug/word/TfIdfAug",
    # TextAttack
    "textattack/CharSwapAugmenter",
    "textattack/DeletionAugmenter",
    "textattack/SwapAugmenter",
    "textattack/SynonymInsertionAugmenter",
    "textattack/EasyDataAugmenter",
    "textattack/CheckListAugmenter",
    "textattack/WordNetAugmenter",
)

AUGMENTATION_BANLIST: tuple[str, ...] = (
    # nlpaug heavy augmenters
    "nlpaug/word/ContextualWordEmbsAug",
    "nlpaug/word/ContextualWordEmbsForSentenceAug",
    "nlpaug/word/BackTranslationAug",
    "nlpaug/summarizer/AbstSummAug",
    "nlpaug/word/LambadaAug",
    "nlpaug/word/WordEmbsAug",
    # TextAttack heavy augmenters
    "textattack/CLAREAugmenter",
    "textattack/BackTranslationAugmenter",
    "textattack/BackTranscriptionAugmenter",
    "textattack/EmbeddingAugmenter",
)

LEGACY_NAME_MAP = {
    "nlpaug/word/SynonymAug": "nlpaug/word/SynonymAug(wordnet)",
    "nlpaug/word/AntonymAug": "nlpaug/word/AntonymAug(wordnet)",
}


class AugmenterWrapper:
    """Normalise augmenter outputs to a single string result."""

    def __init__(self, name: str, augmenter: Any, *, returns_list: bool = False):
        self.name = name
        self._augmenter = augmenter
        self._returns_list = returns_list

    def augment_one(self, text: str) -> str:
        try:
            result = self._augmenter.augment(text)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Augmenter %s failed: %s", self.name, exc)
            return text

        if isinstance(result, str):
            return result or text
        if isinstance(result, list):
            for candidate in result:
                if isinstance(candidate, str) and candidate:
                    return candidate
            return text
        if result is None:
            return text
        return str(result)


def _load_reserved_tokens(reserved_map_path: str | Path) -> dict[str, str] | list[str]:
    path = Path(reserved_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Reserved map not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) or isinstance(raw, list):
        return raw
    raise ValueError("Reserved map must be dict or list")


AugmenterFactory = Callable[..., AugmenterWrapper]


@dataclass(frozen=True)
class RegistryEntry:
    lib: str
    factory: AugmenterFactory


def _wrap(
    factory: Callable[..., Any], *, returns_list: bool = False, name: str | None = None
) -> AugmenterFactory:
    def _builder(**kwargs: Any) -> AugmenterWrapper:
        augmenter = factory(**kwargs)
        augmenter_name = name or getattr(factory, "__name__", "augmenter")
        return AugmenterWrapper(augmenter_name, augmenter, returns_list=returns_list)

    return _builder


def _make_reserved(
    reserved_map_path: str | Path | None, **kwargs: Any
) -> AugmenterWrapper:
    if reserved_map_path is None:
        raise ValueError("reserved_map_path is required for ReservedAug")
    tokens = _load_reserved_tokens(reserved_map_path)
    augmenter = naw.ReservedAug(reserved_tokens=tokens, **kwargs)
    return AugmenterWrapper("ReservedAug", augmenter)


def _make_tfidf(model_path: str | Path | None, **kwargs: Any) -> AugmenterWrapper:
    if model_path is None:
        raise ValueError("model_path is required for TfIdfAug")
    augmenter = naw.TfIdfAug(model_path=str(model_path), **kwargs)
    return AugmenterWrapper("TfIdfAug", augmenter)


def _ensure_allowlist_consistency() -> None:
    for banned in AUGMENTATION_BANLIST:
        if banned in ALLOWED_METHODS:
            raise AssertionError(f"Banned augmenter {banned} present in allowlist")


_ensure_allowlist_consistency()

REGISTRY: dict[str, RegistryEntry] = {
    "nlpaug/char/KeyboardAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.KeyboardAug, name="KeyboardAug")
    ),
    "nlpaug/char/OcrAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.OcrAug, name="OcrAug")
    ),
    "nlpaug/char/RandomCharAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.RandomCharAug, name="RandomCharAug")
    ),
    "nlpaug/word/RandomWordAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.RandomWordAug, name="RandomWordAug")
    ),
    "nlpaug/word/ReservedAug": RegistryEntry(lib="nlpaug", factory=_make_reserved),
    "nlpaug/word/SpellingAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.SpellingAug, name="SpellingAug")
    ),
    "nlpaug/word/SplitAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.SplitAug, name="SplitAug")
    ),
    "nlpaug/word/SynonymAug(wordnet)": RegistryEntry(
        lib="nlpaug",
        factory=_wrap(
            lambda **kw: naw.SynonymAug(aug_src="wordnet", **kw),
            name="SynonymAug(wordnet)",
        ),
    ),
    "nlpaug/word/AntonymAug(wordnet)": RegistryEntry(
        lib="nlpaug",
        factory=_wrap(
            lambda **kw: naw.AntonymAug(**kw),
            name="AntonymAug(wordnet)",
        ),
    ),
    "nlpaug/word/TfIdfAug": RegistryEntry(lib="nlpaug", factory=_make_tfidf),
    "textattack/CharSwapAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(CharSwapAugmenter, returns_list=True, name="CharSwapAugmenter"),
    ),
    "textattack/DeletionAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(DeletionAugmenter, returns_list=True, name="DeletionAugmenter"),
    ),
    "textattack/SwapAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(SwapAugmenter, returns_list=True, name="SwapAugmenter"),
    ),
    "textattack/SynonymInsertionAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(
            SynonymInsertionAugmenter,
            returns_list=True,
            name="SynonymInsertionAugmenter",
        ),
    ),
    "textattack/EasyDataAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(EasyDataAugmenter, returns_list=True, name="EasyDataAugmenter"),
    ),
    "textattack/CheckListAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(CheckListAugmenter, returns_list=True, name="CheckListAugmenter"),
    ),
    "textattack/WordNetAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(WordNetAugmenter, returns_list=True, name="WordNetAugmenter"),
    ),
}

missing_allow = set(ALLOWED_METHODS) - set(REGISTRY.keys())
if missing_allow:
    raise RuntimeError(
        f"Augmentation registry missing allowlisted methods: {sorted(missing_allow)}"
    )

extra_registered = set(REGISTRY.keys()) - set(ALLOWED_METHODS)
if extra_registered:
    raise RuntimeError(
        f"Augmentation registry has non-allowlisted methods: {sorted(extra_registered)}"
    )

ALL_METHODS: list[str] = list(ALLOWED_METHODS)
NLPAUG_METHODS: list[str] = [
    name for name in ALL_METHODS if REGISTRY[name].lib == "nlpaug"
]
TEXTATTACK_METHODS: list[str] = [
    name for name in ALL_METHODS if REGISTRY[name].lib == "textattack"
]

__all__ = [
    "REGISTRY",
    "ALL_METHODS",
    "NLPAUG_METHODS",
    "TEXTATTACK_METHODS",
    "ALLOWED_METHODS",
    "AUGMENTATION_BANLIST",
    "LEGACY_NAME_MAP",
    "AugmenterWrapper",
    "RegistryEntry",
]
