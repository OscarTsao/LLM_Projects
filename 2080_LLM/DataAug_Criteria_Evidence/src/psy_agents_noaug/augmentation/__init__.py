"""Augmentation package initialisation."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def _ensure_nltk_wordnet() -> None:
    try:
        import nltk
    except Exception as exc:  # pragma: no cover - safeguard for optional import
        LOGGER.warning("nltk unavailable, cannot download wordnet resources: %s", exc)
        return

    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as exc:  # pragma: no cover - network issues
                LOGGER.warning("Failed to download NLTK resource %s: %s", resource, exc)


_ensure_nltk_wordnet()

from .pipeline import (  # noqa: E402  (import after nltk setup)
    AugConfig,
    AugmenterPipeline,
    AugResources,
    is_enabled,
    worker_init,
)
from .registry import (  # noqa: E402
    ALL_METHODS,
    ALLOWED_METHODS,
    AUGMENTATION_BANLIST,
    LEGACY_NAME_MAP,
    NLPAUG_METHODS,
    REGISTRY,
    TEXTATTACK_METHODS,
)
from .tfidf_cache import TfidfResource, load_or_fit_tfidf  # noqa: E402

__all__ = [
    "AugConfig",
    "AugResources",
    "AugmenterPipeline",
    "is_enabled",
    "worker_init",
    "ALL_METHODS",
    "ALLOWED_METHODS",
    "AUGMENTATION_BANLIST",
    "NLPAUG_METHODS",
    "TEXTATTACK_METHODS",
    "REGISTRY",
    "LEGACY_NAME_MAP",
    "load_or_fit_tfidf",
    "TfidfResource",
]
