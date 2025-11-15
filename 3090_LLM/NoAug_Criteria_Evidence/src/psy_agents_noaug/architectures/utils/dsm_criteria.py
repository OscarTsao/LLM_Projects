"""Helpers for loading and resolving DSM criteria text."""

from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path

DEFAULT_DSM_CRITERIA_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "raw"
    / "redsm5"
    / "dsm_criteria.json"
)


@cache
def _load_raw_criteria(path: str | Path) -> Iterable[dict[str, str]]:
    """Load raw criteria entries from JSON."""
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"DSM criteria file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"DSM criteria JSON must be a list, got {type(data)}")

    for entry in data:
        if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
            raise ValueError(
                "Each DSM criterion entry must be an object with 'id' and 'text' fields"
            )
        yield entry


@cache
def build_criterion_text_map(
    path: str | Path = DEFAULT_DSM_CRITERIA_PATH,
) -> dict[str, str]:
    """Return a mapping from criterion identifier to its descriptive text."""
    mapping: dict[str, str] = {}
    for entry in _load_raw_criteria(path):
        identifier = str(entry["id"]).strip()
        text = str(entry.get("text", "")).strip()
        if not identifier or not text:
            continue
        candidates = {
            identifier,
            identifier.upper(),
            identifier.lower(),
        }
        for key in candidates:
            mapping[key] = text
    return mapping


def resolve_criterion_text(
    criterion_id: str | int,
    mapping: dict[str, str],
    *,
    fallback: str | None = None,
) -> str:
    """Resolve criterion identifier to descriptive text.

    Args:
        criterion_id: Identifier present in the dataset row.
        mapping: Lookup produced by ``build_criterion_text_map``.
        fallback: Optional string to use when no direct mapping is available.

    Returns:
        Human-readable criterion text suitable for pairing with the post.
    """
    key = str(criterion_id).strip()
    if not key:
        if fallback:
            return fallback
        raise ValueError("Empty criterion identifier encountered")

    for candidate in (key, key.upper(), key.lower(), key.title()):
        if candidate in mapping:
            return mapping[candidate]

    if fallback:
        return fallback

    # Final fallback: prettify the identifier itself for tokenizer pairing.
    return key.replace("_", " ").strip() or key
