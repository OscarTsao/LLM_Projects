from __future__ import annotations

from typing import Dict, Tuple


DEBERTA_V3_FAMILY = {
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
}


def resolve_model_config(model_name: str | None, max_length: int | None) -> Tuple[str, int, Dict[str, str]]:
    """Resolve model and sequence length with safety caps and warnings.

    - Default to microsoft/deberta-v3-base if model_name is None.
    - Cap max_length at 512 for DeBERTa-v3 family; return warning in tags.
    """
    tags: Dict[str, str] = {}
    resolved = model_name or "microsoft/deberta-v3-base"
    length = int(max_length or 256)
    if resolved in DEBERTA_V3_FAMILY and length > 512:
        tags["max_length_capped"] = "true"
        length = 512
    return resolved, length, tags

