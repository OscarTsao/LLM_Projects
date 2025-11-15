"""Utility helpers for timestamped artifact naming."""
from __future__ import annotations

from datetime import datetime, timezone


def utc_timestamp(compact: bool = True) -> str:
    """Return a deterministic UTC timestamp suitable for artifact names."""
    fmt = "%Y%m%dT%H%M%SZ" if compact else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.now(timezone.utc).strftime(fmt)

