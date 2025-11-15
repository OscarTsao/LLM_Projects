"""Shared type aliases for type checking."""

from pathlib import Path
from typing import Any, Union

# Config types
ConfigDict = dict[str, Any]
PathLike = Union[str, Path]

# Model types
ModelOutput = dict[str, Any]
BatchDict = dict[str, Any]

# Training types
MetricsDict = dict[str, float]

__all__ = [
    "ConfigDict",
    "PathLike",
    "ModelOutput",
    "BatchDict",
    "MetricsDict",
]
