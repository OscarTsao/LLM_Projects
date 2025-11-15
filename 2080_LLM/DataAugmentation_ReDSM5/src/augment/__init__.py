"""
Utilities for managing deterministic, evidence-only augmentation pipelines.

The modules exposed here provide:
    - MethodRegistry: parses YAML registry and instantiates augmenters.
    - EvidenceSpan utilities: locate and replace evidence spans with fidelity.
    - Combination utilities: deterministic combo generation and sharding helpers.
"""

from .methods import MethodRegistry, MethodSpec, load_method_specs
from .evidence import EvidenceMatch, EvidenceReplacer
from .combinator import ComboDescriptor, ComboGenerator

__all__ = [
    "MethodRegistry",
    "MethodSpec",
    "load_method_specs",
    "EvidenceMatch",
    "EvidenceReplacer",
    "ComboDescriptor",
    "ComboGenerator",
]
