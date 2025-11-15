"""Model components for multi-label classification."""

from .encoder_factory import build_encoder
from .model import EvidenceModel

__all__ = ["build_encoder", "EvidenceModel"]
