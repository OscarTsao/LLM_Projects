from typing import Tuple

from omegaconf import DictConfig
from transformers import PreTrainedModel

from . import bert_encoder, deberta_encoder, roberta_encoder


ENCODER_LOADERS = {
    "bert": bert_encoder.load_encoder,
    "deberta": deberta_encoder.load_encoder,
    "roberta": roberta_encoder.load_encoder,
}


def build_encoder(cfg: DictConfig) -> Tuple[PreTrainedModel, int]:
    """Instantiate encoder based on the provided configuration."""
    encoder_type = cfg.get("type", "roberta")
    if encoder_type not in ENCODER_LOADERS:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    loader = ENCODER_LOADERS[encoder_type]
    model, hidden_size = loader(cfg)
    return model, hidden_size

