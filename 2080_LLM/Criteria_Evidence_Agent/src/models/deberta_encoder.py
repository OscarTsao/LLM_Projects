from typing import Tuple

from omegaconf import DictConfig
from transformers import DebertaConfig, DebertaModel


def load_encoder(cfg: DictConfig) -> Tuple[DebertaModel, int]:
    """Load a DeBERTa encoder with optional overrides."""
    config = DebertaConfig.from_pretrained(cfg.pretrained_model_name_or_path)
    if cfg.get("layer_norm_eps") is not None:
        config.layer_norm_eps = cfg.layer_norm_eps

    model = DebertaModel.from_pretrained(cfg.pretrained_model_name_or_path, config=config)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model, config.hidden_size

