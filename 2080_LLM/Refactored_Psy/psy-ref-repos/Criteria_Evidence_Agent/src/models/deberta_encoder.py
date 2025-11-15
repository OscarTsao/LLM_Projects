from typing import Tuple

from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig


def _resolve_hidden_size(config: PretrainedConfig) -> int:
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "d_model"):
        return int(config.d_model)
    raise AttributeError("Configuration does not expose hidden_size or d_model.")


def load_encoder(cfg: DictConfig) -> Tuple[PreTrainedModel, int]:
    """Load a DeBERTa-family encoder (including v2/v3 variants) with optional overrides."""
    config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
    if cfg.get("layer_norm_eps") is not None and hasattr(config, "layer_norm_eps"):
        config.layer_norm_eps = cfg.layer_norm_eps

    model = AutoModel.from_pretrained(cfg.pretrained_model_name_or_path, config=config)
    if cfg.get("gradient_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    hidden_size = _resolve_hidden_size(model.config)
    return model, hidden_size
