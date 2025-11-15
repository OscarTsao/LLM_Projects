import logging
from typing import Tuple

from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


def _resolve_hidden_size(config: PretrainedConfig) -> int:
    """Resolve hidden size from XLNet config."""
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "d_model"):
        return int(config.d_model)
    raise AttributeError("Configuration does not expose hidden_size or d_model.")


def load_encoder(cfg: DictConfig) -> Tuple[PreTrainedModel, int]:
    """Load an XLNet encoder with XLNet-specific handling.
    
    XLNet models have specific requirements:
    - Do not support gradient checkpointing
    - Use relative attention which has different memory characteristics
    - May have different configuration parameter names
    """
    config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
    if cfg.get("layer_norm_eps") is not None and hasattr(config, "layer_norm_eps"):
        config.layer_norm_eps = cfg.layer_norm_eps

    model = AutoModel.from_pretrained(cfg.pretrained_model_name_or_path, config=config)
    
    # XLNet models do not support gradient checkpointing
    if cfg.get("gradient_checkpointing", False):
        logger.warning(
            f"Gradient checkpointing requested for XLNet model {cfg.pretrained_model_name_or_path}, "
            f"but XLNet models do not support gradient checkpointing due to their relative attention mechanism. "
            f"Continuing without gradient checkpointing."
        )

    hidden_size = _resolve_hidden_size(model.config)
    return model, hidden_size
