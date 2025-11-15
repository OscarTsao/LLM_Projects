import logging
from typing import Tuple

from omegaconf import DictConfig
from transformers import RobertaConfig, RobertaModel

logger = logging.getLogger(__name__)


def load_encoder(cfg: DictConfig) -> Tuple[RobertaModel, int]:
    """Load a RoBERTa encoder with optional overrides."""
    config = RobertaConfig.from_pretrained(cfg.pretrained_model_name_or_path)
    if cfg.get("layer_norm_eps") is not None:
        config.layer_norm_eps = cfg.layer_norm_eps

    model = RobertaModel.from_pretrained(cfg.pretrained_model_name_or_path, config=config)
    if cfg.get("gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
            logger.debug(f"Gradient checkpointing enabled for {cfg.pretrained_model_name_or_path}")
        except ValueError as e:
            logger.warning(
                f"Gradient checkpointing requested but not supported by {cfg.pretrained_model_name_or_path}: {e}. "
                f"Continuing without gradient checkpointing."
            )

    return model, config.hidden_size
