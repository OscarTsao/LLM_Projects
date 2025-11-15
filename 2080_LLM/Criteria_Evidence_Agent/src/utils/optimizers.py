"""Optimizer factory utilities."""

import torch
from omegaconf import DictConfig
from torch.optim import AdamW


def get_layerwise_lr_params(
    model: torch.nn.Module,
    learning_rate: float,
    layerwise_lr_decay: float = 1.0,
    weight_decay: float = 0.01,
) -> list:
    """Get parameter groups with layerwise learning rate decay.

    Args:
        model: PyTorch model
        learning_rate: Base learning rate
        layerwise_lr_decay: Decay factor per layer (1.0 = no decay)
        weight_decay: Weight decay for regularization

    Returns:
        List of parameter groups for optimizer
    """
    if layerwise_lr_decay == 1.0:
        # No decay, return all parameters with same LR
        return [{"params": filter(lambda p: p.requires_grad, model.parameters())}]

    param_groups = []
    no_decay_params = ["bias", "LayerNorm", "layer_norm"]

    # Get encoder layers
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if hasattr(encoder, "base_model"):  # For LoRA
            encoder = encoder.base_model

        # Get number of layers
        if hasattr(encoder, "encoder"):
            num_layers = len(encoder.encoder.layer)
        elif hasattr(encoder, "layers"):
            num_layers = len(encoder.layers)
        else:
            num_layers = 12  # Default for BERT-base

        # Create parameter groups for each layer
        for layer_idx in range(num_layers):
            lr_scale = layerwise_lr_decay ** (num_layers - layer_idx - 1)
            layer_lr = learning_rate * lr_scale

            # With weight decay
            layer_params_decay = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if f"layer.{layer_idx}." in n
                    and p.requires_grad
                    and not any(nd in n for nd in no_decay_params)
                ],
                "lr": layer_lr,
                "weight_decay": weight_decay,
            }

            # Without weight decay
            layer_params_no_decay = {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if f"layer.{layer_idx}." in n
                    and p.requires_grad
                    and any(nd in n for nd in no_decay_params)
                ],
                "lr": layer_lr,
                "weight_decay": 0.0,
            }

            if layer_params_decay["params"]:
                param_groups.append(layer_params_decay)
            if layer_params_no_decay["params"]:
                param_groups.append(layer_params_no_decay)

    # Embedding and pooling layers (base learning rate)
    other_params_decay = {
        "params": [
            p
            for n, p in model.named_parameters()
            if "layer." not in n
            and p.requires_grad
            and not any(nd in n for nd in no_decay_params)
        ],
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    other_params_no_decay = {
        "params": [
            p
            for n, p in model.named_parameters()
            if "layer." not in n and p.requires_grad and any(nd in n for nd in no_decay_params)
        ],
        "lr": learning_rate,
        "weight_decay": 0.0,
    }

    if other_params_decay["params"]:
        param_groups.append(other_params_decay)
    if other_params_no_decay["params"]:
        param_groups.append(other_params_no_decay)

    return param_groups


def get_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Get optimizer based on configuration.

    Args:
        model: PyTorch model
        optimizer_cfg: Optimizer configuration

    Returns:
        PyTorch optimizer

    Raises:
        ValueError: If optimizer name is not supported
    """
    name = optimizer_cfg.get("name", "adamw").lower()
    learning_rate = optimizer_cfg.get("learning_rate", 2e-5)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    eps = optimizer_cfg.get("eps", 1e-8)
    layerwise_lr_decay = optimizer_cfg.get("layerwise_lr_decay", 1.0)

    # Get parameter groups with optional layerwise LR decay
    param_groups = get_layerwise_lr_params(model, learning_rate, layerwise_lr_decay, weight_decay)

    if name == "adamw":
        return AdamW(
            param_groups,
            lr=learning_rate,
            eps=eps,
        )

    if name == "lamb":
        try:
            from torch_optimizer import Lamb
        except ImportError:
            raise ImportError(
                "LAMB optimizer requires torch-optimizer. "
                "Install with: pip install torch-optimizer"
            )
        return Lamb(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=eps,
        )

    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError:
            raise ImportError(
                "Adafactor optimizer requires transformers. "
                "It should already be installed."
            )
        return Adafactor(
            param_groups,
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
        )

    raise ValueError(f"Unsupported optimizer: {name}")
