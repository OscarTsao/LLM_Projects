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
    named_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    if not named_params:
        return []

    param_groups = []
    consumed: set[str] = set()
    no_decay_terms = ("bias", "LayerNorm", "layer_norm")

    def split_params(
        params: list[tuple[str, torch.nn.Parameter]],
    ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        decay_list: list[torch.nn.Parameter] = []
        no_decay_list: list[torch.nn.Parameter] = []
        for name, parameter in params:
            if any(term in name for term in no_decay_terms):
                no_decay_list.append(parameter)
            else:
                decay_list.append(parameter)
        return decay_list, no_decay_list

    # Determine encoder layers if available
    if hasattr(model, "encoder"):
        encoder = model.encoder
        if hasattr(encoder, "base_model"):  # PEFT wraps encoders in base_model
            encoder = encoder.base_model

        num_layers = 0
        if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
            num_layers = len(encoder.encoder.layer)
        elif hasattr(encoder, "layers"):
            num_layers = len(encoder.layers)
        else:
            num_layers = 12  # Fallback for models exposing transformer blocks differently

        for layer_idx in range(num_layers):
            layer_prefix = f"layer.{layer_idx}."
            layer_named = [
                (name, param)
                for name, param in named_params
                if name not in consumed and layer_prefix in name
            ]
            if not layer_named:
                continue

            lr_scale = layerwise_lr_decay ** (num_layers - layer_idx - 1)
            layer_lr = learning_rate * lr_scale
            decay_params, no_decay_params = split_params(layer_named)

            if decay_params:
                param_groups.append(
                    {"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay}
                )
            if no_decay_params:
                param_groups.append(
                    {"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0}
                )
            consumed.update(name for name, _ in layer_named)

    remaining = [(name, param) for name, param in named_params if name not in consumed]
    if remaining:
        decay_params, no_decay_params = split_params(remaining)
        if decay_params:
            param_groups.append(
                {"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay}
            )
        if no_decay_params:
            param_groups.append(
                {"params": no_decay_params, "lr": learning_rate, "weight_decay": 0.0}
            )

    if not param_groups:
        param_groups.append(
            {
                "params": [param for _, param in named_params],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )

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
                "Adafactor optimizer requires transformers. " "It should already be installed."
            )
        return Adafactor(
            param_groups,
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
        )

    raise ValueError(f"Unsupported optimizer: {name}")
