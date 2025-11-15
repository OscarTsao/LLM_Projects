"""Optimizer factory for HPO with advanced optimizers.

Provides unified interface for creating optimizers including:
- Standard: AdamW, Adam
- Transformers: Adafactor
- Advanced: Lion, LAMB, AdamW-8bit

Usage:
    optimizer = create_optimizer(
        name="lamb",
        model_parameters=model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torch import optim

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.nn import Parameter

LOGGER = logging.getLogger(__name__)


def create_optimizer(
    name: str,
    model_parameters: Iterable[Parameter],
    lr: float,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> optim.Optimizer:
    """Create optimizer by name with fallback for missing dependencies.

    Args:
        name: Optimizer name (adamw, adam, adafactor, lion, lamb, adamw_8bit)
        model_parameters: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer name is unknown
        ImportError: If optional dependency is missing (with helpful message)
    """
    name = name.lower().strip()

    # Standard PyTorch optimizers
    if name == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

    if name == "adam":
        return optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
        )

    # Transformers Adafactor
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor

            return Adafactor(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=kwargs.get("scale_parameter", False),
                relative_step=kwargs.get("relative_step", False),
                warmup_init=kwargs.get("warmup_init", False),
            )
        except ImportError as exc:
            raise ImportError(
                "Adafactor requires transformers: pip install transformers"
            ) from exc

    # Lion optimizer (memory-efficient, strong performance)
    elif name == "lion":
        try:
            from lion_pytorch import Lion

            return Lion(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.99)),
            )
        except ImportError:
            LOGGER.warning(
                "Lion optimizer not available, falling back to AdamW. "
                "Install with: pip install lion-pytorch"
            )
            return optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
            )

    # LAMB optimizer (Layer-wise Adaptive Moments, good for large batches)
    elif name == "lamb":
        try:
            # PyTorch 2.1+ has experimental LAMB support
            from torch.optim import LAMB  # type: ignore[attr-defined]

            return LAMB(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-6),
            )
        except (ImportError, AttributeError):
            # Fallback: AdamW with LAMB-like settings
            LOGGER.warning(
                "LAMB optimizer not available (requires PyTorch 2.1+), "
                "falling back to AdamW with LAMB-like settings"
            )
            return optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-6,
            )

    # AdamW 8-bit (memory-efficient via quantization)
    elif name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
            )
        except ImportError:
            LOGGER.warning(
                "bitsandbytes not available, falling back to standard AdamW. "
                "Install with: pip install bitsandbytes"
            )
            return optim.AdamW(
                model_parameters,
                lr=lr,
                weight_decay=weight_decay,
            )

    else:
        raise ValueError(
            f"Unknown optimizer: {name}. "
            f"Supported: adamw, adam, adafactor, lion, lamb, adamw_8bit"
        )


def get_optimizer_info(name: str) -> dict[str, Any]:
    """Get metadata about an optimizer.

    Args:
        name: Optimizer name

    Returns:
        Dictionary with optimizer metadata:
        - name: Canonical optimizer name
        - memory_efficient: Whether optimizer uses less memory
        - recommended_lr: Suggested learning rate range
        - requires: Optional dependency package name
        - notes: Usage notes and recommendations
    """
    name = name.lower().strip()

    optimizer_info = {
        "adamw": {
            "name": "AdamW",
            "memory_efficient": False,
            "recommended_lr": (1e-5, 5e-5),
            "requires": None,
            "notes": "Standard choice for transformers. Reliable and well-tested.",
        },
        "adam": {
            "name": "Adam",
            "memory_efficient": False,
            "recommended_lr": (1e-5, 5e-5),
            "requires": None,
            "notes": "Classic optimizer. AdamW often preferred for transformers.",
        },
        "adafactor": {
            "name": "Adafactor",
            "memory_efficient": True,
            "recommended_lr": (1e-3, 1e-2),
            "requires": "transformers",
            "notes": "Memory-efficient, good for large models. Uses higher LR.",
        },
        "lion": {
            "name": "Lion",
            "memory_efficient": True,
            "recommended_lr": (1e-5, 1e-4),
            "requires": "lion-pytorch",
            "notes": "Memory-efficient with strong performance. Use ~10x lower LR than Adam.",
        },
        "lamb": {
            "name": "LAMB",
            "memory_efficient": False,
            "recommended_lr": (1e-4, 1e-3),
            "requires": "torch>=2.1 (optional)",
            "notes": "Layer-wise adaptive moments. Good for large batch sizes.",
        },
        "adamw_8bit": {
            "name": "AdamW-8bit",
            "memory_efficient": True,
            "recommended_lr": (1e-5, 5e-5),
            "requires": "bitsandbytes",
            "notes": "Quantized AdamW. Saves ~75% optimizer memory.",
        },
    }

    return optimizer_info.get(
        name,
        {
            "name": name,
            "memory_efficient": False,
            "recommended_lr": (1e-5, 5e-5),
            "requires": "unknown",
            "notes": "Unknown optimizer",
        },
    )


def list_available_optimizers() -> list[str]:
    """List all optimizer names that can be used.

    Returns:
        List of optimizer names
    """
    return ["adamw", "adam", "adafactor", "lion", "lamb", "adamw_8bit"]


def check_optimizer_available(name: str) -> tuple[bool, str | None]:
    """Check if an optimizer and its dependencies are available.

    Args:
        name: Optimizer name

    Returns:
        Tuple of (is_available, error_message)
        - is_available: True if optimizer can be created
        - error_message: None if available, otherwise error description
    """
    name = name.lower().strip()

    if name in {"adamw", "adam"}:
        return True, None

    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor  # noqa: F401

            return True, None
        except ImportError:
            return False, "Requires: pip install transformers"

    if name == "lion":
        try:
            from lion_pytorch import Lion  # noqa: F401

            return True, None
        except ImportError:
            return False, "Requires: pip install lion-pytorch (will fallback to AdamW)"

    if name == "lamb":
        try:
            from torch.optim import LAMB  # type: ignore[attr-defined] # noqa: F401

            return True, None
        except (ImportError, AttributeError):
            return False, "Requires PyTorch 2.1+ (will fallback to AdamW)"

    if name == "adamw_8bit":
        try:
            import bitsandbytes  # noqa: F401

            return True, None
        except ImportError:
            return (
                False,
                "Requires: pip install bitsandbytes (will fallback to AdamW)",
            )

    return False, f"Unknown optimizer: {name}"


__all__ = [
    "create_optimizer",
    "get_optimizer_info",
    "list_available_optimizers",
    "check_optimizer_available",
]
