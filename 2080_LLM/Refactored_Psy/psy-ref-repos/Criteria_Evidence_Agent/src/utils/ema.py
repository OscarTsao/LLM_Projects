"""Exponential Moving Average for model parameters."""

import torch
from torch import nn


class EMA:
    """Exponential Moving Average of model parameters.

    This maintains a shadow copy of model parameters and updates them
    using exponential moving average. Useful for stabilizing training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Initialize EMA.

        Args:
            model: The model to track
            decay: EMA decay rate (0.9999 means 99.99% old, 0.01% new)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get state dict for saving."""
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
