from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    r"""Implements the Lion optimizer from https://arxiv.org/abs/2302.06675."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group.get("weight_decay", 0.0)

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad

                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                # First update using second momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                update = exp_avg.sign()

                if weight_decay != 0:
                    param.add_(param, alpha=-lr * weight_decay)

                param.add_(update, alpha=-lr)

                # Primary moving average update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        return loss
