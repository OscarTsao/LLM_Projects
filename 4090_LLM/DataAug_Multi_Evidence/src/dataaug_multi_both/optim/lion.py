"""Minimal Lion optimizer implementation.

Based on the algorithm described in https://arxiv.org/abs/2302.06675.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] <= 1.0 or not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.add_(torch.sign(update), alpha=-lr)

        return loss

