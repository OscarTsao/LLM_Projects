from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def update(self, value: float):
        if self.best is None:
            self.best = value
            return
        improve = (value - self.best) if self.mode == "max" else (self.best - value)
        if improve > self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


__all__ = ["EarlyStopping"]
