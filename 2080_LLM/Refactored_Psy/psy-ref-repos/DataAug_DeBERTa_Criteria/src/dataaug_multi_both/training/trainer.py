from __future__ import annotations

import copy
import logging
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

try:  # pragma: no cover - torch is an optional dependency in some environments
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

from .checkpoint_manager import CheckpointManager, StorageStats

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainerConfig:
    trial_id: str
    optimization_metric: str
    seed: int = 1337
    gradient_accumulation_steps: int = 1
    max_epochs: int = 1
    log_seeds_to_mlflow: bool = True
    resume_if_available: bool = True


@dataclass(slots=True)
class TrainingState:
    epoch: int
    global_step: int
    best_metric: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def seed_everything(base_seed: int) -> dict[str, int]:
    """Seed Python, NumPy, and PyTorch for deterministic executions."""

    random.seed(base_seed)
    np.random.seed(base_seed % (2**32 - 1))

    torch_seed = None
    cuda_seed = None
    if torch is not None:
        torch.manual_seed(base_seed)
        torch_seed = base_seed
        if torch.cuda.is_available():  # pragma: no cover - depends on CI capabilities
            torch.cuda.manual_seed_all(base_seed)
            cuda_seed = base_seed
        # Use warn_only=True to avoid errors from non-deterministic ops
        # Note: CUBLAS_WORKSPACE_CONFIG must be set before torch import (done in CLI entry point)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Disable benchmarking for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seeds = {
        "python": base_seed,
        "numpy": base_seed % (2**32 - 1),
    }
    if torch_seed is not None:
        seeds["torch"] = torch_seed
    if cuda_seed is not None:
        seeds["torch_cuda"] = cuda_seed
    return seeds


def build_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """Create a deterministic worker init function for PyTorch DataLoader."""

    def _init_fn(worker_id: int) -> None:
        worker_seed = (base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        if torch is not None:
            torch.manual_seed(worker_seed)

    return _init_fn


class Trainer:
    """Minimal trainer scaffold focusing on checkpoint resume and deterministic seeding."""

    def __init__(
        self,
        config: TrainerConfig,
        checkpoint_manager: CheckpointManager,
        logger: logging.Logger | None = None,
        mlflow_client: object | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger or LOGGER
        self.mlflow_client = mlflow_client
        self.worker_init_fn = build_worker_init_fn(config.seed)
        self._resume_lock = threading.Lock()
        self._cached_state: TrainingState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prepare(self) -> TrainingState:
        """Prepare the training session: seed RNGs and optionally resume state."""

        seeds = seed_everything(self.config.seed)
        self._record_seeds(seeds)

        if not self.config.resume_if_available:
            self._cached_state = TrainingState(epoch=0, global_step=0)
            return copy.deepcopy(self._cached_state)

        return self.resume()

    def resume(self) -> TrainingState:
        """Resume from the most recent checkpoint or initialize a zero state."""

        with self._resume_lock:
            if self._cached_state is not None:
                return copy.deepcopy(self._cached_state)

            try:
                record, payload = self.checkpoint_manager.load_latest_checkpoint()
                state = TrainingState(
                    epoch=int(payload.get("epoch", record.epoch)),
                    global_step=int(payload.get("global_step", 0)),
                    best_metric=payload.get("best_metric"),
                    metrics=dict(payload.get("metrics", {})),
                )
                self.logger.info(
                    "Resumed trial %s from checkpoint epoch=%s metric=%s",
                    self.config.trial_id,
                    state.epoch,
                    state.best_metric,
                )
            except FileNotFoundError:
                state = TrainingState(epoch=0, global_step=0)
                self.logger.info(
                    "No checkpoints found for trial %s; starting fresh.", self.config.trial_id
                )

            self._cached_state = state
            return copy.deepcopy(self._cached_state)

    def save_state(
        self,
        state: TrainingState,
        metric_value: float | None,
        storage_stats: StorageStats | None = None,
    ) -> None:
        """Persist the training state via the checkpoint manager."""

        state_payload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "metrics": state.metrics,
        }
        self.checkpoint_manager.save_checkpoint(
            state=state_payload,
            epoch=state.epoch,
            metric_value=metric_value,
            extra_metadata={"trial_id": self.config.trial_id},
            storage_stats=storage_stats,
        )
        self._cached_state = copy.deepcopy(state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_seeds(self, seeds: dict[str, int]) -> None:
        if not self.config.log_seeds_to_mlflow:
            return

        if self.mlflow_client is not None and hasattr(self.mlflow_client, "set_tag"):
            for name, value in seeds.items():
                self.mlflow_client.set_tag(f"seed.{name}", str(value))
            return

        try:  # pragma: no cover - optional dependency
            import mlflow

            if mlflow.active_run() is not None:
                for name, value in seeds.items():
                    mlflow.set_tag(f"seed.{name}", str(value))
        except ModuleNotFoundError:
            self.logger.debug("MLflow not installed; skipping seed tagging.")


__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "seed_everything",
    "build_worker_init_fn",
]
