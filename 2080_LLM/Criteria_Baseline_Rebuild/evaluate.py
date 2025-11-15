"""Hydra-powered CLI to evaluate a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src import training


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    metrics = training.evaluate_checkpoint(cfg)
    checkpoint_path = Path(str(cfg.evaluation.checkpoint)).expanduser()

    print("=== Evaluation ===")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
