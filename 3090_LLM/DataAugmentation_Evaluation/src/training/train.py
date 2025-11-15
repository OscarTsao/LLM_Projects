"""Standard training entrypoint without Optuna."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from .engine import train_model


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    result = train_model(cfg)
    best_metric = result.get("best_metric")
    output_dir: Path = result.get("output_dir")
    if best_metric is not None:
        print(f"Best {cfg.metric_for_best_model}: {best_metric:.4f}")
    print(f"Artifacts stored in {output_dir}")


if __name__ == "__main__":
    main()
