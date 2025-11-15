from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from .psya_agent.train_utils import run_training


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    outputs = run_training(cfg, save_artifacts=True)
    logging.info("Validation metrics: %s", outputs.val_metrics)
    logging.info("Test metrics: %s", outputs.test_metrics)
    logging.info("Artifacts stored in: %s", outputs.artifact_dir)


if __name__ == "__main__":
    main()
