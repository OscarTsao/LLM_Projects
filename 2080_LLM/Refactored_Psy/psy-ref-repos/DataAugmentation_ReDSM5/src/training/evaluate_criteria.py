"""Evaluation script for criteria matching agent (Mode 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.agents.criteria_matching import CriteriaMatchingAgent
from src.training.data_module import DataModule, DataModuleConfig
from src.training.dataset_builder import build_splits
from src.training.engine import evaluate, save_json, set_global_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Evaluate criteria matching agent on specified split."""
    set_global_seed(cfg.seed)
    
    # Build datasets
    splits = build_splits(cfg.dataset)
    data_module = DataModule(
        split=splits,
        config=DataModuleConfig(
            tokenizer_name=cfg.model.pretrained_model_name,
            max_seq_length=cfg.model.max_seq_length,
            batch_size=cfg.model.batch_size,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            persistent_workers=cfg.dataloader.persistent_workers,
            prefetch_factor=cfg.dataloader.prefetch_factor,
        ),
    )
    
    # Select split
    split_name = cfg.evaluation.split
    if split_name == "train":
        dataloader = data_module.train_dataloader()
    elif split_name == "val":
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()
    
    # Load checkpoint
    checkpoint_path = cfg.evaluation.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.output_dir) / "best" / "model.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create agent and load state
    from src.training.train_criteria import create_criteria_agent
    agent = create_criteria_agent(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.to(device)
    
    # Evaluate
    metrics = evaluate(agent, dataloader, device)
    
    # Save results
    output_path = Path(cfg.output_dir) / cfg.evaluation.output_path
    save_json(output_path, metrics)
    logger.info(f"Criteria agent evaluation metrics saved to {output_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
