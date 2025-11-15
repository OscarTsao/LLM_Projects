"""Evaluation script for evidence binding agent (Mode 2)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.agents.evidence_binding import EvidenceBindingAgent
from src.data.joint_dataset import build_joint_datasets, JointCollator
from src.training.engine import save_json, set_global_seed
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Evaluate evidence binding agent on specified split."""
    set_global_seed(cfg.seed)
    
    # Build datasets
    train_dataset, val_dataset, test_dataset = build_joint_datasets(cfg.dataset)
    
    # Select split
    split_name = cfg.evaluation.split
    if split_name == "train":
        dataset = train_dataset
    elif split_name == "val":
        dataset = val_dataset
    else:
        dataset = test_dataset
    
    # Create collator
    collator = JointCollator(
        tokenizer_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )
    
    # Load checkpoint
    checkpoint_path = cfg.evaluation.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.output_dir) / "best" / "model.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Create agent and load state
    from src.training.train_evidence import create_evidence_agent
    agent = create_evidence_agent(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.to(device)
    
    # Evaluate (simplified - implement proper evidence evaluation)
    agent.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = agent(**batch)
            total_loss += outputs["loss"].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = {"loss": avg_loss}
    
    # Save results
    output_path = Path(cfg.output_dir) / cfg.evaluation.output_path
    save_json(output_path, metrics)
    logger.info(f"Evidence agent evaluation metrics saved to {output_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
