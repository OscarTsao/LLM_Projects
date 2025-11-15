"""Standalone evaluation script for trained checkpoints."""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .data_module import DataModule, DataModuleConfig
from .dataset_builder import build_splits
from .engine import evaluate, save_json, set_global_seed
from .modeling import BertPairClassifier, ModelConfig


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)

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

    split_name = cfg.evaluation.split
    if split_name == "train":
        dataloader = data_module.train_dataloader()
    elif split_name == "val":
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    model = BertPairClassifier(
        ModelConfig(
            pretrained_model_name=cfg.model.pretrained_model_name,
            classifier_hidden_sizes=cfg.model.classifier_hidden_sizes,
            dropout=cfg.model.classifier_dropout,
            num_labels=2,
        )
    )

    checkpoint_path = cfg.evaluation.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(cfg.output_dir) / "best" / "model.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    metrics = evaluate(model, dataloader, device)
    output_path = Path(cfg.output_dir) / cfg.evaluation.output_path
    save_json(output_path, metrics)
    print(f"Evaluation metrics saved to {output_path}")


if __name__ == "__main__":
    main()
