"""High-level interface for training and inference of the Evidence agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.evidence.train_pairclf import run_training
from src.evidence.infer_pairclf import run_inference


class EvidenceAgent:
    def __init__(self, config_path: Path = Path("configs/evidence/pairclf.yaml")) -> None:
        self.config_path = Path(config_path)

    def train(
        self,
        dataset,
        output_dir: Path,
        seed: Optional[int] = None,
        dry_run: bool = False,
        hparams: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return run_training(
            dataset,
            config_path=self.config_path,
            output_dir=output_dir,
            seed=seed,
            dry_run=dry_run,
            hparams=hparams,
        )

    def infer(
        self,
        dataset,
        checkpoint_path: Path,
        output_path: Path,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return run_inference(
            dataset,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            seed=seed,
        )


__all__ = ["EvidenceAgent"]
