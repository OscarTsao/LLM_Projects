# File: src/training/callbacks.py
"""Training callbacks for early stopping and best model tracking."""

import torch
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ..utils.io import save_json, safe_mkdir

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "combined_score",
        mode: str = "max",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: "max" or "min" optimization
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.should_stop = False

        if mode == "max":
            self.is_better = lambda score, best: score > best + min_delta
            self.best_score = float('-inf')
        elif mode == "min":
            self.is_better = lambda score, best: score < best - min_delta
            self.best_score = float('inf')
        else:
            raise ValueError(f"Mode {mode} is unknown. Use 'max' or 'min'.")

    def __call__(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics

        Returns:
            True if training should stop
        """
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return False

        current_score = metrics[self.monitor]

        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            logger.info(
                f"Early stopping at epoch {epoch}. "
                f"Best {self.monitor}: {self.best_score:.4f}"
            )

        return self.should_stop


class ModelCheckpoint:
    """Save best k models during training."""

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "combined_score",
        mode: str = "max",
        save_top_k: int = 2,
        save_last: bool = True,
    ) -> None:
        """Initialize model checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: "max" or "min" optimization
            save_top_k: Number of best models to keep
            save_last: Whether to save the last checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.best_scores: List[Dict[str, Any]] = []

        safe_mkdir(self.checkpoint_dir)

        if mode == "max":
            self.is_better = lambda score, best: score > best
        elif mode == "min":
            self.is_better = lambda score, best: score < best
        else:
            raise ValueError(f"Mode {mode} is unknown. Use 'max' or 'min'.")

    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        tokenizer: Any,
        config: Dict[str, Any],
        label_mappings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save checkpoint if it's among the best.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics
            model: Model to save
            tokenizer: Tokenizer to save
            config: Configuration dictionary
            label_mappings: Optional label mappings
        """
        if self.monitor not in metrics:
            logger.warning(f"Monitor metric '{self.monitor}' not found in metrics")
            return

        current_score = metrics[self.monitor]

        # Check if this is a top-k score
        should_save = False
        if len(self.best_scores) < self.save_top_k:
            should_save = True
        else:
            worst_idx = self._get_worst_score_idx()
            if worst_idx is not None:
                worst_score = self.best_scores[worst_idx]["score"]
                if self.is_better(current_score, worst_score):
                    # Remove worst checkpoint
                    self._remove_checkpoint(self.best_scores[worst_idx])
                    self.best_scores.pop(worst_idx)
                    should_save = True

        if should_save:
            checkpoint_info = {
                "epoch": epoch,
                "score": current_score,
                "metrics": metrics.copy(),
                "checkpoint_dir": str(self.checkpoint_dir / f"epoch_{epoch}"),
            }

            self._save_checkpoint(
                checkpoint_info,
                model,
                tokenizer,
                config,
                label_mappings,
            )

            self.best_scores.append(checkpoint_info)
            logger.info(
                f"Saved checkpoint at epoch {epoch} with {self.monitor}: {current_score:.4f}"
            )

        # Save last checkpoint if requested
        if self.save_last:
            last_checkpoint_info = {
                "epoch": epoch,
                "score": current_score,
                "metrics": metrics.copy(),
                "checkpoint_dir": str(self.checkpoint_dir / "last"),
            }

            self._save_checkpoint(
                last_checkpoint_info,
                model,
                tokenizer,
                config,
                label_mappings,
            )

        # Save best checkpoint link
        if self.best_scores:
            best_checkpoint = max(self.best_scores, key=lambda x: x["score"])
            if self.mode == "min":
                best_checkpoint = min(self.best_scores, key=lambda x: x["score"])

            best_link = self.checkpoint_dir / "best"
            if best_link.exists():
                if best_link.is_symlink():
                    best_link.unlink()
                elif best_link.is_dir():
                    import shutil
                    shutil.rmtree(best_link)

            # Create symlink to best checkpoint
            try:
                best_link.symlink_to(Path(best_checkpoint["checkpoint_dir"]).name)
            except OSError:
                # Fallback: copy instead of symlink
                import shutil
                shutil.copytree(best_checkpoint["checkpoint_dir"], best_link)

    def _get_worst_score_idx(self) -> Optional[int]:
        """Get index of worst score in best_scores list."""
        if not self.best_scores:
            return None

        if self.mode == "max":
            return min(range(len(self.best_scores)), key=lambda i: self.best_scores[i]["score"])
        else:
            return max(range(len(self.best_scores)), key=lambda i: self.best_scores[i]["score"])

    def _save_checkpoint(
        self,
        checkpoint_info: Dict[str, Any],
        model: torch.nn.Module,
        tokenizer: Any,
        config: Dict[str, Any],
        label_mappings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint to disk."""
        checkpoint_dir = Path(checkpoint_info["checkpoint_dir"])
        safe_mkdir(checkpoint_dir)

        # Save model state
        model_path = checkpoint_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)

        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        config_path = checkpoint_dir / "config.yaml"
        from ..utils.io import save_yaml
        save_yaml(config, config_path)

        # Save training info
        training_info = {
            "epoch": checkpoint_info["epoch"],
            "metrics": checkpoint_info["metrics"],
        }
        save_json(training_info, checkpoint_dir / "training_args.json")

        # Save label mappings
        if label_mappings:
            save_json(label_mappings, checkpoint_dir / "label_mappings.json")

    def _remove_checkpoint(self, checkpoint_info: Dict[str, Any]) -> None:
        """Remove a checkpoint directory."""
        checkpoint_dir = Path(checkpoint_info["checkpoint_dir"])
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            logger.info(f"Removed checkpoint: {checkpoint_dir}")

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_link = self.checkpoint_dir / "best"
        if best_link.exists():
            return best_link
        return None