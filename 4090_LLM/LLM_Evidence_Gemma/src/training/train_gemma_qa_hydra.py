"""
Hydra-based training script for 5-fold CV on ReDSM5 evidence extraction.

Supports:
- Hydra configuration management
- Stratified 5-fold cross-validation
- Mixed precision training (bfloat16)
- Early stopping
- Comprehensive logging and experiment tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm.auto import tqdm
import logging
from contextlib import nullcontext
import math

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_qa import GemmaQA
from data.cv_splits import create_cv_splits, load_fold_split
from training.qa_metrics import extract_answers_batch, compute_metrics_batch
from utils.logger import setup_logger, log_experiment_config
from utils.experiment_tracking import ExperimentTracker
from utils.console_viz import render_training_progress

logger = logging.getLogger(__name__)


def load_fold_history_summary(fold_dir: Path, fold_idx: int) -> Optional[Dict[str, float]]:
    """
    Load cached metrics from a completed fold if history is available.

    Returns:
        Dict with best/final metrics if history exists, else None.
    """
    history_path = fold_dir / 'history.json'
    if not history_path.exists():
        return None

    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive parsing
        logger.warning("Failed to load history for fold %d from %s: %s", fold_idx, history_path, exc)
        return None

    val_em = history.get('val_em') or []
    val_f1 = history.get('val_f1') or []
    if not val_em or not val_f1:
        return None

    return {
        'fold': fold_idx,
        'best_val_em': max(val_em),
        'best_val_f1': max(val_f1),
        'final_val_em': val_em[-1],
        'final_val_f1': val_f1[-1],
    }


class FoldTrainer:
    """Trainer for single fold with early stopping and AMP."""

    def __init__(
        self,
        model: GemmaQA,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        fold_idx: int,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.device = device
        self.use_amp = bool(cfg.training.use_amp) and str(device).startswith('cuda')
        if cfg.training.use_amp and not self.use_amp:
            logger.warning("AMP requested but CUDA device unavailable; using full precision.")

        amp_dtype_name = str(getattr(cfg.training, 'amp_dtype', 'bfloat16')).lower()
        amp_dtype_lookup = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
        }
        if amp_dtype_name not in amp_dtype_lookup:
            logger.warning(
                "Unsupported AMP dtype '%s'; defaulting to bfloat16.",
                amp_dtype_name,
            )
        self.amp_dtype = amp_dtype_lookup.get(amp_dtype_name, torch.bfloat16)

        self.grad_accum_steps = max(
            1,
            int(getattr(cfg.training, 'gradient_accumulation_steps', 1)),
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found to optimize.")

        optimizer_request = str(getattr(cfg.training, 'optimizer', 'adamw')).lower()
        optimizer_aliases = {
            'adamw': 'adamw',
            'adamw8bit': 'adamw_8bit',
            'adamw_8bit': 'adamw_8bit',
        }
        optimizer_name = optimizer_aliases.get(optimizer_request)
        if optimizer_name is None:
            logger.warning(
                "Unsupported optimizer '%s'; defaulting to AdamW.",
                optimizer_request,
            )
            optimizer_name = 'adamw'

        if optimizer_name == 'adamw_8bit':
            try:
                from bitsandbytes.optim import AdamW8bit
            except ImportError as exc:
                raise ImportError(
                    "optimizer 'adamw_8bit' requires bitsandbytes>=0.43.0"
                ) from exc
            optimizer_cls = AdamW8bit
            logger.info("Using bitsandbytes AdamW8bit optimizer for reduced memory.")
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            trainable_params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Scheduler
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        total_steps = steps_per_epoch * cfg.training.num_epochs
        warmup_steps = int(total_steps * cfg.training.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision scaler
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        # Early stopping
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_em': [], 'val_f1': []}

    def train_epoch(self) -> float:
        """Train for one epoch with AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        progress_bar = tqdm(self.train_loader, desc=f'Fold {self.fold_idx} Training')
        for step, batch in enumerate(progress_bar, start=1):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            autocast_ctx = (
                torch.amp.autocast('cuda', dtype=self.amp_dtype)
                if self.use_amp else nullcontext()
            )
            with autocast_ctx:
                loss, _, _ = self.model(
                    input_ids, attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

            total_loss += loss.item()
            loss = loss / self.grad_accum_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step % self.grad_accum_steps == 0) or (step == num_batches)
            if should_step:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.max_grad_norm
                )
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            avg_loss = total_loss / step
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        all_predictions = []
        all_ground_truths = []
        total_loss = 0

        for batch in tqdm(self.val_loader, desc=f'Fold {self.fold_idx} Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            loss, start_logits, end_logits = self.model(
                input_ids, attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            total_loss += loss.item()

            # Extract answers
            start_logits_np = start_logits.float().cpu().numpy()
            end_logits_np = end_logits.float().cpu().numpy()
            input_ids_list = input_ids.cpu().tolist()

            predicted_answers = extract_answers_batch(
                start_logits_np, end_logits_np, input_ids_list, self.tokenizer
            )

            # Ground truth answers
            for i in range(len(input_ids)):
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()

                if start_idx == 0 and end_idx == 0:
                    ground_truth = ""
                else:
                    answer_tokens = input_ids[i][start_idx:end_idx + 1]
                    ground_truth = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                all_predictions.append(predicted_answers[i])
                all_ground_truths.append(ground_truth)

        metrics = compute_metrics_batch(all_predictions, all_ground_truths)
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def train(self, output_dir: Path) -> Dict:
        """Train with early stopping."""
        logger.info(f"Training fold {self.fold_idx}")

        for epoch in range(self.cfg.training.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Evaluate
            val_metrics = self.evaluate()
            val_f1 = val_metrics['f1']

            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_em'].append(val_metrics['exact_match'])
            self.history['val_f1'].append(val_f1)

            logger.info(
                f"Fold {self.fold_idx} Epoch {epoch + 1}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, "
                f"Val EM={val_metrics['exact_match']:.4f}, Val F1={val_f1:.4f}"
            )

            progress_panel = render_training_progress(self.history)
            if progress_panel:
                logger.info("\n" + progress_panel)

            # Early stopping check
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0

                # Save best checkpoint
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'metrics': val_metrics,
                    'model_kwargs': self.model.export_init_kwargs(),
                }, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save history
        history_path = output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return {
            'fold': self.fold_idx,
            'best_val_em': max(self.history['val_em']),
            'best_val_f1': self.best_val_f1,
            'final_val_em': self.history['val_em'][-1],
            'final_val_f1': self.history['val_f1'][-1],
        }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """Main training loop for 5-fold CV."""
    logger.info("Starting 5-fold CV training for evidence extraction")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    # Create CV splits
    cv_splits_dir = Path(cfg.data.cv_splits_dir)
    if not cv_splits_dir.exists() or cfg.cv.recreate_splits:
        logger.info("Creating CV splits...")
        posts_path = Path(cfg.data.data_dir) / 'redsm5_posts.csv'
        annotations_path = Path(cfg.data.data_dir) / 'redsm5_annotations.csv'

        create_cv_splits(
            str(posts_path),
            str(annotations_path),
            num_folds=cfg.cv.num_folds,
            random_seed=cfg.data.random_seed,
            output_dir=str(cv_splits_dir),
        )

    use_cached_dataset = bool(getattr(cfg.data, 'use_cached_dataset', False))
    cache_dir = getattr(cfg.data, 'cache_dir', None)
    overwrite_cache = bool(getattr(cfg.data, 'overwrite_cache', False))
    if use_cached_dataset:
        logger.info(
            "Cached datasets enabled -> dir: %s (overwrite=%s)",
            cache_dir or f"{cv_splits_dir}/_cache",
            overwrite_cache,
        )

    use_qlora = bool(getattr(cfg.model, 'use_qlora', False))
    lora_target_modules = getattr(cfg.model, 'lora_target_modules', None)
    if lora_target_modules:
        lora_target_modules = list(lora_target_modules)
    if use_qlora:
        logger.info(
            "QLoRA enabled -> r=%d, alpha=%d, dropout=%.3f",
            getattr(cfg.model, 'lora_r', 64),
            getattr(cfg.model, 'lora_alpha', 16),
            getattr(cfg.model, 'lora_dropout', 0.05),
        )

    # Train each fold (with optional resume support)
    fold_results = []

    start_fold = int(getattr(cfg.cv, 'start_fold', 0))
    resume_cfg = getattr(cfg, 'resume', None)
    if resume_cfg and bool(getattr(resume_cfg, 'enabled', False)):
        start_fold = int(getattr(resume_cfg, 'start_fold', start_fold))
    start_fold = max(0, min(start_fold, cfg.cv.num_folds))

    preloaded_results = []
    if start_fold > 0:
        logger.info(
            "Resume requested: loading cached metrics for folds [0, %d)",
            start_fold,
        )
        for fold_idx in range(start_fold):
            fold_output_dir = Path(cfg.output_dir) / f'fold_{fold_idx}'
            summary = load_fold_history_summary(fold_output_dir, fold_idx)
            if summary is None:
                logger.warning(
                    "Unable to resume fold %d (missing history in %s). "
                    "Training will restart from this fold.",
                    fold_idx,
                    fold_output_dir,
                )
                start_fold = fold_idx
                break
            preloaded_results.append(summary)
            logger.info(
                "Loaded fold %d history -> best Val F1 %.4f",
                fold_idx,
                summary['best_val_f1'],
            )

    if preloaded_results:
        fold_results.extend(preloaded_results[:start_fold])

    for fold_idx in range(start_fold, cfg.cv.num_folds):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Training Fold {fold_idx + 1}/{cfg.cv.num_folds}")
        logger.info(f"{'=' * 80}\n")

        # Create fold output directory
        fold_output_dir = Path(cfg.output_dir) / f'fold_{fold_idx}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Load fold data
        train_dataset, val_dataset = load_fold_split(
            str(cv_splits_dir),
            fold_idx,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
            use_cached_dataset=use_cached_dataset,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
        )

        # Initialize model
        model = GemmaQA(
            model_name=cfg.model.name,
            freeze_encoder=cfg.model.freeze_encoder,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            device=device,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
            use_qlora=use_qlora,
            qlora_r=getattr(cfg.model, 'lora_r', 64),
            qlora_alpha=getattr(cfg.model, 'lora_alpha', 16),
            qlora_dropout=getattr(cfg.model, 'lora_dropout', 0.05),
            qlora_target_modules=lora_target_modules,
        )

        # Train fold
        fold_trainer = FoldTrainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            fold_idx=fold_idx,
            device=device,
        )

        fold_result = fold_trainer.train(fold_output_dir)
        fold_results.append(fold_result)

    # Aggregate results
    results_df = pd.DataFrame(fold_results)
    logger.info(f"\n{'=' * 80}")
    logger.info("5-Fold CV Results")
    logger.info(f"{'=' * 80}\n")
    logger.info(results_df.to_string())

    # Compute aggregate statistics
    aggregate_results = {
        'mean_val_em': results_df['best_val_em'].mean(),
        'std_val_em': results_df['best_val_em'].std(),
        'mean_val_f1': results_df['best_val_f1'].mean(),
        'std_val_f1': results_df['best_val_f1'].std(),
        'min_val_f1': results_df['best_val_f1'].min(),
        'max_val_f1': results_df['best_val_f1'].max(),
    }

    logger.info("\nAggregate Results:")
    logger.info(f"Mean Val EM: {aggregate_results['mean_val_em']:.4f} ± {aggregate_results['std_val_em']:.4f}")
    logger.info(f"Mean Val F1: {aggregate_results['mean_val_f1']:.4f} ± {aggregate_results['std_val_f1']:.4f}")

    # Save results
    output_dir = Path(cfg.output_dir)
    results_df.to_csv(output_dir / 'cv_results.csv', index=False)

    with open(output_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
