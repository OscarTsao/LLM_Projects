"""
Training script for Gemma QA on ReDSM5 evidence extraction task.

Implements extractive QA (SQuAD-style) with:
- Bidirectional attention fine-tuning
- Start/end position prediction
- EM and F1 token overlap metrics
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
from tqdm.auto import tqdm
import argparse
import math
import faulthandler
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_qa import GemmaQA
from data.cv_splits import create_cv_splits, load_fold_split
from training.qa_metrics import extract_answers_batch, compute_metrics_batch
from utils.logger import setup_logger
from utils.console_viz import render_training_progress

logger = setup_logger('train_gemma_qa')
faulthandler.enable()


def determine_micro_batch_size(
    effective_batch_size: int,
    requested_micro_batch_size: Optional[int],
    model_name: str,
) -> int:
    """
    Choose a per-device micro batch size that fits GPU memory.

    If the user provides --micro_batch_size we honor it (capped by effective batch size).
    Otherwise we use a simple heuristic based on the available CUDA memory.
    """
    if effective_batch_size <= 0:
        return 1

    if requested_micro_batch_size is not None:
        return max(1, min(requested_micro_batch_size, effective_batch_size))

    if not torch.cuda.is_available():
        return 1

    try:
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        total_mem_gb = 0

    model_name_lower = model_name.lower()

    # Gemma 9B is memory hungry – force micro batch size 1 unless the user overrides.
    if '9b' in model_name_lower:
        return 1

    if total_mem_gb >= 60:
        return min(effective_batch_size, 4)
    if total_mem_gb >= 40:
        return min(effective_batch_size, 2)

    return 1


def prepare_cv_splits(
    data_dir: str,
    cv_splits_dir: str,
    num_folds: int,
    random_seed: int,
    include_negatives: bool,
    recreate: bool,
):
    """Ensure CV split CSVs exist for the requested configuration."""
    cv_path = Path(cv_splits_dir)
    existing_trains = []
    if cv_path.exists():
        existing_trains = list(cv_path.glob('fold_*_train.csv'))

    needs_creation = (
        recreate
        or not cv_path.exists()
        or len(existing_trains) < num_folds
    )

    if not needs_creation:
        return

    posts_path = Path(data_dir) / 'redsm5_posts.csv'
    annotations_path = Path(data_dir) / 'redsm5_annotations.csv'
    cv_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Creating %d-fold stratified splits at %s",
        num_folds,
        cv_path,
    )
    create_cv_splits(
        str(posts_path),
        str(annotations_path),
        num_folds=num_folds,
        random_seed=random_seed,
        output_dir=str(cv_path),
        include_negatives=include_negatives,
    )


class GemmaQATrainer:
    """Trainer for Gemma QA on evidence extraction."""

    def __init__(
        self,
        model: GemmaQA,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
        output_dir: str = './outputs',
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 20,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.early_stopping_patience = max(0, early_stopping_patience)
        self.patience_counter = 0

        self.num_update_steps_per_epoch = max(
            1, math.ceil(len(self.train_loader) / self.gradient_accumulation_steps)
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        total_steps = self.num_update_steps_per_epoch * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.best_val_f1 = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_em': [], 'val_f1': []}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc='Training')
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            loss, _, _ = self.model(
                input_ids,
                attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            scaled_loss = loss / self.gradient_accumulation_steps
            scaled_loss.backward()
            total_loss += loss.item()

            should_step = (
                (step + 1) % self.gradient_accumulation_steps == 0
                or (step + 1 == len(self.train_loader))
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / self.num_update_steps_per_epoch

    @torch.no_grad()
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict:
        """Evaluate on validation set."""
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        all_predictions = []
        all_ground_truths = []
        total_loss = 0

        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            start_positions = batch['start_positions'].to(self.device)
            end_positions = batch['end_positions'].to(self.device)

            # Forward pass with loss
            loss, start_logits, end_logits = self.model(
                input_ids,
                attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            total_loss += loss.item()

            # Extract predicted answers
            start_logits_np = start_logits.float().cpu().numpy()
            end_logits_np = end_logits.float().cpu().numpy()
            input_ids_list = input_ids.cpu().tolist()

            predicted_answers = extract_answers_batch(
                start_logits_np,
                end_logits_np,
                input_ids_list,
                self.tokenizer,
            )

            # Extract ground truth answers
            for i in range(len(input_ids)):
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()

                if start_idx == 0 and end_idx == 0:
                    # No answer / truncated
                    ground_truth = ""
                else:
                    answer_tokens = input_ids[i][start_idx:end_idx + 1]
                    ground_truth = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

                all_predictions.append(predicted_answers[i])
                all_ground_truths.append(ground_truth)

        # Compute metrics
        metrics = compute_metrics_batch(all_predictions, all_ground_truths)
        metrics['loss'] = total_loss / len(dataloader)

        return metrics

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        final_epoch = 0

        for epoch in range(self.num_epochs):
            final_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Evaluate
            val_metrics = self.evaluate()
            val_loss = val_metrics['loss']
            val_em = val_metrics['exact_match']
            val_f1 = val_metrics['f1']

            self.history['val_loss'].append(val_loss)
            self.history['val_em'].append(val_em)
            self.history['val_f1'].append(val_f1)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val EM: {val_em:.4f}")
            logger.info(f"Val F1: {val_f1:.4f}")

            progress_panel = render_training_progress(self.history)
            if progress_panel:
                logger.info("\n" + progress_panel)

            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                logger.info(f"Saved best model with F1: {val_f1:.4f}")
            else:
                if self.early_stopping_patience > 0:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info(
                            "Early stopping triggered after %d epochs without improvement",
                            self.patience_counter,
                        )
                        break

        # Save final checkpoint
        self.save_checkpoint('final_model.pt', final_epoch, val_metrics)
        self.save_history()

        logger.info(f"\nTraining complete! Best Val F1: {self.best_val_f1:.4f}")

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics,
            'model_kwargs': self.model.export_init_kwargs(),
        }, checkpoint_path)

    def save_history(self):
        """Save training history."""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Gemma QA on ReDSM5 evidence extraction')
    parser.add_argument('--data_dir', type=str, default='data/redsm5',
                        help='Path to data directory')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b',
                        help='Gemma model name')
    parser.add_argument('--output_dir', type=str, default='outputs/gemma_qa',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Effective batch size (after gradient accumulation)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Number of eval passes without F1 improvement before stopping')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--cv_splits_dir', type=str, default='data/cv_splits',
                        help='Directory storing CSVs for each CV fold')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of CV folds to run')
    parser.add_argument('--recreate_splits', action='store_true',
                        help='Regenerate CV splits before training')
    parser.add_argument('--include_negatives', action='store_true',
                        help='Include status=0 annotations when building CV splits')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for split generation')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Force-enable gradient checkpointing')
    parser.add_argument('--disable_auto_gradient_checkpointing', action='store_true',
                        help='Disable automatic activation of gradient checkpointing on smaller GPUs')
    parser.add_argument('--micro_batch_size', type=int, default=None,
                        help='Per-device micro batch size (auto-selected if not provided)')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Evaluation batch size (defaults to micro batch size)')
    parser.add_argument('--trainable_encoder_layers', type=int, default=None,
                        help='Number of final encoder layers to fine-tune (auto-selected on low-memory GPUs)')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to store cached dataset tensors')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Force regeneration of cached dataset tensors')
    parser.add_argument('--use_cached_dataset', action='store_true', dest='use_cached_dataset',
                        help='Enable cached datasets for faster IO')
    parser.add_argument('--no_cached_dataset', action='store_false', dest='use_cached_dataset',
                        help='Disable cached datasets')
    parser.set_defaults(use_cached_dataset=True)
    parser.add_argument('--use_qlora', action='store_true',
                        help='Enable 4-bit QLoRA adapters for parameter-efficient finetuning')
    parser.add_argument('--lora_r', type=int, default=64, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, default=None,
                        help='Comma-separated module names to target with LoRA (auto-inferred if omitted)')

    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    available_mem_gb = None
    if torch.cuda.is_available():
        try:
            available_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            available_mem_gb = None

    if (
        not args.disable_auto_gradient_checkpointing
        and not args.use_gradient_checkpointing
        and available_mem_gb is not None
    ):
        if available_mem_gb < 40:
            logger.info(
                "Auto-enabling gradient checkpointing for GPU memory %.1f GB (< 40 GB)",
                available_mem_gb,
            )
            args.use_gradient_checkpointing = True

    trainable_encoder_layers = args.trainable_encoder_layers
    if available_mem_gb is not None:
        if (
            trainable_encoder_layers is None
            and available_mem_gb < 30
            and not args.freeze_encoder
        ):
            # Keep only the last two layers trainable on 24 GB GPUs to stay within memory.
            trainable_encoder_layers = 2
            logger.info(
                "Auto-limiting trainable encoder layers to last %d (GPU memory %.1f GB)",
                trainable_encoder_layers,
                available_mem_gb,
            )

        if available_mem_gb < 30 and args.max_length > 384:
            logger.info(
                "Auto-reducing max_length from %d to 384 for GPU memory %.1f GB",
                args.max_length,
                available_mem_gb,
            )
            args.max_length = 384

    micro_batch_size = determine_micro_batch_size(
        effective_batch_size=args.batch_size,
        requested_micro_batch_size=args.micro_batch_size,
        model_name=args.model_name,
    )
    gradient_accumulation_steps = max(1, math.ceil(args.batch_size / micro_batch_size))
    eval_batch_size = args.eval_batch_size or micro_batch_size
    eval_batch_size = max(1, eval_batch_size)

    logger.info(
        "Batch configuration -> effective: %d | micro: %d | grad_accum_steps: %d | eval: %d",
        args.batch_size,
        micro_batch_size,
        gradient_accumulation_steps,
        eval_batch_size,
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Prepare CV splits
    prepare_cv_splits(
        data_dir=args.data_dir,
        cv_splits_dir=args.cv_splits_dir,
        num_folds=args.num_folds,
        random_seed=args.random_seed,
        include_negatives=args.include_negatives,
        recreate=args.recreate_splits,
    )

    # Create run-specific output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_output_dir = Path(args.output_dir) / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving fold artifacts to {run_output_dir}")

    if args.use_cached_dataset:
        default_cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.data_dir) / 'cache'
        cache_dir = str(default_cache_dir)
        logger.info(
            "Using cached datasets -> dir: %s (overwrite=%s)",
            cache_dir,
            args.overwrite_cache,
        )
    else:
        cache_dir = None

    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = [
            module.strip()
            for module in args.lora_target_modules.split(',')
            if module.strip()
        ]

    if args.use_qlora:
        logger.info(
            "QLoRA enabled -> r=%d, alpha=%d, dropout=%.3f",
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
        )

    fold_results: List[Dict[str, float]] = []

    for fold_idx in range(args.num_folds):
        logger.info("\n%s", "=" * 80)
        logger.info("Training fold %d/%d", fold_idx + 1, args.num_folds)
        logger.info("%s\n", "=" * 80)

        fold_output_dir = run_output_dir / f'fold_{fold_idx}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset, val_dataset = load_fold_split(
            args.cv_splits_dir,
            fold_idx,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_cached_dataset=args.use_cached_dataset,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
        )

        logger.info("Fold %d -> train size: %d | val size: %d", fold_idx, len(train_dataset), len(val_dataset))

        train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)

        if args.freeze_encoder:
            logger.info("Encoder freeze requested -> training QA head only.")
        elif trainable_encoder_layers is not None:
            logger.info("Training only the last %d encoder layers.", trainable_encoder_layers)
        else:
            logger.info("Training full encoder.")

        logger.info(f"Initializing model {args.model_name}")
        model = GemmaQA(
            model_name=args.model_name,
            freeze_encoder=args.freeze_encoder,
            device=device,
            use_gradient_checkpointing=args.use_gradient_checkpointing,
            trainable_encoder_layers=trainable_encoder_layers,
            use_qlora=args.use_qlora,
            qlora_r=args.lora_r,
            qlora_alpha=args.lora_alpha,
            qlora_dropout=args.lora_dropout,
            qlora_target_modules=lora_target_modules,
        )

        trainer = GemmaQATrainer(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            num_epochs=args.num_epochs,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            device=device,
            output_dir=fold_output_dir,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
        )

        trainer.train()

        best_val_em = max(trainer.history['val_em']) if trainer.history['val_em'] else 0.0
        best_val_f1 = trainer.best_val_f1
        final_val_em = trainer.history['val_em'][-1] if trainer.history['val_em'] else 0.0
        final_val_f1 = trainer.history['val_f1'][-1] if trainer.history['val_f1'] else 0.0

        fold_results.append({
            'fold': fold_idx,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'best_val_em': best_val_em,
            'best_val_f1': best_val_f1,
            'final_val_em': final_val_em,
            'final_val_f1': final_val_f1,
        })

    if not fold_results:
        logger.warning("No folds were processed. Please check your data configuration.")
        return

    best_f1_values = [fr['best_val_f1'] for fr in fold_results]
    best_em_values = [fr['best_val_em'] for fr in fold_results]

    aggregate_results = {
        'mean_val_em': float(np.mean(best_em_values)),
        'std_val_em': float(np.std(best_em_values)),
        'mean_val_f1': float(np.mean(best_f1_values)),
        'std_val_f1': float(np.std(best_f1_values)),
        'min_val_f1': float(np.min(best_f1_values)),
        'max_val_f1': float(np.max(best_f1_values)),
    }

    logger.info("\n%s", "=" * 80)
    logger.info("Cross-validation results")
    logger.info("%s\n", "=" * 80)
    for fr in fold_results:
        logger.info(
            "Fold %d | best EM %.4f | best F1 %.4f | final EM %.4f | final F1 %.4f",
            fr['fold'],
            fr['best_val_em'],
            fr['best_val_f1'],
            fr['final_val_em'],
            fr['final_val_f1'],
        )

    logger.info("\nAggregate: mean EM %.4f ± %.4f | mean F1 %.4f ± %.4f",
                aggregate_results['mean_val_em'],
                aggregate_results['std_val_em'],
                aggregate_results['mean_val_f1'],
                aggregate_results['std_val_f1'])

    with open(run_output_dir / 'cv_results.json', 'w') as f:
        json.dump(fold_results, f, indent=2)
    with open(run_output_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    logger.info("Results saved to %s", run_output_dir)


if __name__ == '__main__':
    main()
