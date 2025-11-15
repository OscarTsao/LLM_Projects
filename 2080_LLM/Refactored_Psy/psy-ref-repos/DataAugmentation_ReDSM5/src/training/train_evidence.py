"""Training script for evidence binding agent (Mode 2)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.agents.base import setup_hardware_optimizations
from src.agents.evidence_binding import EvidenceBindingAgent, EvidenceBindingConfig
from src.data.evidence_loader import EvidenceCollator, EvidenceDataset, load_evidence_annotations
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)


def create_evidence_agent(cfg: DictConfig) -> EvidenceBindingAgent:
    """Create evidence binding agent from config."""
    config = EvidenceBindingConfig(
        model_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.classifier_dropout,
        label_smoothing=cfg.model.get("label_smoothing", 0.0),
        max_span_length=cfg.model.get("max_span_length", 50),
        span_threshold=cfg.model.get("span_threshold", 0.5),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_ratio=cfg.model.warmup_ratio,
        use_amp=cfg.model.get("use_amp", True),
        use_compile=cfg.model.get("compile_model", False),
        use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", True)
    )
    
    agent = EvidenceBindingAgent(config)
    
    # Apply optimizations
    if config.use_gradient_checkpointing:
        agent.enable_gradient_checkpointing()
    
    if config.use_compile:
        agent = agent.compile_model()
    
    return agent


def create_evidence_datasets(cfg: DictConfig) -> Tuple[EvidenceDataset, EvidenceDataset, EvidenceDataset]:
    """Create evidence binding datasets."""
    # Load evidence annotations
    examples = load_evidence_annotations(
        posts_path=cfg.data.posts_path,
        annotations_path=cfg.data.annotations_path,
        criteria_path=cfg.data.get("criteria_path")
    )
    
    # Split data (simple split for now - could use group-based splitting)
    from sklearn.model_selection import train_test_split
    
    train_examples, temp_examples = train_test_split(
        examples, 
        test_size=0.3, 
        random_state=cfg.get("seed", 42),
        stratify=[ex.has_evidence for ex in examples]
    )
    
    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.5,
        random_state=cfg.get("seed", 42),
        stratify=[ex.has_evidence for ex in temp_examples]
    )
    
    # Create datasets
    train_dataset = EvidenceDataset(
        examples=train_examples,
        tokenizer_name=cfg.model.pretrained_model_name,
        max_length=cfg.model.max_seq_length
    )
    
    val_dataset = EvidenceDataset(
        examples=val_examples,
        tokenizer_name=cfg.model.pretrained_model_name,
        max_length=cfg.model.max_seq_length
    )
    
    test_dataset = EvidenceDataset(
        examples=test_examples,
        tokenizer_name=cfg.model.pretrained_model_name,
        max_length=cfg.model.max_seq_length
    )
    
    return train_dataset, val_dataset, test_dataset


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function for evidence binding agent."""
    
    # Setup hardware optimizations
    setup_hardware_optimizations()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting evidence binding agent training (Mode 2)")
    
    # Initialize MLflow
    mlflow_run = mlflow_utils.start_run(
        experiment_name="redsm5-evidence",
        run_name=f"evidence-{cfg.model.pretrained_model_name.split('/')[-1]}"
    )
    
    try:
        with mlflow_run:
            # Log configuration
            flat_params = hydra.utils.instantiate(cfg, _convert_="all")
            mlflow_utils.log_params(flat_params)
            mlflow_utils.set_tag("training_mode", "evidence_only")
            mlflow_utils.set_tag("agent_type", "evidence_binding")
            
            # Create datasets
            logger.info("Creating evidence binding datasets...")
            train_dataset, val_dataset, test_dataset = create_evidence_datasets(cfg)
            
            # Create data loaders
            collator = EvidenceCollator(
                tokenizer_name=cfg.model.pretrained_model_name,
                max_length=cfg.model.max_seq_length
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.model.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=cfg.dataloader.get("num_workers", 0),
                pin_memory=cfg.dataloader.get("pin_memory", True)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.model.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=cfg.dataloader.get("num_workers", 0),
                pin_memory=cfg.dataloader.get("pin_memory", True)
            )
            
            # Create agent
            logger.info("Creating evidence binding agent...")
            agent = create_evidence_agent(cfg)
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent = agent.to(device)
            logger.info(f"Using device: {device}")
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                agent.parameters(),
                lr=cfg.model.learning_rate,
                weight_decay=cfg.model.weight_decay,
                eps=cfg.model.get("adam_eps", 1e-8)
            )
            
            # Create scheduler
            num_training_steps = len(train_loader) * cfg.model.num_epochs
            num_warmup_steps = int(cfg.model.warmup_ratio * num_training_steps)
            
            if cfg.model.scheduler == "linear":
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif cfg.model.scheduler == "cosine":
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                scheduler = None
            
            # Training configuration
            train_config = {
                "num_epochs": cfg.model.num_epochs,
                "gradient_accumulation_steps": cfg.model.gradient_accumulation_steps,
                "max_grad_norm": cfg.model.max_grad_norm,
                "use_amp": cfg.model.get("use_amp", True),
                "early_stopping_patience": cfg.get("early_stopping_patience", 10),
                "metric_for_best_model": cfg.get("metric_for_best_model", "span_f1"),
                "output_dir": cfg.output_dir,
                "save_steps": cfg.get("save_steps", None),
                "eval_steps": cfg.get("eval_steps", None),
                "logging_steps": cfg.get("logging_steps", 100),
            }
            
            # Train model
            logger.info("Starting training...")
            best_metric = train_evidence_agent(
                agent=agent,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=train_config
            )
            
            logger.info(f"Training completed. Best {train_config['metric_for_best_model']}: {best_metric:.4f}")
            
            # Log final metrics
            mlflow_utils.log_metrics({"best_metric": best_metric})
            
            return best_metric
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if mlflow_run:
            mlflow_run.__exit__(None, None, None)


def train_evidence_agent(
    agent: EvidenceBindingAgent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict
) -> float:
    """Train the evidence binding agent."""
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from tqdm.auto import tqdm
    import numpy as np
    
    # Training setup
    scaler = torch.amp.GradScaler('cuda') if config["use_amp"] else None
    best_metric = 0.0
    patience_counter = 0
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config["num_epochs"]):
        # Training phase
        agent.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {
                "start_positions": batch.pop("start_positions"),
                "end_positions": batch.pop("end_positions")
            }
            
            # Forward pass
            if config["use_amp"] and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = agent(**batch)
                    loss = agent.get_loss(outputs, targets)
                    loss = loss / config["gradient_accumulation_steps"]

                scaler.scale(loss).backward()
            else:
                outputs = agent(**batch)
                loss = agent.get_loss(outputs, targets)
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                if config["use_amp"] and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * config["gradient_accumulation_steps"]})
        
        # Validation phase
        agent.eval()
        val_start_preds = []
        val_start_labels = []
        val_end_preds = []
        val_end_labels = []
        val_spans = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                start_targets = batch.pop("start_positions")
                end_targets = batch.pop("end_positions")
                
                outputs = agent(**batch)
                
                # Token-level predictions
                start_probs = outputs.probabilities["start"]
                end_probs = outputs.probabilities["end"]
                
                start_preds = (start_probs > 0.5).float()
                end_preds = (end_probs > 0.5).float()
                
                val_start_preds.extend(start_preds.cpu().numpy().flatten())
                val_start_labels.extend(start_targets.cpu().numpy().flatten())
                val_end_preds.extend(end_preds.cpu().numpy().flatten())
                val_end_labels.extend(end_targets.cpu().numpy().flatten())
                
                # Span-level predictions
                val_spans.extend(outputs.predictions)
        
        # Calculate token-level metrics
        start_accuracy = accuracy_score(val_start_labels, val_start_preds)
        start_f1 = f1_score(val_start_labels, val_start_preds, zero_division=0)
        end_accuracy = accuracy_score(val_end_labels, val_end_preds)
        end_f1 = f1_score(val_end_labels, val_end_labels, zero_division=0)
        
        token_accuracy = (start_accuracy + end_accuracy) / 2
        token_f1 = (start_f1 + end_f1) / 2
        
        # Calculate span-level metrics (simplified)
        span_f1 = calculate_span_f1(val_spans, val_loader.dataset)
        
        avg_train_loss = total_loss / max(num_batches, 1)
        
        # Log metrics
        metrics = {
            "train_loss": avg_train_loss,
            "val_token_accuracy": token_accuracy,
            "val_token_f1": token_f1,
            "val_start_accuracy": start_accuracy,
            "val_start_f1": start_f1,
            "val_end_accuracy": end_accuracy,
            "val_end_f1": end_f1,
            "val_span_f1": span_f1,
        }
        
        mlflow_utils.log_metrics(metrics, step=epoch)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Loss: {avg_train_loss:.4f}, "
            f"Token F1: {token_f1:.4f}, "
            f"Span F1: {span_f1:.4f}, "
            f"Token Acc: {token_accuracy:.4f}"
        )
        
        # Check for best model
        current_metric = metrics[f"val_{config['metric_for_best_model']}"]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / "best" / "model.pt"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), best_model_path)
            
            # Save config
            import yaml
            config_path = output_dir / "best" / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(dict(config), f)
                
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config["early_stopping_patience"]:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return best_metric


def calculate_span_f1(predicted_spans: List[List[Tuple[int, int]]], dataset) -> float:
    """Calculate span-level F1 score."""
    # This is a simplified implementation
    # In practice, you'd want to compare against ground truth spans
    
    total_predicted = sum(len(spans) for spans in predicted_spans)
    total_actual = len([ex for ex in dataset.processed_examples if ex['has_evidence']])
    
    if total_predicted == 0 and total_actual == 0:
        return 1.0
    elif total_predicted == 0 or total_actual == 0:
        return 0.0
    
    # Simplified F1 calculation
    # This should be replaced with proper span matching logic
    precision = min(total_actual, total_predicted) / total_predicted if total_predicted > 0 else 0
    recall = min(total_actual, total_predicted) / total_actual if total_actual > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    main()
