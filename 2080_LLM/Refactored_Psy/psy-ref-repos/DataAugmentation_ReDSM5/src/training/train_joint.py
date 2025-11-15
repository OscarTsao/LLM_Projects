"""Training script for joint training of both agents (Mode 3)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.agents.base import setup_hardware_optimizations
from src.agents.multi_agent_pipeline import JointTrainingModel, JointTrainingConfig
from src.data.joint_dataset import JointCollator, build_joint_datasets
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)


def create_joint_model(cfg: DictConfig) -> JointTrainingModel:
    """Create joint training model from config."""
    from src.agents.base import CriteriaMatchingConfig, EvidenceBindingConfig
    
    criteria_config = CriteriaMatchingConfig(
        model_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.classifier_dropout,
        classifier_hidden_sizes=cfg.model.get("classifier_hidden_sizes", [256]),
        loss_type=cfg.model.get("loss_type", "adaptive_focal"),
        alpha=cfg.model.get("alpha", 0.25),
        gamma=cfg.model.get("gamma", 2.0),
        delta=cfg.model.get("delta", 1.0),
    )
    
    evidence_config = EvidenceBindingConfig(
        model_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.classifier_dropout,
        label_smoothing=cfg.model.get("label_smoothing", 0.0),
        max_span_length=cfg.model.get("max_span_length", 50),
        span_threshold=cfg.model.get("span_threshold", 0.5),
    )
    
    config = JointTrainingConfig(
        model_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.classifier_dropout,
        criteria_config=criteria_config,
        evidence_config=evidence_config,
        criteria_loss_weight=cfg.model.get("criteria_loss_weight", 0.5),
        evidence_loss_weight=cfg.model.get("evidence_loss_weight", 0.5),
        shared_encoder=cfg.model.get("shared_encoder", True),
        freeze_encoder_epochs=cfg.model.get("freeze_encoder_epochs", 0),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_ratio=cfg.model.warmup_ratio,
        use_amp=cfg.model.get("use_amp", True),
        use_compile=cfg.model.get("compile_model", False),
        use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", True)
    )
    
    model = JointTrainingModel(config)
    
    # Apply optimizations
    if config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    if config.use_compile:
        model = model.compile_model()
    
    return model


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function for joint training."""
    
    # Setup hardware optimizations
    setup_hardware_optimizations()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting joint training (Mode 3)")
    
    # Initialize MLflow
    mlflow_run = mlflow_utils.start_run(
        experiment_name="redsm5-joint",
        run_name=f"joint-{cfg.model.pretrained_model_name.split('/')[-1]}"
    )
    
    try:
        with mlflow_run:
            # Log configuration
            flat_params = hydra.utils.instantiate(cfg, _convert_="all")
            mlflow_utils.log_params(flat_params)
            mlflow_utils.set_tag("training_mode", "joint")
            mlflow_utils.set_tag("agent_type", "joint_training")
            
            # Create datasets
            logger.info("Creating joint training datasets...")
            train_dataset, val_dataset, test_dataset = build_joint_datasets(
                ground_truth_path=cfg.data.ground_truth_path,
                posts_path=cfg.data.posts_path,
                annotations_path=cfg.data.annotations_path,
                tokenizer_name=cfg.model.pretrained_model_name,
                criteria_path=cfg.data.get("criteria_path"),
                max_length=cfg.model.max_seq_length,
                random_state=cfg.get("seed", 42)
            )
            
            # Create data loaders
            collator = JointCollator(
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
            
            # Create joint model
            logger.info("Creating joint training model...")
            model = create_joint_model(cfg)
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            logger.info(f"Using device: {device}")
            
            # Create optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
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
                "metric_for_best_model": cfg.get("metric_for_best_model", "combined_f1"),
                "output_dir": cfg.output_dir,
                "save_steps": cfg.get("save_steps", None),
                "eval_steps": cfg.get("eval_steps", None),
                "logging_steps": cfg.get("logging_steps", 100),
                "freeze_encoder_epochs": cfg.model.get("freeze_encoder_epochs", 0),
            }
            
            # Train model
            logger.info("Starting joint training...")
            best_metric = train_joint_model(
                model=model,
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


def train_joint_model(
    model: JointTrainingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict
) -> float:
    """Train the joint model."""
    
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
        # Handle encoder freezing
        if epoch < config["freeze_encoder_epochs"]:
            if hasattr(model, 'shared_encoder'):
                for param in model.shared_encoder.parameters():
                    param.requires_grad = False
            logger.info(f"Encoder frozen for epoch {epoch+1}")
        else:
            if hasattr(model, 'shared_encoder'):
                for param in model.shared_encoder.parameters():
                    param.requires_grad = True
        
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {
                "criteria": batch.pop("criteria"),
                "start_positions": batch.pop("start_positions"),
                "end_positions": batch.pop("end_positions")
            }
            
            # Forward pass
            if config["use_amp"] and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(**batch)
                    loss = model.get_loss(outputs, targets)
                    loss = loss / config["gradient_accumulation_steps"]

                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = model.get_loss(outputs, targets)
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                if config["use_amp"] and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * config["gradient_accumulation_steps"]})
        
        # Validation phase
        model.eval()
        val_criteria_preds = []
        val_criteria_labels = []
        val_start_preds = []
        val_start_labels = []
        val_end_preds = []
        val_end_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                criteria_targets = batch.pop("criteria")
                start_targets = batch.pop("start_positions")
                end_targets = batch.pop("end_positions")
                
                outputs = model(**batch)
                
                # Criteria predictions
                criteria_probs = outputs.probabilities["criteria"]
                criteria_preds = outputs.predictions["criteria"]
                
                val_criteria_preds.extend(criteria_preds.cpu().numpy())
                val_criteria_labels.extend(criteria_targets.cpu().numpy())
                
                # Evidence predictions
                start_probs = outputs.probabilities["start"]
                end_probs = outputs.probabilities["end"]
                
                start_preds = (start_probs > 0.5).float()
                end_preds = (end_probs > 0.5).float()
                
                val_start_preds.extend(start_preds.cpu().numpy().flatten())
                val_start_labels.extend(start_targets.cpu().numpy().flatten())
                val_end_preds.extend(end_preds.cpu().numpy().flatten())
                val_end_labels.extend(end_targets.cpu().numpy().flatten())
        
        # Calculate criteria metrics
        criteria_accuracy = accuracy_score(val_criteria_labels, val_criteria_preds)
        criteria_f1 = f1_score(val_criteria_labels, val_criteria_preds, zero_division=0)
        criteria_precision = precision_score(val_criteria_labels, val_criteria_preds, zero_division=0)
        criteria_recall = recall_score(val_criteria_labels, val_criteria_preds, zero_division=0)
        
        # Calculate evidence metrics
        start_accuracy = accuracy_score(val_start_labels, val_start_preds)
        start_f1 = f1_score(val_start_labels, val_start_preds, zero_division=0)
        end_accuracy = accuracy_score(val_end_labels, val_end_preds)
        end_f1 = f1_score(val_end_labels, val_end_preds, zero_division=0)
        
        evidence_accuracy = (start_accuracy + end_accuracy) / 2
        evidence_f1 = (start_f1 + end_f1) / 2
        
        # Combined metrics
        combined_f1 = (criteria_f1 + evidence_f1) / 2
        combined_accuracy = (criteria_accuracy + evidence_accuracy) / 2
        
        avg_train_loss = total_loss / max(num_batches, 1)
        
        # Log metrics
        metrics = {
            "train_loss": avg_train_loss,
            "val_criteria_accuracy": criteria_accuracy,
            "val_criteria_f1": criteria_f1,
            "val_criteria_precision": criteria_precision,
            "val_criteria_recall": criteria_recall,
            "val_evidence_accuracy": evidence_accuracy,
            "val_evidence_f1": evidence_f1,
            "val_start_accuracy": start_accuracy,
            "val_start_f1": start_f1,
            "val_end_accuracy": end_accuracy,
            "val_end_f1": end_f1,
            "val_combined_f1": combined_f1,
            "val_combined_accuracy": combined_accuracy,
        }
        
        mlflow_utils.log_metrics(metrics, step=epoch)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Loss: {avg_train_loss:.4f}, "
            f"Criteria F1: {criteria_f1:.4f}, "
            f"Evidence F1: {evidence_f1:.4f}, "
            f"Combined F1: {combined_f1:.4f}"
        )
        
        # Check for best model
        current_metric = metrics[f"val_{config['metric_for_best_model']}"]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # Save best model
            best_model_path = output_dir / "best" / "model.pt"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            
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


if __name__ == "__main__":
    main()
