"""Training script for criteria matching agent (Mode 1)."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from src.agents.base import setup_hardware_optimizations
from src.agents.criteria_matching import CriteriaMatchingAgent, CriteriaMatchingConfig
from src.training.data_module import DataModule, DataModuleConfig
from src.training.dataset_builder import build_splits
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)


def create_criteria_agent(cfg: DictConfig) -> CriteriaMatchingAgent:
    """Create criteria matching agent from config."""
    config = CriteriaMatchingConfig(
        model_name=cfg.model.pretrained_model_name,
        max_seq_length=cfg.model.max_seq_length,
        dropout=cfg.model.classifier_dropout,
        classifier_hidden_sizes=cfg.model.classifier_hidden_sizes,
        loss_type=cfg.model.get("loss_type", "adaptive_focal"),
        alpha=cfg.model.get("alpha", 0.25),
        gamma=cfg.model.get("gamma", 2.0),
        delta=cfg.model.get("delta", 1.0),
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        warmup_ratio=cfg.model.warmup_ratio,
        use_amp=cfg.model.get("use_amp", True),
        use_compile=cfg.model.get("compile_model", False),
        use_gradient_checkpointing=cfg.model.get("use_gradient_checkpointing", True)
    )
    
    agent = CriteriaMatchingAgent(config)
    
    # Apply optimizations
    if config.use_gradient_checkpointing:
        agent.enable_gradient_checkpointing()
    
    if config.use_compile:
        agent = agent.compile_model()
    
    return agent


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function for criteria matching agent."""
    
    # Setup hardware optimizations
    setup_hardware_optimizations()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting criteria matching agent training (Mode 1)")
    
    # Initialize MLflow
    mlflow_run = mlflow_utils.start_run(
        experiment_name="redsm5-criteria",
        run_name=f"criteria-{cfg.model.pretrained_model_name.split('/')[-1]}"
    )
    
    try:
        with mlflow_run:
            # Log configuration
            flat_params = hydra.utils.instantiate(cfg, _convert_="all")
            mlflow_utils.log_params(flat_params)
            mlflow_utils.set_tag("training_mode", "criteria_only")
            mlflow_utils.set_tag("agent_type", "criteria_matching")
            
            # Build data splits
            logger.info("Building data splits...")
            splits = build_splits(cfg.dataset)
            
            # Create data module
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
            
            # Create agent
            logger.info("Creating criteria matching agent...")
            agent = create_criteria_agent(cfg)
            
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
            num_training_steps = len(data_module.train_dataloader()) * cfg.model.num_epochs
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
                "metric_for_best_model": cfg.get("metric_for_best_model", "f1"),
                "output_dir": cfg.output_dir,
                "save_steps": cfg.get("save_steps", None),
                "eval_steps": cfg.get("eval_steps", None),
                "logging_steps": cfg.get("logging_steps", 100),
            }
            
            # Train model
            logger.info("Starting training...")
            best_metric = train_criteria_agent(
                agent=agent,
                data_module=data_module,
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


def train_criteria_agent(
    agent: CriteriaMatchingAgent,
    data_module: DataModule,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    config: dict
) -> float:
    """Train the criteria matching agent."""
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from tqdm.auto import tqdm
    import math
    
    # Data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
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
            labels = batch.pop("labels")
            
            # Forward pass
            if config["use_amp"] and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = agent(**batch)
                    loss = agent.get_loss(outputs, labels)
                    loss = loss / config["gradient_accumulation_steps"]
                
                scaler.scale(loss).backward()
            else:
                outputs = agent(**batch)
                loss = agent.get_loss(outputs, labels)
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
        val_predictions = []
        val_labels = []
        val_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                outputs = agent(**batch)
                
                val_predictions.extend(outputs.predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probabilities.extend(outputs.probabilities.cpu().numpy())
        
        # Calculate metrics
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, zero_division=0)
        val_precision = precision_score(val_labels, val_predictions, zero_division=0)
        val_recall = recall_score(val_labels, val_predictions, zero_division=0)
        
        try:
            val_auc = roc_auc_score(val_labels, val_probabilities)
        except ValueError:
            val_auc = 0.0
        
        avg_train_loss = total_loss / max(num_batches, 1)
        
        # Log metrics
        metrics = {
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_auc": val_auc,
        }
        
        mlflow_utils.log_metrics(metrics, step=epoch)
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Loss: {avg_train_loss:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Val AUC: {val_auc:.4f}"
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


if __name__ == "__main__":
    main()
