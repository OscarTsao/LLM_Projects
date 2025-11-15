"""Training loop with MLflow logging and early stopping."""

import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Training loop with MLflow integration.
    
    Features:
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - MLflow metric logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 10,
        patience: int = 3,
        gradient_clip: Optional[float] = 1.0,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            scheduler: Optional learning rate scheduler
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.scheduler = scheduler
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.save_dir:
            return
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_checkpoint.pt")
    
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with final metrics
        """
        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch + 1
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["epoch_time"] = time.time() - epoch_start
            
            # Log to MLflow
            mlflow.log_metrics(metrics, step=epoch)
            
            # Save history
            self.training_history.append(metrics)
            
            # Print progress
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )
            
            # Check for improvement
            improved = val_metrics["val_loss"] < self.best_val_loss
            
            if improved:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_acc = val_metrics["val_accuracy"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  New best model! Val Loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Log best metrics
        mlflow.log_metrics({
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_acc,
        })
        
        return {
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_acc,
            "total_epochs": len(self.training_history),
        }
