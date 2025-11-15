"""Base agent interface for the multi-agent system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class AgentOutput:
    """Standard output format for all agents."""
    
    # Core outputs
    predictions: Union[torch.Tensor, List[Any]]
    confidence: Union[float, torch.Tensor]
    
    # Optional outputs
    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    
    # Agent-specific outputs
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentConfig:
    """Base configuration for agents."""
    
    model_name: str = "google-bert/bert-base-uncased"
    max_seq_length: int = 512
    dropout: float = 0.1
    device: Optional[str] = None
    
    # Training settings
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Hardware optimizations
    use_amp: bool = True
    use_compile: bool = False
    use_gradient_checkpointing: bool = True


class BaseAgent(nn.Module, ABC):
    """Abstract base class for all agents in the multi-agent system."""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
    @abstractmethod
    def forward(self, **inputs) -> AgentOutput:
        """Forward pass through the agent."""
        pass
    
    @abstractmethod
    def predict(self, **inputs) -> AgentOutput:
        """Make predictions without computing gradients."""
        pass
    
    @abstractmethod
    def get_loss(self, outputs: AgentOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for training."""
        pass
    
    def to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move inputs to the appropriate device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
    
    def compile_model(self):
        """Compile model for faster inference (PyTorch 2.0+)."""
        if self.config.use_compile and hasattr(torch, 'compile'):
            return torch.compile(self)
        return self


@dataclass
class CriteriaMatchingConfig(AgentConfig):
    """Configuration for criteria matching agent."""
    
    num_labels: int = 2
    classifier_hidden_sizes: List[int] = None
    loss_type: str = "adaptive_focal"  # bce, focal, adaptive_focal
    
    # Focal loss parameters
    alpha: float = 0.25
    gamma: float = 2.0
    delta: float = 1.0  # For adaptive focal loss
    
    def __post_init__(self):
        if self.classifier_hidden_sizes is None:
            self.classifier_hidden_sizes = [256]


@dataclass  
class EvidenceBindingConfig(AgentConfig):
    """Configuration for evidence binding agent."""
    
    num_labels: int = 3  # B, I, O for BIO tagging
    use_crf: bool = False
    label_smoothing: float = 0.0
    
    # For span-based evaluation
    max_span_length: int = 50
    span_threshold: float = 0.5


@dataclass
class JointTrainingConfig(AgentConfig):
    """Configuration for joint training of both agents."""
    
    criteria_config: CriteriaMatchingConfig = None
    evidence_config: EvidenceBindingConfig = None
    
    # Task weighting
    criteria_loss_weight: float = 0.5
    evidence_loss_weight: float = 0.5
    
    # Shared encoder settings
    shared_encoder: bool = True
    freeze_encoder_epochs: int = 0
    
    def __post_init__(self):
        if self.criteria_config is None:
            self.criteria_config = CriteriaMatchingConfig()
        if self.evidence_config is None:
            self.evidence_config = EvidenceBindingConfig()


class MultiTaskLoss(nn.Module):
    """Multi-task loss for joint training."""
    
    def __init__(self, criteria_weight: float = 0.5, evidence_weight: float = 0.5):
        super().__init__()
        self.criteria_weight = criteria_weight
        self.evidence_weight = evidence_weight
        
    def forward(self, criteria_loss: torch.Tensor, evidence_loss: torch.Tensor) -> torch.Tensor:
        """Combine losses from both tasks."""
        return self.criteria_weight * criteria_loss + self.evidence_weight * evidence_loss
    
    def update_weights(self, criteria_weight: float, evidence_weight: float):
        """Update task weights during training."""
        self.criteria_weight = criteria_weight
        self.evidence_weight = evidence_weight


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["memory_allocated"] = torch.cuda.memory_allocated()
        info["memory_reserved"] = torch.cuda.memory_reserved()
        
    return info


def setup_hardware_optimizations():
    """Setup hardware optimizations for training."""
    if torch.cuda.is_available():
        # Enable TF32 for faster training on Ampere GPUs (PyTorch 2.9+ API)
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
