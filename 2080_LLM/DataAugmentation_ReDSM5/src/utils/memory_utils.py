"""Utilities for GPU memory monitoring and management."""
from __future__ import annotations

import gc
import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information in GB."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_memory = total_memory - allocated
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "total_gb": total_memory,
        "free_gb": free_memory,
        "utilization_pct": (allocated / total_memory) * 100
    }


def log_memory_usage(stage: str, trial_number: Optional[int] = None) -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    
    memory_info = get_gpu_memory_info()
    prefix = f"Trial {trial_number} - " if trial_number is not None else ""
    
    logger.info(
        f"{prefix}{stage}: "
        f"Allocated: {memory_info['allocated_gb']:.2f}GB "
        f"({memory_info['utilization_pct']:.1f}%), "
        f"Reserved: {memory_info['reserved_gb']:.2f}GB, "
        f"Free: {memory_info['free_gb']:.2f}GB, "
        f"Total: {memory_info['total_gb']:.2f}GB"
    )


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def check_memory_available(required_gb: float, buffer_gb: float = 2.0) -> bool:
    """Check if enough GPU memory is available for a given requirement."""
    if not torch.cuda.is_available():
        return True
    
    memory_info = get_gpu_memory_info()
    available = memory_info["free_gb"] - buffer_gb
    
    return available >= required_gb


def estimate_model_memory(
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    classifier_layers: int = 0,
    gradient_accumulation_steps: int = 1
) -> float:
    """
    Estimate GPU memory requirements for a model configuration.
    
    Args:
        model_name: Name of the model (bert_base, roberta_base, deberta_base)
        batch_size: Training batch size
        max_seq_length: Maximum sequence length
        classifier_layers: Number of classifier layers
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        Estimated memory usage in GB
    """
    # Base model memory estimates (in GB) - includes model weights
    base_memory = {
        "bert_base": 0.5,
        "roberta_base": 0.5,
        "deberta_base": 0.6,  # DeBERTa is slightly larger
    }
    
    # Get base memory for model type
    model_base = base_memory.get(model_name, 0.5)
    for key in base_memory:
        if key in model_name.lower():
            model_base = base_memory[key]
            break
    
    # DeBERTa has higher memory overhead due to relative position embeddings
    if "deberta" in model_name.lower():
        model_base *= 1.2  # 20% overhead for DeBERTa
    
    # Memory scaling factors
    seq_length_factor = max_seq_length / 256  # Baseline at 256 tokens
    batch_factor = batch_size / 32  # Baseline at batch size 32
    classifier_factor = 1 + (classifier_layers * 0.1)  # Each layer adds ~10% overhead
    
    # Effective batch size considering gradient accumulation
    effective_batch_size = batch_size * gradient_accumulation_steps
    effective_batch_factor = effective_batch_size / 32
    
    # Memory components:
    # 1. Model weights: model_base
    # 2. Activations: scales with batch_size and seq_length
    # 3. Gradients: same size as model weights
    # 4. Optimizer states (AdamW): 2x model weights (momentum + variance)
    
    activation_memory = model_base * seq_length_factor * effective_batch_factor * classifier_factor
    gradient_memory = model_base * classifier_factor
    optimizer_memory = model_base * classifier_factor * 2  # AdamW states
    
    total_memory = model_base + activation_memory + gradient_memory + optimizer_memory
    
    # Add 20% safety margin
    total_memory *= 1.2
    
    return total_memory


def is_configuration_memory_safe(
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    classifier_layers: int = 0,
    gradient_accumulation_steps: int = 1,
    buffer_gb: float = 2.0
) -> tuple[bool, float, float]:
    """
    Check if a configuration is likely to fit in GPU memory.
    
    Returns:
        Tuple of (is_safe, estimated_memory_gb, available_memory_gb)
    """
    if not torch.cuda.is_available():
        return True, 0.0, float('inf')
    
    estimated_memory = estimate_model_memory(
        model_name, batch_size, max_seq_length, 
        classifier_layers, gradient_accumulation_steps
    )
    
    memory_info = get_gpu_memory_info()
    available_memory = memory_info["free_gb"] - buffer_gb
    
    is_safe = estimated_memory <= available_memory
    
    return is_safe, estimated_memory, available_memory


class MemoryMonitor:
    """Context manager for monitoring memory usage during training."""
    
    def __init__(self, stage: str, trial_number: Optional[int] = None):
        self.stage = stage
        self.trial_number = trial_number
        self.start_memory = None
    
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_memory = get_gpu_memory_info()
            log_memory_usage(f"{self.stage} Start", self.trial_number)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_memory = get_gpu_memory_info()
            if self.start_memory:
                memory_diff = end_memory["allocated_gb"] - self.start_memory["allocated_gb"]
                prefix = f"Trial {self.trial_number} - " if self.trial_number is not None else ""
                logger.info(
                    f"{prefix}{self.stage} End: "
                    f"Memory change: {memory_diff:+.2f}GB, "
                    f"Current: {end_memory['allocated_gb']:.2f}GB"
                )
            else:
                log_memory_usage(f"{self.stage} End", self.trial_number)
        
        # Clear memory on exit
        clear_gpu_memory()
