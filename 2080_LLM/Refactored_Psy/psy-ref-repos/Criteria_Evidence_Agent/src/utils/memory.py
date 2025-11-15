"""Memory optimization utilities for GPU training."""

import os
import gc
from typing import Dict, Any, Optional, Tuple

import torch


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage information.

    Returns:
        Dictionary with memory information in GB
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "allocated": 0.0, "cached": 0.0, "free": 0.0}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    cached = torch.cuda.memory_reserved(0) / (1024**3)
    free = total - cached

    return {"total": total, "allocated": allocated, "cached": cached, "free": free}


def clear_gpu_cache() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def estimate_memory_usage(
    batch_size: int, max_length: int, model_name: str, gradient_accumulation_steps: int = 1
) -> float:
    """Estimate GPU memory usage for given parameters.

    Args:
        batch_size: Training batch size
        max_length: Maximum sequence length
        model_name: Model name (for model-specific estimates)
        gradient_accumulation_steps: Gradient accumulation steps

    Returns:
        Estimated memory usage in GB
    """
    # Base memory estimates (empirical values)
    base_memory = {
        "bert": 1.2,  # BERT-base baseline
        "roberta": 1.3,  # RoBERTa-base slightly higher
        "deberta": 1.8,  # DeBERTa-base significantly higher
    }

    # Determine model type
    model_type = "bert"  # default
    if "roberta" in model_name.lower():
        model_type = "roberta"
    elif "deberta" in model_name.lower():
        model_type = "deberta"

    # Base model memory
    base_mem = base_memory[model_type]

    # Memory scaling factors
    # Memory scales roughly quadratically with sequence length and linearly with batch size
    length_factor = (max_length / 128) ** 1.5  # Slightly less than quadratic
    batch_factor = batch_size

    # Gradient accumulation doesn't increase peak memory significantly
    # but we add a small factor for intermediate storage
    grad_accum_factor = 1.0 + (gradient_accumulation_steps - 1) * 0.1

    estimated_memory = base_mem * length_factor * batch_factor * grad_accum_factor

    return estimated_memory


def suggest_memory_optimizations(
    batch_size: int, max_length: int, model_name: str, gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """Suggest memory optimizations for given parameters.

    Args:
        batch_size: Current batch size
        max_length: Current max length
        model_name: Model name
        gradient_accumulation_steps: Current gradient accumulation steps

    Returns:
        Dictionary with optimization suggestions
    """
    estimated_memory = estimate_memory_usage(
        batch_size, max_length, model_name, gradient_accumulation_steps
    )
    gpu_info = get_gpu_memory_info()

    suggestions = {
        "estimated_memory_gb": estimated_memory,
        "available_memory_gb": gpu_info["free"],
        "likely_oom": estimated_memory > gpu_info["free"] * 0.8,  # 80% threshold
        "optimizations": [],
    }

    if suggestions["likely_oom"]:
        # Suggest batch size reduction
        if batch_size > 8:
            new_batch_size = max(8, batch_size // 2)
            suggestions["optimizations"].append(
                {
                    "type": "reduce_batch_size",
                    "current": batch_size,
                    "suggested": new_batch_size,
                    "description": f"Reduce batch size from {batch_size} to {new_batch_size}",
                }
            )

        # Suggest sequence length reduction
        if max_length > 256:
            new_max_length = max(256, max_length // 2)
            suggestions["optimizations"].append(
                {
                    "type": "reduce_max_length",
                    "current": max_length,
                    "suggested": new_max_length,
                    "description": f"Reduce max length from {max_length} to {new_max_length}",
                }
            )

        # Suggest gradient checkpointing
        suggestions["optimizations"].append(
            {
                "type": "enable_gradient_checkpointing",
                "description": "Enable gradient checkpointing to trade compute for memory",
            }
        )

        # Suggest increasing gradient accumulation
        if gradient_accumulation_steps < 8:
            new_grad_accum = min(8, gradient_accumulation_steps * 2)
            suggestions["optimizations"].append(
                {
                    "type": "increase_gradient_accumulation",
                    "current": gradient_accumulation_steps,
                    "suggested": new_grad_accum,
                    "description": f"Increase gradient accumulation from {gradient_accumulation_steps} to {new_grad_accum}",
                }
            )

    return suggestions


def apply_dynamic_batch_size_reduction(original_batch_size: int, model_name: str) -> int:
    """Dynamically reduce batch size based on model type and available memory.

    Args:
        original_batch_size: Original batch size
        model_name: Model name

    Returns:
        Reduced batch size
    """
    gpu_info = get_gpu_memory_info()

    # Conservative reduction based on available memory
    if gpu_info["free"] < 8.0:  # Less than 8GB free
        reduction_factor = 0.5
    elif gpu_info["free"] < 12.0:  # Less than 12GB free
        reduction_factor = 0.7
    else:
        reduction_factor = 0.8

    # Additional reduction for memory-intensive models
    if "deberta" in model_name.lower():
        reduction_factor *= 0.7

    new_batch_size = max(1, int(original_batch_size * reduction_factor))
    return new_batch_size


def set_memory_efficient_environment() -> None:
    """Set environment variables for memory efficiency."""
    # Enable expandable segments for better memory management
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Reduce memory fragmentation
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + ",max_split_size_mb:128",
    )


def log_memory_usage(stage: str, trial_number: Optional[int] = None) -> None:
    """Log current memory usage.

    Args:
        stage: Stage description (e.g., "before_training", "after_training")
        trial_number: Optional trial number for HPO
    """
    if not torch.cuda.is_available():
        return

    memory_info = get_gpu_memory_info()
    prefix = f"Trial {trial_number} - " if trial_number is not None else ""

    print(f"🔍 {prefix}Memory usage ({stage}):")
    print(f"  Total: {memory_info['total']:.1f} GB")
    print(f"  Allocated: {memory_info['allocated']:.1f} GB")
    print(f"  Cached: {memory_info['cached']:.1f} GB")
    print(f"  Free: {memory_info['free']:.1f} GB")


class MemoryMonitor:
    """Context manager for monitoring memory usage during training."""

    def __init__(self, trial_number: Optional[int] = None, clear_cache: bool = True):
        """Initialize memory monitor.

        Args:
            trial_number: Optional trial number for logging
            clear_cache: Whether to clear cache on exit
        """
        self.trial_number = trial_number
        self.clear_cache = clear_cache
        self.start_memory = None

    def __enter__(self):
        """Enter context manager."""
        if torch.cuda.is_available():
            clear_gpu_cache()  # Start with clean cache
            self.start_memory = get_gpu_memory_info()
            log_memory_usage("start", self.trial_number)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if torch.cuda.is_available():
            end_memory = get_gpu_memory_info()
            log_memory_usage("end", self.trial_number)

            if self.start_memory:
                memory_diff = end_memory["allocated"] - self.start_memory["allocated"]
                print(f"📊 Memory change: {memory_diff:+.1f} GB")

            if self.clear_cache:
                clear_gpu_cache()
