"""Reproducibility utilities for deterministic training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Random seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get torch device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
    }


def print_model_info(model: torch.nn.Module):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    param_counts = count_parameters(model)
    
    print("\nModel Information:")
    print(f"  Total parameters:     {param_counts['total']:,}")
    print(f"  Trainable parameters: {param_counts['trainable']:,}")
    print(f"  Frozen parameters:    {param_counts['frozen']:,}")
    print()
