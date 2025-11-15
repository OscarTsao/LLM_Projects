"""Reproducibility utilities for deterministic training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed
        deterministic: Enable deterministic mode for CUDA operations
        cudnn_benchmark: Enable CuDNN benchmark mode (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = cudnn_benchmark
    
    # Environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # For PyTorch >= 1.7
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception:
            pass
    
    print(f"Random seed set to {seed}")
    print(f"Deterministic mode: {deterministic}")
    print(f"CuDNN benchmark: {cudnn_benchmark}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        prefer_cuda: Prefer CUDA if available
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def print_system_info():
    """Print system and environment information."""
    print("\n" + "=" * 70)
    print("System Information".center(70))
    print("=" * 70)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"Number of CPU cores: {os.cpu_count()}")
    print("=" * 70 + "\n")
