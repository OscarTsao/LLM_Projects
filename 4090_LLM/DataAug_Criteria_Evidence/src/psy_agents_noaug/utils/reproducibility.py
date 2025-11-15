"""Reproducibility utilities for deterministic training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False):
    """
    Set random seed for reproducibility across all libraries.

    Best Practices (2025):
    - For reproducibility: deterministic=True, cudnn_benchmark=False
    - For speed: deterministic=False, cudnn_benchmark=True
    - torch.use_deterministic_algorithms affects all PyTorch ops
    - cudnn.deterministic only affects cuDNN convolutions

    Args:
        seed: Random seed
        deterministic: Enable deterministic mode for all CUDA operations
        cudnn_benchmark: Enable CuDNN benchmark mode (faster but non-deterministic)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN settings
    if deterministic:
        # Full determinism - slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Performance mode - faster but non-deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = cudnn_benchmark

    # Environment variable for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch deterministic algorithms (>= 1.7)
    # This makes more operations deterministic beyond just cuDNN
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception as e:
            # Some operations don't have deterministic implementations
            # Set warn_only mode to continue training
            if deterministic:
                try:
                    torch.use_deterministic_algorithms(deterministic, warn_only=True)
                    print(
                        f"Warning: Some operations may still be non-deterministic: {e}"
                    )
                except Exception:
                    pass

    print(f"Random seed set to {seed}")
    print(f"Deterministic mode: {deterministic}")
    print(f"CuDNN benchmark: {cudnn_benchmark}")
    if deterministic:
        print("Note: Deterministic mode may reduce training speed")


def get_device(prefer_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Get the appropriate device for training with hardware info.

    Args:
        prefer_cuda: Prefer CUDA if available
        device_id: Specific CUDA device ID to use

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        print(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB"
        )

        # Check for TF32 support (Ampere and later)
        compute_capability = torch.cuda.get_device_capability(device_id)
        if compute_capability[0] >= 8:  # Ampere (80) and later
            print(
                f"Compute Capability: {compute_capability[0]}.{compute_capability[1]} (Ampere+)"
            )
            print("TF32 and BFloat16 are supported on this GPU")
        else:
            print(
                f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}"
            )
    else:
        device = torch.device("cpu")
        print("Using CPU device")
        print(f"CPU cores: {os.cpu_count()}")

    return device


def get_optimal_dataloader_kwargs(
    device: torch.device,
    num_workers: int = None,
    pin_memory: bool = None,
    persistent_workers: bool = None,
    prefetch_factor: int = 2,
) -> dict:
    """
    Get optimal DataLoader kwargs based on hardware.

    Best Practices (2025):
    - num_workers: Start with 2x CPU cores per GPU, tune for best performance
    - pin_memory: True for GPU training (faster host->device transfer)
    - persistent_workers: True to avoid reinitializing workers each epoch
    - prefetch_factor: Default of 2 is usually optimal

    Args:
        device: Training device
        num_workers: Number of worker processes (auto-detect if None)
        pin_memory: Pin memory for faster GPU transfer (auto-detect if None)
        persistent_workers: Keep workers alive between epochs (auto-detect if None)
        prefetch_factor: Batches prefetched per worker

    Returns:
        Dictionary of optimal DataLoader kwargs
    """
    is_cuda = device.type == "cuda"

    # Auto-detect num_workers if not specified
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        if is_cuda:
            # Start with 2x CPU cores per GPU as a good default
            num_workers = min(cpu_count, 8)  # Cap at 8 to avoid diminishing returns
        else:
            num_workers = min(cpu_count // 2, 4)  # Use fewer for CPU training

    # Auto-detect pin_memory if not specified
    if pin_memory is None:
        pin_memory = is_cuda  # Always True for GPU training

    # Auto-detect persistent_workers if not specified
    if persistent_workers is None:
        persistent_workers = num_workers > 0  # Use if workers are enabled

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }

    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    print("\nDataLoader Configuration:")
    print(f"  num_workers: {kwargs.get('num_workers', 0)}")
    print(f"  pin_memory: {kwargs.get('pin_memory', False)}")
    print(f"  persistent_workers: {kwargs.get('persistent_workers', False)}")
    if "prefetch_factor" in kwargs:
        print(f"  prefetch_factor: {kwargs['prefetch_factor']}")

    return kwargs


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
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")

    print(f"Number of CPU cores: {os.cpu_count()}")
    print("=" * 70 + "\n")
