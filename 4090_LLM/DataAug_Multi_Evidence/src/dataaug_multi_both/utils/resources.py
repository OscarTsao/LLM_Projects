from __future__ import annotations

import os
from typing import Any, Tuple

import torch


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def recommend_num_workers() -> int:
    """Recommend DataLoader workers using 90% CPU rule capped at 16.

    Returns a non-negative integer; can be zero on very small machines.
    """
    cores = _cpu_count()
    workers = int(cores * 0.9)
    workers = max(0, workers)
    workers = min(16, workers)
    return workers


def default_pin_memory() -> bool:
    """Pin memory only when CUDA is available."""
    return bool(torch.cuda.is_available())


def recommend_prefetch_factor(num_workers: int) -> int | None:
    """Simple heuristic for prefetch factor based on workers.

    - >=4 workers -> 4
    - >0 workers  -> 2
    - 0 workers   -> None
    """
    if num_workers >= 4:
        return 4
    if num_workers > 0:
        return 2
    return None


def build_dataloader_kwargs(
    resources_cfg: dict[str, Any] | None,
    training_cfg: dict[str, Any] | None,
) -> Tuple[dict[str, Any], dict[str, Any]]:
    """Build unified DataLoader kwargs (base and train-specific).

    Returns (base_kwargs, train_kwargs). The train_kwargs include drop_last.
    """
    resources_cfg = resources_cfg or {}
    training_cfg = training_cfg or {}

    cuda_available = torch.cuda.is_available()

    # num_workers (90% rule capped at 16) with config overrides
    configured_workers = training_cfg.get("num_workers")
    if configured_workers is None:
        configured_workers = resources_cfg.get("num_workers", recommend_num_workers())
    num_workers = max(int(configured_workers), 0)

    # pin_memory: default True if CUDA available; allow overrides
    pin_memory = bool(
        training_cfg.get(
            "pin_memory",
            resources_cfg.get("pin_memory", default_pin_memory()),
        )
    )

    # persistent_workers: True by default when workers > 0; allow overrides
    persistent_workers = (
        bool(
            training_cfg.get(
                "persistent_workers",
                resources_cfg.get("persistent_workers", True),
            )
        )
        and num_workers > 0
    )

    # prefetch_factor heuristic with overrides
    pf_override = training_cfg.get("prefetch_factor", resources_cfg.get("prefetch_factor"))
    prefetch_factor = int(pf_override) if pf_override is not None else recommend_prefetch_factor(num_workers)

    base_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and cuda_available,
    }
    if pin_memory and cuda_available:
        base_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        if persistent_workers:
            base_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            base_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_kwargs = dict(base_kwargs)
    train_kwargs["drop_last"] = bool(training_cfg.get("drop_last", False))

    return base_kwargs, train_kwargs


def summarize_system() -> dict[str, Any]:
    """Provide a small system summary useful for logging."""
    gpu = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            gpu["current_device"] = idx
            gpu["name"] = torch.cuda.get_device_name(idx)
            gpu["capability"] = torch.cuda.get_device_capability(idx)
            gpu["total_mem_bytes"] = torch.cuda.get_device_properties(idx).total_memory
        except Exception:
            pass
    return {
        "cpu_cores": _cpu_count(),
        "recommended_workers": recommend_num_workers(),
        "gpu": gpu,
    }

