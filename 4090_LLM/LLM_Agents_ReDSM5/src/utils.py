import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

import numpy as np
import torch
import yaml


DEFAULT_LOGGER_NAME = "redsm5"


def setup_logger(name: str = DEFAULT_LOGGER_NAME, log_level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(log_level)
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _deep_update(base: MutableMapping[str, Any], updates: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, MutableMapping) and key in base and isinstance(base[key], MutableMapping):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_dicts(base: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if overrides is None:
        return dict(base)
    merged = dict(base)
    return _deep_update(merged, overrides)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour when possible
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class DeviceConfig:
    device: torch.device
    n_gpu: int
    bf16: bool
    fp16: bool


def detect_device(prefer_bf16: bool = True) -> DeviceConfig:
    """Detect CUDA availability and preferred dtype support."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        major, minor = torch.cuda.get_device_capability(0)
        supports_bf16 = major >= 8  # Ampere+
        bf16 = prefer_bf16 and supports_bf16
        fp16 = not bf16
        return DeviceConfig(device=device, n_gpu=torch.cuda.device_count(), bf16=bf16, fp16=fp16)
    device = torch.device("cpu")
    return DeviceConfig(device=device, n_gpu=0, bf16=False, fp16=False)


def get_git_hash(default: str = "unknown") -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return default


def chunk_iterable(iterable: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def resolve_samples_limit(limit: Optional[int], total: int) -> int:
    if limit is None or limit <= 0:
        return total
    return min(limit, total)


def maybe_float(value: Any) -> float:
    if isinstance(value, (float, int, np.floating)):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value)}")


__all__ = [
    "setup_logger",
    "load_yaml",
    "save_yaml",
    "save_json",
    "load_json",
    "merge_dicts",
    "ensure_dir",
    "set_seed",
    "detect_device",
    "chunk_iterable",
    "get_git_hash",
    "resolve_samples_limit",
    "maybe_float",
]
