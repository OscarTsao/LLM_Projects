from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig


def resolve_config_paths(cfg: DictConfig, base_dir: Optional[Path] = None) -> DictConfig:
    """Ensure config paths are absolute and update project root metadata."""
    if base_dir is None:
        if "paths" in cfg and "project_root" in cfg.paths:
            base_dir = Path(cfg.paths.project_root)
        else:
            base_dir = Path(__file__).resolve().parents[1]
    base_dir = base_dir.resolve()

    cfg.paths.project_root = str(base_dir)

    def _to_abs(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return str(path)

    cfg.training.output_dir = _to_abs(cfg.training.output_dir)
    cfg.logging.artifact_location = _to_abs(cfg.logging.artifact_location)
    cfg.data.annotation_path = _to_abs(cfg.data.annotation_path)
    cfg.data.post_path = _to_abs(cfg.data.post_path)

    return cfg

