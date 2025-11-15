from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from omegaconf import OmegaConf

from dataaug_multi_both import CONFIG_DIR, ROOT_DIR

DEFAULT_CONFIG_FILES = (
    CONFIG_DIR / "train.yaml",
    CONFIG_DIR / "data.yaml",
    CONFIG_DIR / "augmentation.yaml",
    CONFIG_DIR / "mlflow.yaml",
    CONFIG_DIR / "hpo.yaml",
)


def _ensure_config_files(paths: Iterable[Path]) -> list[Path]:
    resolved = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        resolved.append(path)
    return resolved


def load_project_config(
    extra_files: Sequence[os.PathLike[str] | str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict:
    """Load and merge the project configuration.

    Parameters
    ----------
    extra_files:
        Additional config paths to merge after the defaults.
    overrides:
        Optional dictionary merged last for programmatic overrides (e.g. Optuna trial).
    """

    files = list(DEFAULT_CONFIG_FILES)
    if extra_files:
        files.extend(Path(f) if not isinstance(f, Path) else f for f in extra_files)

    merged = OmegaConf.create()
    for path in _ensure_config_files(files):
        cfg = OmegaConf.load(path)
        merged = OmegaConf.merge(merged, cfg)

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.create(overrides))

    return OmegaConf.to_container(merged, resolve=True)


def flatten_dict(dct: Mapping[str, object], parent_key: str = "", sep: str = ".") -> dict[str, object]:
    """Flatten a nested dictionary for logging.

    Examples
    --------
    >>> flatten_dict({"a": {"b": 1}})
    {'a.b': 1}
    """

    items: MutableMapping[str, object] = {}
    for key, value in dct.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_dict(value, parent_key=new_key, sep=sep))
        else:
            items[new_key] = value
    return dict(items)
