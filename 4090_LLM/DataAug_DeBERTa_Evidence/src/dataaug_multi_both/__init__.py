from __future__ import annotations

from importlib import metadata
from pathlib import Path

PKG_NAME = "dataaug_multi_both"
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR.parent / "configs"

try:
    __version__ = metadata.version("dataaug-multi-both")
except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__", "PKG_NAME", "ROOT_DIR", "CONFIG_DIR"]
