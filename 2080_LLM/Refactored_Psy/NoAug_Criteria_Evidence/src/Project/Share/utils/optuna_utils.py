from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import optuna

_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_STORAGE_URI = f"sqlite:///{(_ROOT / 'optuna.db').resolve()}"


def get_optuna_storage(storage: Optional[str] = None) -> str:
    """Return the storage URI, defaulting to the repository-level SQLite DB."""
    uri = storage or _DEFAULT_STORAGE_URI
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return uri


def create_study(
    study_name: str,
    storage: Optional[str] = None,
    direction: str = "minimize",
    load_if_exists: bool = True,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    **kwargs: Any,
) -> optuna.Study:
    """Create (or reuse) an Optuna study backed by the shared SQLite DB."""
    storage_uri = get_optuna_storage(storage)
    return optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        load_if_exists=load_if_exists,
        sampler=sampler,
        pruner=pruner,
        **kwargs,
    )


def load_study(study_name: str, storage: Optional[str] = None) -> optuna.Study:
    """Load an existing Optuna study from the shared SQLite DB."""
    storage_uri = get_optuna_storage(storage)
    return optuna.load_study(study_name=study_name, storage=storage_uri)


__all__ = ["get_optuna_storage", "create_study", "load_study"]
