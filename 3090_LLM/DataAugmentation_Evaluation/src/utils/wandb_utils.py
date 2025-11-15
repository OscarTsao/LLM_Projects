"""Weights & Biases utilities to keep logging optional and ergonomic."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore


def _resolve_mode(explicit_mode: str | None = None) -> str:
    """Resolve the W&B mode with environment-based fallback."""
    if explicit_mode:
        return explicit_mode
    return os.getenv("WANDB_MODE", "online")


def is_enabled(explicit_mode: str | None = None) -> bool:
    """Check whether W&B logging should be active."""
    if not WANDB_AVAILABLE:
        return False
    mode = _resolve_mode(explicit_mode)
    return mode.lower() not in {"disabled", "off", "none"}


@contextmanager
def start_run(
    enabled: bool,
    project: str,
    entity: str | None = None,
    name: str | None = None,
    config: Dict[str, Any] | None = None,
    tags: Iterable[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str | None = None,
    reinit: bool = True,
) -> Iterator[None]:
    """Context manager that wraps `wandb.init` but degrades gracefully."""
    if not enabled or not is_enabled(mode):
        yield
        return

    settings = wandb.Settings(start_method="thread")
    run = wandb.init(  # type: ignore[assignment]
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=list(tags) if tags else None,
        group=group,
        job_type=job_type,
        mode=_resolve_mode(mode),
        reinit=reinit,
        settings=settings,
    )
    try:
        yield
    finally:
        if run is not None:
            run.finish()


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    """Log metrics into the active W&B run."""
    if not is_enabled() or wandb.run is None:
        return
    wandb.log(dict(metrics), step=step)


def log_config(config: Dict[str, Any]) -> None:
    """Update the current run configuration."""
    if not is_enabled() or wandb.run is None:
        return
    wandb.config.update(config, allow_val_change=True)


def log_summary(summary: Dict[str, Any]) -> None:
    """Attach summary metrics to the active run."""
    if not is_enabled() or wandb.run is None:
        return
    wandb.run.summary.update(summary)
