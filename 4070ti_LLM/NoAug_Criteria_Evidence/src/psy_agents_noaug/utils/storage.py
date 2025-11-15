"""Utilities for managing experiment storage and pruning artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - mlflow optional during tests
    MlflowClient = None  # type: ignore[assignment]


def ensure_directory(path: Path) -> Path:
    """Create path (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_filestore(tracking_uri: str | None) -> Path | None:
    """Return local MLflow filestore path if tracking URI is file-based."""
    if not tracking_uri:
        return None
    parsed = urlparse(tracking_uri)
    if parsed.scheme != "file":
        return None
    if parsed.netloc and parsed.netloc != "":  # Support file://hostname/path
        candidate = Path(f"{parsed.netloc}{parsed.path}")
    else:
        candidate = Path(parsed.path)
    return candidate if candidate.exists() else candidate.resolve()


def archive_trial_summary(
    destination: Path,
    trial_number: int,
    params: dict,
    metrics: dict,
    extra_attrs: dict | None = None,
) -> None:
    """Save key data for a trial into destination directory."""
    ensure_directory(destination)
    with (destination / "params.json").open("w", encoding="utf-8") as fp:
        json.dump(params, fp, indent=2, sort_keys=True)
    with (destination / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, sort_keys=True)
    if extra_attrs:
        with (destination / "attrs.json").open("w", encoding="utf-8") as fp:
            json.dump(extra_attrs, fp, indent=2, sort_keys=True)


def download_mlflow_artifacts(
    client: MlflowClient,
    run_id: str,
    destination: Path,
) -> None:
    """Download all MLflow artifacts for run into destination."""
    ensure_directory(destination)
    try:
        client.download_artifacts(run_id, "", str(destination))
    except Exception:
        # Best-effort only; downstream code still has params/metrics snapshot.
        pass


def cleanup_mlflow_runs(
    run_ids: Iterable[str],
    *,
    tracking_uri: str | None,
    preserve: set[str] | None = None,
) -> None:
    """
    Delete MLflow runs (and local artifacts for file stores) except those preserved.

    Args:
        run_ids: Iterable of run IDs to consider for deletion.
        tracking_uri: Active MLflow tracking URI (used for local artifact cleanup).
        preserve: Run IDs to keep (even if present in run_ids).
    """
    if MlflowClient is None:
        return

    client = MlflowClient()
    preserve = preserve or set()
    base_dir = _resolve_filestore(tracking_uri)

    for run_id in run_ids:
        if not run_id or run_id in preserve:
            continue
        try:
            run = client.get_run(run_id)
        except Exception:
            continue
        try:
            client.delete_run(run_id)
        except Exception:
            pass

        if not base_dir:
            continue

        exp_dir = base_dir / run.info.experiment_id / run_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)


def prune_directory_except(path: Path, keep_filenames: set[str]) -> None:
    """
    Remove files inside path unless their name is listed in keep_filenames.

    Directories not in keep_filenames are removed recursively.
    """
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.name in keep_filenames:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except OSError:
                pass
