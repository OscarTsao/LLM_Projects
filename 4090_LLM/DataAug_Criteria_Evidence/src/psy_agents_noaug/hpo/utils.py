"""Utility helpers for the PSY Agents HPO stack."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

DEFAULT_STORAGE = "sqlite:///./_optuna/noaug.db"
DEFAULT_PROFILE = "noaug"
DEFAULT_TOPK = 10
DEFAULT_REPORT_DIR = Path("_runs") / "reports"


def resolve_storage(storage: str | None) -> str:
    """Return configured Optuna storage or the project default."""

    return storage or DEFAULT_STORAGE


def resolve_profile(profile: str | None) -> str:
    """Hydra/experiment profile name helper."""

    return profile or DEFAULT_PROFILE


def ensure_directory(path: str | Path) -> Path:
    """Create ``path`` (and parents) if it does not already exist."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@dataclass
class TrialSummary:
    """Summary payload for Top-K exports."""

    trial_number: int
    f1_macro: float
    ece: float | None
    logloss: float | None
    runtime_s: float | None
    seed_info: Sequence[int]
    params: dict[str, Any]
    artifact_uri: str | None = None
    started_at: float | None = None


@dataclass
class TopKStore:
    """Maintain running Top-K trials and persist them to disk."""

    outdir: Path
    agent: str
    study: str
    k: int = DEFAULT_TOPK
    entries: list[TrialSummary] = field(default_factory=list)

    def record(self, summary: TrialSummary) -> None:
        """Insert ``summary`` keeping only the best ``k`` entries."""

        self.entries.append(summary)
        self.entries.sort(  # Highest F1, then lowest ECE/logloss
            key=lambda s: (
                -s.f1_macro,
                s.ece if s.ece is not None else math.inf,
                s.logloss if s.logloss is not None else math.inf,
            )
        )
        del self.entries[self.k :]
        self.flush()

    def _targets(self) -> tuple[Path, Path]:
        topk_dir = self.outdir / "topk"
        topk_dir.mkdir(parents=True, exist_ok=True)
        json_path = topk_dir / f"{self.agent}_{self.study}_topk.json"
        csv_path = topk_dir / f"{self.agent}_{self.study}_topk.csv"
        return json_path, csv_path

    def flush(self) -> None:
        """Write Top-K summaries to JSON and CSV artefacts."""

        json_path, csv_path = self._targets()
        payload = [
            {
                "rank": idx + 1,
                "f1_macro": entry.f1_macro,
                "ece": entry.ece,
                "logloss": entry.logloss,
                "runtime_s": entry.runtime_s,
                "params": entry.params,
                "artifact_uri": entry.artifact_uri,
                "seed_info": list(entry.seed_info),
                "started_at": entry.started_at,
            }
            for idx, entry in enumerate(self.entries)
        ]
        json_path.write_text(json.dumps(payload, indent=2))

        fieldnames = [
            "rank",
            "f1_macro",
            "ece",
            "logloss",
            "runtime_s",
            "artifact_uri",
            "seed_info",
            "started_at",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for idx, entry in enumerate(self.entries):
                writer.writerow(
                    {
                        "rank": idx + 1,
                        "f1_macro": entry.f1_macro,
                        "ece": entry.ece,
                        "logloss": entry.logloss,
                        "runtime_s": entry.runtime_s,
                        "artifact_uri": entry.artifact_uri,
                        "seed_info": ";".join(map(str, entry.seed_info)),
                        "started_at": entry.started_at,
                    }
                )


def load_backbone_configs(config_dir: Path) -> list[str]:
    """Enumerate encoder names declared under ``configs/model``."""

    backbones: list[str] = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            import yaml

            data = yaml.safe_load(path.read_text())
        except Exception:  # pragma: no cover - invalid YAML handled downstream
            continue
        encoder = data.get("encoder_name") or data.get("model_name")
        if encoder:
            backbones.append(str(encoder))
    # Deduplicate while preserving order
    return list(dict.fromkeys(backbones))


def limit_dataframe(df, max_samples: int | None, seed: int):
    """Return a dataframe limited to ``max_samples`` rows if requested."""

    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def collect_top_trials(
    study: optuna.Study,
    metric_key: str,
    limit: int,
) -> list[optuna.trial.FrozenTrial]:
    """Return best ``limit`` trials sorted by ``metric_key`` attr."""

    trials = [t for t in study.get_trials(deepcopy=False) if t.values is not None]
    trials.sort(
        key=lambda t: (
            -(
                t.user_attrs.get(
                    metric_key, t.value if t.value is not None else -math.inf
                )
            )
        )
    )
    return trials[:limit]


def trial_to_summary(
    trial: optuna.trial.FrozenTrial,
    *,
    key_f1: str = "primary",
    key_ece: str = "ece",
    key_logloss: str = "logloss",
) -> TrialSummary:
    """Convert an Optuna trial to ``TrialSummary``."""

    params_json = trial.user_attrs.get("config_json")
    params_dict = json.loads(params_json) if isinstance(params_json, str) else {}
    seeds = trial.user_attrs.get("seeds")
    if isinstance(seeds, (list, tuple)):
        seed_info: Sequence[int] = list(map(int, seeds))
    elif isinstance(seeds, int):
        seed_info = [int(seeds)]
    else:
        seed_info = []
    return TrialSummary(
        trial_number=trial.number,
        f1_macro=float(trial.user_attrs.get(key_f1, trial.value or 0.0)),
        ece=_safe_float(trial.user_attrs.get(key_ece)),
        logloss=_safe_float(trial.user_attrs.get(key_logloss)),
        runtime_s=_safe_float(trial.user_attrs.get("runtime_s")),
        seed_info=seed_info,
        params=params_dict,
        artifact_uri=trial.user_attrs.get("artifact_uri"),
        started_at=_safe_float(trial.user_attrs.get("started_at")),
    )


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def write_report(agent: str, mode: str, report_dir: Path, content: str) -> Path:
    """Persist a Markdown report for ``agent`` and return its path."""

    ensure_directory(report_dir)
    path = report_dir / f"{agent}_{mode}.md"
    path.write_text(content)
    return path


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:  # pragma: no cover - defensive
        return default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:  # pragma: no cover - defensive
        return default


def env_list(name: str, default: Iterable[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


class TrialTimer:
    """Context manager recording wall-clock runtime for a trial."""

    def __enter__(self):  # pragma: no cover - trivial
        self.start = time.time()
        return self

    def __exit__(self, *exc):  # pragma: no cover - trivial
        self.end = time.time()
        self.duration = self.end - self.start
        return False
