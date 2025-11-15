#!/usr/bin/env python
"""Multi-stage HPO driver (S0 → S1 → S2 → Refit)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import optuna

from psy_agents_noaug.hpo import (
    DEFAULT_REPORT_DIR,
    ObjectiveBuilder,
    ObjectiveSettings,
    SearchSpace,
    SpaceConstraints,
    create_pruner,
    create_sampler,
    evaluation,
    resolve_profile,
    resolve_storage,
    utils,
)

LOGGER = logging.getLogger("run_hpo_stage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-stage HPO for an agent")
    parser.add_argument(
        "--agent",
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
    )
    parser.add_argument("--storage", default=None)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--outdir", default=os.getenv("HPO_OUTDIR", "./_runs"))
    parser.add_argument("--profile", default=os.getenv("HPO_PROFILE", "noaug"))
    parser.add_argument("--seeds", default=os.getenv("HPO_SEEDS", "1"))
    parser.add_argument(
        "--stage0-trials", type=int, default=int(os.getenv("HPO_TRIALS_S0", "64"))
    )
    parser.add_argument(
        "--stage1-trials", type=int, default=int(os.getenv("HPO_TRIALS_S1", "32"))
    )
    parser.add_argument(
        "--stage2-trials", type=int, default=int(os.getenv("HPO_TRIALS_S2", "16"))
    )
    parser.add_argument(
        "--stage0-epochs", type=int, default=int(os.getenv("HPO_EPOCHS_S0", "3"))
    )
    parser.add_argument(
        "--stage1-epochs", type=int, default=int(os.getenv("HPO_EPOCHS_S1", "6"))
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=int(os.getenv("HPO_EPOCHS_S2", "10"))
    )
    parser.add_argument(
        "--refit-epochs", type=int, default=int(os.getenv("HPO_REFIT_EPOCHS", "12"))
    )
    parser.add_argument(
        "--patience", type=int, default=int(os.getenv("HPO_PATIENCE", "2"))
    )
    parser.add_argument(
        "--max-samples", type=int, default=int(os.getenv("HPO_MAX_SAMPLES", "512"))
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--log-level", default=os.getenv("HPO_LOG_LEVEL", "INFO"))
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_seeds(seed_arg: str) -> list[int]:
    seeds: list[int] = []
    for chunk in seed_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    return seeds or [1]


def _create_study(
    *,
    storage: str,
    study_name: str,
    sampler,
    pruner,
) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize"],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def _derive_constraints(
    trials: Sequence[optuna.trial.FrozenTrial],
    *,
    categorical_keys: Sequence[str],
    float_keys: Sequence[str],
    int_keys: Sequence[str],
    span_factor: float,
) -> SpaceConstraints:
    constraints = SpaceConstraints()
    configs: list[dict[str, Any]] = []
    for trial in trials:
        payload = trial.user_attrs.get("config_json")
        if not payload:
            continue
        try:
            configs.append(json.loads(payload))
        except Exception:
            continue

    if not configs:
        return constraints

    for key in categorical_keys:
        values = [cfg.get(key) for cfg in configs if key in cfg]
        if not values:
            continue
        most_common = [value for value, _ in Counter(values).most_common(3)]
        constraints.categorical[key] = most_common

    for key in float_keys:
        collected = [float(cfg.get(key)) for cfg in configs if key in cfg]
        if not collected:
            continue
        low = min(collected)
        high = max(collected)
        span = (high - low) or max(abs(low), 1e-6)
        pad = span * span_factor
        lower = low - pad
        upper = high + pad
        if key in {"optim.lr", "optim.weight_decay"}:
            lower = max(lower, 1e-6)
        if lower <= 0 < upper:
            lower = max(1e-6, upper * 0.1)
        if upper <= lower:
            upper = lower * 1.1
        constraints.floats[key] = (lower, upper)

    for key in int_keys:
        collected = [int(cfg.get(key)) for cfg in configs if key in cfg]
        if not collected:
            continue
        constraints.ints[key] = (min(collected), max(collected))

    return constraints


def _run_stage(
    *,
    agent: str,
    study_name: str,
    storage: str,
    mlflow_uri: str | None,
    outdir: Path,
    profile: str,
    trials: int,
    epochs: int,
    seeds: list[int],
    patience: int,
    max_samples: int | None,
    topk: int,
    constraints: SpaceConstraints,
) -> optuna.Study:
    sampler = create_sampler(multi_objective=False, seed=2025)
    pruner = create_pruner(
        "asha", min_resource=1, max_resource=max(1, epochs), reduction_factor=3
    )
    study = _create_study(
        storage=storage,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    search_space = SearchSpace(agent)
    settings = ObjectiveSettings(
        agent=agent,
        study=study_name,
        outdir=outdir,
        epochs=epochs,
        seeds=seeds,
        patience=patience,
        max_samples=max_samples,
        multi_objective=False,
        topk=topk,
        mlflow_uri=mlflow_uri,
        mlflow_experiment=f"{profile}-{agent}-{study_name}",
    )
    objective = ObjectiveBuilder(search_space, settings, constraints=constraints)

    LOGGER.info("Stage %s -> trials=%s epochs=%s", study_name, trials, epochs)
    study.optimize(
        objective, n_trials=trials, gc_after_trial=True, catch=(RuntimeError,)
    )
    return study


def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    trials = [t for t in study.get_trials(deepcopy=False) if t.values is not None]
    trials.sort(key=lambda t: -(t.user_attrs.get("primary", t.value or 0.0)))
    if not trials:
        raise RuntimeError("No successful trials to refit")
    return trials[0]


def _refit(
    *,
    agent: str,
    params: dict[str, Any],
    epochs: int,
    seeds: list[int],
    patience: int,
    max_samples: int | None,
) -> dict[str, float]:
    LOGGER.info("Refitting best configuration with epochs=%s seeds=%s", epochs, seeds)
    return evaluation.run_experiment(
        agent,
        params,
        epochs=epochs,
        seeds=seeds,
        patience=patience,
        max_samples=max_samples,
    )


def main() -> None:  # pragma: no cover - CLI orchestration
    args = parse_args()
    setup_logging(args.log_level)

    storage = resolve_storage(args.storage)
    profile = resolve_profile(args.profile)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    max_samples = args.max_samples if args.max_samples > 0 else None

    stage0_name = f"{profile}-{args.agent}-stage0"
    stage1_name = f"{profile}-{args.agent}-stage1"
    stage2_name = f"{profile}-{args.agent}-stage2"

    study0 = _run_stage(
        agent=args.agent,
        study_name=stage0_name,
        storage=storage,
        mlflow_uri=args.mlflow_uri,
        outdir=outdir,
        profile=profile,
        trials=args.stage0_trials,
        epochs=args.stage0_epochs,
        seeds=seeds,
        patience=args.patience,
        max_samples=max_samples,
        topk=args.topk,
        constraints=SpaceConstraints(),
    )

    top_s0 = utils.collect_top_trials(study0, "primary", limit=24)
    constraints_s1 = _derive_constraints(
        top_s0,
        categorical_keys=["model.name", "head.pooling", "optim.name", "sched.name"],
        float_keys=["optim.lr", "optim.weight_decay", "head.dropout"],
        int_keys=["train.batch_size"],
        span_factor=0.2,
    )

    study1 = _run_stage(
        agent=args.agent,
        study_name=stage1_name,
        storage=storage,
        mlflow_uri=args.mlflow_uri,
        outdir=outdir,
        profile=profile,
        trials=args.stage1_trials,
        epochs=args.stage1_epochs,
        seeds=seeds,
        patience=args.patience,
        max_samples=max_samples,
        topk=args.topk,
        constraints=constraints_s1,
    )

    top_s1 = utils.collect_top_trials(study1, "primary", limit=12)
    constraints_s2 = _derive_constraints(
        top_s1,
        categorical_keys=["model.name", "head.pooling", "optim.name", "sched.name"],
        float_keys=[
            "optim.lr",
            "optim.weight_decay",
            "head.dropout",
            "null.threshold",
            "null.ratio",
            "null.temperature",
        ],
        int_keys=["train.batch_size"],
        span_factor=0.1,
    )

    study2 = _run_stage(
        agent=args.agent,
        study_name=stage2_name,
        storage=storage,
        mlflow_uri=args.mlflow_uri,
        outdir=outdir,
        profile=profile,
        trials=args.stage2_trials,
        epochs=args.stage2_epochs,
        seeds=seeds,
        patience=args.patience,
        max_samples=max_samples,
        topk=args.topk,
        constraints=constraints_s2,
    )

    best_trial = _select_best_trial(study2)
    params = json.loads(best_trial.user_attrs["config_json"])
    refit_seeds = parse_seeds(os.getenv("HPO_REFIT_SEEDS", "1,2,3"))
    refit_metrics = _refit(
        agent=args.agent,
        params=params,
        epochs=args.refit_epochs,
        seeds=refit_seeds,
        patience=max(args.patience, 2),
        max_samples=max_samples,
    )

    report_lines = [
        "# Multi-stage HPO Summary",
        f"Agent: {args.agent}",
        f"Stage0 trials: {len(study0.trials)}",
        f"Stage1 trials: {len(study1.trials)}",
        f"Stage2 trials: {len(study2.trials)}",
        "",
        "## Refit Metrics",
    ]
    for key, value in refit_metrics.items():
        report_lines.append(f"- {key}: {value:.4f}")
    utils.write_report(
        args.agent, "multistage", DEFAULT_REPORT_DIR, "\n".join(report_lines)
    )


if __name__ == "__main__":
    main()
