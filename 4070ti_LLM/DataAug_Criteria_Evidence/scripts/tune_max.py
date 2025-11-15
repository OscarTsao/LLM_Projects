#!/usr/bin/env python
"""Maximal single-stage hyperparameter optimisation entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import optuna

from psy_agents_noaug.hpo import (
    DEFAULT_REPORT_DIR,
    ObjectiveBuilder,
    ObjectiveSettings,
    SearchSpace,
    SpaceConstraints,
    create_pruner,
    create_sampler,
    resolve_profile,
    resolve_storage,
)
from psy_agents_noaug.hpo import utils as hpo_utils

LOGGER = logging.getLogger("tune_max")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run maximal HPO for PSY Agents")
    parser.add_argument(
        "--agent",
        required=True,
        choices=["criteria", "evidence", "share", "joint"],
        help="Agent/architecture to optimise",
    )
    parser.add_argument(
        "--trials", type=int, default=int(os.getenv("HPO_TRIALS", "100"))
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("HPO_EPOCHS", "6")))
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--multi-objective", action="store_true")
    parser.add_argument("--timeout-min", type=int, default=None)
    parser.add_argument("--seeds", default=os.getenv("HPO_SEEDS", "1"))
    parser.add_argument("--outdir", default=os.getenv("HPO_OUTDIR", "./_runs"))
    parser.add_argument("--profile", default=os.getenv("HPO_PROFILE", "noaug"))
    parser.add_argument("--sampler", choices=["auto", "tpe", "nsga2"], default="auto")
    parser.add_argument("--pruner", choices=["asha", "median", "none"], default="asha")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--patience", type=int, default=int(os.getenv("HPO_PATIENCE", "2"))
    )
    parser.add_argument(
        "--max-samples", type=int, default=int(os.getenv("HPO_MAX_SAMPLES", "512"))
    )
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


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    storage = resolve_storage(args.storage)
    profile = resolve_profile(args.profile)
    study_name = args.study_name or f"{profile}-{args.agent}-max"
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting maximal HPO - agent=%s study=%s", args.agent, study_name)

    sampler = create_sampler(
        multi_objective=args.multi_objective,
        seed=2025,
        sampler=args.sampler,
    )

    pruner = create_pruner(
        args.pruner,
        min_resource=1,
        max_resource=max(1, args.epochs),
        reduction_factor=3,
    )

    directions = ["maximize"] if not args.multi_objective else ["maximize", "minimize"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    seeds = parse_seeds(args.seeds)
    search_space = SearchSpace(args.agent)

    settings = ObjectiveSettings(
        agent=args.agent,
        study=study_name,
        outdir=outdir,
        epochs=args.epochs,
        seeds=seeds,
        patience=max(1, args.patience),
        max_samples=args.max_samples if args.max_samples > 0 else None,
        multi_objective=args.multi_objective,
        topk=args.topk,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=f"{profile}-{args.agent}-{study_name}",
    )

    objective = ObjectiveBuilder(
        space=search_space,
        settings=settings,
        constraints=SpaceConstraints(),
    )

    timeout = args.timeout_min * 60 if args.timeout_min else None
    LOGGER.info("Optimising for up to %s trials (timeout=%s)", args.trials, timeout)

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=timeout,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    LOGGER.info("Study complete: %s", study_name)
    for idx, trial in enumerate(study.best_trials[:5]):
        value = trial.values if args.multi_objective else trial.value
        LOGGER.info("Top %d trial -> value=%s params=%s", idx + 1, value, trial.params)

    report_content = (
        f"# Maximal HPO Summary\n\n"
        f"Agent: {args.agent}\n"
        f"Study: {study_name}\n"
        f"Trials: {len(study.trials)}\n"
        f"Best value: {study.best_trials[0].values if args.multi_objective else study.best_value}\n"
    )
    hpo_utils.write_report(args.agent, "max", DEFAULT_REPORT_DIR, report_content)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
