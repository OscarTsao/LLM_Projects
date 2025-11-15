#!/usr/bin/env python
"""Export Top-K Optuna trials to JSON/CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import optuna

from psy_agents_noaug.hpo import resolve_storage
from psy_agents_noaug.hpo import utils as hpo_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Top-K Optuna trials")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--outdir", default=os.getenv("HPO_OUTDIR", "./_runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    storage = resolve_storage(args.storage)
    study = optuna.load_study(study_name=args.study, storage=storage)
    trials = hpo_utils.collect_top_trials(study, "primary", args.topk)
    topk_store = hpo_utils.TopKStore(
        outdir=Path(args.outdir),
        agent=args.agent,
        study=args.study,
        k=args.topk,
    )
    for trial in trials:
        topk_store.record(hpo_utils.trial_to_summary(trial))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
