from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import optuna

from dataaug_multi_both.hpo.two_stage import STAGE2_CONFIG, simulate_configuration


def _prepare_params(trial: optuna.trial.FrozenTrial) -> dict:
    params = dict(trial.params)
    for key in ("batch_size", "gradient_accumulation_steps", "effective_batch_size"):
        if key in trial.user_attrs and key not in params:
            params[key] = trial.user_attrs[key]
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-simulate best Stage-2 configuration.")
    parser.add_argument("--storage", required=True, help="Optuna storage URI.")
    parser.add_argument(
        "--study-name",
        default=STAGE2_CONFIG.study_name,
        help="Stage-2 study name (defaults to stage2_exploit).",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments/exports",
        help="Directory to store export simulations.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/exports",
        help="Directory to write summary JSON.",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to evaluate.")
    parser.add_argument("--base-seed", type=int, default=9000, help="Base seed for simulations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    if study.best_trial is None:
        raise RuntimeError("Study does not contain a best trial.")

    params = _prepare_params(study.best_trial)
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.base_seed + idx for idx in range(args.seeds)]
    outcomes = []
    for index, seed in enumerate(seeds):
        trial_id = f"export_seed{index:02d}"
        outcome = simulate_configuration(
            params,
            config=STAGE2_CONFIG,
            experiments_dir=experiments_dir,
            seed=seed,
            trial_id=trial_id,
        )
        outcomes.append({
            "seed": seed,
            "score": outcome.score,
            "report_path": str(outcome.report_path),
            "checkpoint_path": str(outcome.checkpoint_path),
        })

    scores = [item["score"] for item in outcomes]
    score_mean = mean(scores)
    score_std = pstdev(scores) if len(scores) > 1 else 0.0
    adjusted_score = score_mean - 0.1 * score_std

    summary = {
        "study_name": study.study_name,
        "trial_number": study.best_trial.number,
        "params": params,
        "seeds": seeds,
        "outcomes": outcomes,
        "score_mean": score_mean,
        "score_std": score_std,
        "adjusted_score": adjusted_score,
    }

    summary_path = output_dir / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Export completed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
