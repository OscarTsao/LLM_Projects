from __future__ import annotations

import argparse
import json
import logging
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import os

_TEXTATTACK_CACHE_ENV = "TA_CACHE_DIR"
if _TEXTATTACK_CACHE_ENV not in os.environ:
    default_textattack_cache = Path(__file__).resolve().parents[2] / "artifacts" / "textattack_cache"
    default_textattack_cache.mkdir(parents=True, exist_ok=True)
    cache_dir = str(default_textattack_cache)
    os.environ[_TEXTATTACK_CACHE_ENV] = cache_dir
    os.environ.setdefault("TEXTATTACK_CACHE_DIR", cache_dir)

import optuna

from dataaug_multi_both.config import load_project_config
from dataaug_multi_both.evaluation.reporting import (
    evaluate_checkpoint,
    summarize_study_results,
)
from dataaug_multi_both.hpo.driver import run_two_stage_hpo
from dataaug_multi_both.training.train_loop import run_training_job

logger = logging.getLogger(__name__)


def _parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Overrides must be KEY=VALUE, got '{pair}'")
        key, raw_value = pair.split("=", 1)
        value: Any = raw_value
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            pass
        overrides[key] = value
    return overrides


def cmd_train(args: argparse.Namespace) -> None:
    overrides = _parse_overrides(args.override or [])
    if args.output_dir:
        overrides.setdefault("checkpoint", {})
        overrides["checkpoint"]["dir"] = str(args.output_dir)

    cfg = load_project_config(extra_files=args.config or [], overrides=overrides)
    metrics = run_training_job(cfg, resume=args.resume)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def cmd_tune(args: argparse.Namespace) -> None:
    overrides = _parse_overrides(args.override or [])
    summary = run_two_stage_hpo(
        model=args.model,
        trials_a=args.trials_a,
        epochs_a=args.epochs_a,
        trials_b=args.trials_b,
        epochs_b=args.epochs_b,
        k_top=args.k_top,
        global_seed=args.seed,
        timeout=args.timeout,
        config_files=args.config,
        config_overrides=overrides,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_eval(args: argparse.Namespace) -> None:
    overrides = _parse_overrides(args.override or [])
    cfg = load_project_config(extra_files=args.config or [], overrides=overrides)
    report = evaluate_checkpoint(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def cmd_summarize(args: argparse.Namespace) -> None:
    summarize_study_results(
        study_name=args.study_name,
        storage=args.storage,
        schema_path=args.schema,
        output_path=args.output,
        top_k=args.top_k,
    )


def cmd_retrain_best(args: argparse.Namespace) -> None:
    overrides = _parse_overrides(args.override or [])
    cfg = load_project_config(extra_files=args.config or [], overrides=overrides)
    hpo_cfg = cfg["hpo"]

    default_stage_b_name = f"{hpo_cfg['study_base_name']}_{hpo_cfg['stage_b'].get('name_suffix', 'stage_b')}"
    study_name = args.study_name or default_stage_b_name
    storage = args.storage or hpo_cfg["storage"]

    study = optuna.load_study(study_name=study_name, storage=storage)
    if study.best_trial is None:
        raise RuntimeError("No best trial available for retraining.")

    best_trial = study.best_trial
    best_cfg = best_trial.user_attrs.get("config")
    if not best_cfg:
        raise RuntimeError("Best trial is missing stored configuration.")

    base_seed = int(best_cfg.get("seed", cfg.get("seed", 42)))
    seeds = max(1, args.seeds)
    seed_values = [base_seed + i * 997 for i in range(seeds)]

    objectives: list[float] = []
    run_summaries: list[dict[str, Any]] = []
    best_checkpoint = ""
    best_objective = float("-inf")

    for seed in seed_values:
        cfg_seed = deepcopy(best_cfg)
        cfg_seed["seed"] = seed
        checkpoint_dir = Path(cfg_seed["checkpoint"]["dir"]) / "retrain" / f"seed_{seed}"
        cfg_seed["checkpoint"]["dir"] = str(checkpoint_dir)
        result = run_training_job(cfg_seed, trial=None, resume=False)
        objectives.append(result["objective"])
        run_summaries.append(
            {
                "seed": seed,
                "objective": result["objective"],
                "checkpoint": result.get("checkpoint_path", ""),
            }
        )
        if result["objective"] > best_objective:
            best_objective = result["objective"]
            best_checkpoint = result.get("checkpoint_path", "")

    mean_obj = statistics.mean(objectives)
    std_obj = statistics.pstdev(objectives) if len(objectives) > 1 else 0.0

    artifact_dir = Path("artifacts") / "hpo" / "retrain"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "study_name": study_name,
        "storage": storage,
        "seed_values": seed_values,
        "objectives": objectives,
        "mean_objective": mean_obj,
        "std_objective": std_obj,
        "runs": run_summaries,
        "best_checkpoint": best_checkpoint,
    }
    summary_path = artifact_dir / "retrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DataAug-DeBERTa Evidence training & HPO CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a single training job")
    train_parser.add_argument(
        "--config",
        action="append",
        help="Additional YAML config files to merge after defaults",
    )
    train_parser.add_argument(
        "--override",
        action="append",
        help="Override values via KEY=VALUE (VALUE parsed as JSON when possible)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where checkpoints and artifacts will be stored",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint if available",
    )
    train_parser.set_defaults(func=cmd_train)

    tune_parser = subparsers.add_parser("tune", help="Run two-stage Optuna hyper-parameter search")
    tune_parser.add_argument(
        "--config",
        action="append",
        help="Additional YAML config files to merge after defaults",
    )
    tune_parser.add_argument(
        "--override",
        action="append",
        help="Override values via KEY=VALUE (VALUE parsed as JSON when possible)",
    )
    tune_parser.add_argument("--model", default=None, help="Override base model name")
    tune_parser.add_argument("--trials-a", type=int, default=380)
    tune_parser.add_argument("--epochs-a", type=int, default=100)
    tune_parser.add_argument("--trials-b", type=int, default=120)
    tune_parser.add_argument("--epochs-b", type=int, default=100)
    tune_parser.add_argument("--k-top", type=int, default=5)
    tune_parser.add_argument("--seed", type=int, default=42)
    tune_parser.add_argument("--timeout", type=int, default=604800)
    tune_parser.set_defaults(func=cmd_tune)

    eval_parser = subparsers.add_parser(
        "eval-trial", help="Evaluate a saved checkpoint on a split"
    )
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write evaluation_report.json",
    )
    eval_parser.add_argument(
        "--config",
        action="append",
        help="Additional YAML config files to merge after defaults",
    )
    eval_parser.add_argument(
        "--override",
        action="append",
        help="Override values via KEY=VALUE (VALUE parsed as JSON when possible)",
    )
    eval_parser.set_defaults(func=cmd_eval)

    summary_parser = subparsers.add_parser(
        "summarize-study", help="Generate a study summary JSON"
    )
    summary_parser.add_argument("--study-name", required=True)
    summary_parser.add_argument(
        "--storage", default="sqlite:///optuna.db", help="Optuna storage URI"
    )
    summary_parser.add_argument(
        "--schema",
        type=Path,
        default=Path("specs/002-storage-optimized-training/contracts/study_output_schema.json"),
        help="Path to the study summary JSON schema",
    )
    summary_parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/study_summary.json"),
        help="Where to write the summary JSON",
    )
    summary_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Include up to K trials in the top_trials section",
    )
    summary_parser.set_defaults(func=cmd_summarize)

    retrain_parser = subparsers.add_parser(
        "retrain-best", help="Retrain the best Stage B configuration across multiple seeds"
    )
    retrain_parser.add_argument(
        "--config",
        action="append",
        help="Additional YAML config files to merge after defaults",
    )
    retrain_parser.add_argument(
        "--override",
        action="append",
        help="Override values via KEY=VALUE (VALUE parsed as JSON when possible)",
    )
    retrain_parser.add_argument(
        "--study-name",
        help="Optuna study name (defaults to Stage B study)",
    )
    retrain_parser.add_argument(
        "--storage",
        help="Optuna storage URI (defaults to configured storage)",
    )
    retrain_parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of random seeds to retrain with",
    )
    retrain_parser.set_defaults(func=cmd_retrain_best)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
