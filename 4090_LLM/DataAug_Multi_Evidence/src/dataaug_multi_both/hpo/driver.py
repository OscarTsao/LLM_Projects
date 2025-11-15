"""Two-stage HPO driver utilities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Callable

import optuna

from dataaug_multi_both.mlflow_init import init_mlflow
from dataaug_multi_both.hpo.artifacts import export_stage_artifacts
from .objective import ObjectiveConfig, build_objective
from .run_study import StageResult, StageSettings, run_stage, select_top_trials, split_budget
from .space import narrow_stage_b_space, stage_a_search_space
from rich.console import Console
from rich.table import Table

console = Console()


def _make_progress_callback(stage_label: str, total_trials: int) -> Callable[[optuna.Study, optuna.trial.FrozenTrial], None]:
    total = total_trials if total_trials > 0 else None
    best_value: float | None = None
    start_time = time.time()

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        nonlocal best_value
        value = trial.value
        improved = value is not None and (best_value is None or value > best_value + 1e-12)
        if improved and value is not None:
            best_value = value

        completed = sum(
            1
            for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        total_label = f"{completed}/{total}" if total else str(completed)
        elapsed = time.time() - start_time

        table = Table(show_header=True, expand=False, box=None)
        table.add_column("Stage", style="cyan")
        table.add_column("Trials", style="magenta")
        table.add_column("Value", style="green")
        table.add_column("Best", style="yellow")
        table.add_column("Elapsed", style="white")
        table.add_column("State", style="blue")

        value_str = "-" if value is None else f"{value:.4f}"
        best_str = "-" if best_value is None else f"{best_value:.4f}"
        state = trial.state.name.lower()
        if improved:
            state = f"{state} ↑"
        elif trial.state == optuna.trial.TrialState.PRUNED:
            state = f"{state} (pruned)"

        table.add_row(
            stage_label,
            total_label,
            value_str,
            best_str,
            f"{elapsed:,.1f}s",
            state,
        )
        console.print(table)

    return _callback


DEFAULT_MODEL = "microsoft/deberta-v3-base"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage HPO driver (scaffolding phase)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base encoder model name")
    parser.add_argument("--trials-a", type=int, default=380, help="Number of Stage A trials")
    parser.add_argument("--epochs-a", type=int, default=100, help="Epochs per Stage A trial")
    parser.add_argument("--trials-b", type=int, default=120, help="Number of Stage B trials")
    parser.add_argument("--epochs-b", type=int, default=100, help="Epochs per Stage B trial")
    parser.add_argument("--k-top", type=int, default=5, help="Top-K configs to promote")
    parser.add_argument(
        "--timeout",
        type=int,
        default=604800,
        help="Global timeout (seconds) for both stages",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--objective", default="val_f1", help="Objective metric alias")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the resolved configuration without running Optuna yet",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run both stages with a deterministic synthetic objective",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/data/dataset.yaml",
        help="Path to dataset YAML (used when --use-real-data)",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/hpo_runs",
        help="Directory for stage outputs",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Load dataset from --dataset-config instead of synthetic data",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable all data augmentations (fix num_augmentations=0)",
    )
    parser.add_argument(
        "--synthetic-train-size",
        type=int,
        default=128,
        help="Training examples when using synthetic data",
    )
    parser.add_argument(
        "--synthetic-val-size",
        type=int,
        default=64,
        help="Validation examples when using synthetic data",
    )
    parser.add_argument(
        "--synthetic-seq-len",
        type=int,
        default=128,
        help="Sequence length when using synthetic data",
    )
    return parser


def _summarise_space(space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    categorical = sum(1 for cfg in space.values() if cfg.get("type") == "categorical")
    numeric = sum(
        1 for cfg in space.values() if cfg.get("type") in {"float", "int", "loguniform"}
    )
    return {
        "total": len(space),
        "categorical": categorical,
        "numeric": numeric,
    }


def _make_stage_a_settings(
    args: argparse.Namespace,
    space: Dict[str, Dict[str, Any]],
    progress_callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None = None,
) -> StageSettings:
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=60,
        n_ei_candidates=128,
        seed=args.seed,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=max(1, args.epochs_a),
        reduction_factor=3,
    )
    timeout_a = max(1, int(args.timeout * 0.6))
    frozen_params = None
    if getattr(args, "no_augment", False):
        # Force-disable augmentations by fixing num_augmentations to 0
        frozen_params = {"num_augmentations": 0}
    return StageSettings(
        stage_name="stage_a",
        search_space=space,
        sampler=sampler,
        pruner=pruner,
        n_trials=max(1, args.trials_a),
        timeout=timeout_a,
        plateau_patience=120,
        epochs=args.epochs_a,
        frozen_params=frozen_params,
        progress_callback=progress_callback,
    )


def _make_stage_b_settings(
    args: argparse.Namespace,
    base_seed: int,
    narrowed_space: Dict[str, Dict[str, Any]],
    frozen: Dict[str, Any],
    stage_index: int,
    n_trials: int,
    timeout: int,
    progress_callback: Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None = None,
) -> StageSettings:
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=min(30, max(1, n_trials // 2)),
        n_ei_candidates=128,
        seed=123 + base_seed + stage_index,
    )
    pruner = optuna.pruners.PercentilePruner(25.0, n_startup_trials=10, n_warmup_steps=2)
    return StageSettings(
        stage_name=f"stage_b_top{stage_index}",
        search_space=narrowed_space,
        sampler=sampler,
        pruner=pruner,
        n_trials=max(0, n_trials),
        timeout=max(0, timeout),
        plateau_patience=120,
        epochs=args.epochs_b,
        frozen_params=frozen,
        progress_callback=progress_callback,
    )


def _hash_to_unit(value: str) -> float:
    return (hash(value) & 0xFFFFFF) / float(0xFFFFFF)


def _make_simulation_objective(stage_bias: float, stage_name: str, seed: int):
    def _objective(trial: optuna.Trial, params: Dict[str, Any], settings: StageSettings) -> float:
        total = 0.0
        for idx, (key, value) in enumerate(sorted(params.items())):
            token = f"{stage_name}:{seed}:{idx}:{key}:{value}"
            total += _hash_to_unit(token)
        if params:
            total /= len(params)
        total += stage_bias
        total += 0.01 * trial.number
        return float(total)

    return _objective


def _run_simulation(args: argparse.Namespace) -> int:
    stage_a_space = stage_a_search_space(default_model=args.model)
    stage_a_progress = _make_progress_callback("Stage A", args.trials_a)
    stage_a_settings = _make_stage_a_settings(args, stage_a_space, progress_callback=stage_a_progress)
    objective_a = _make_simulation_objective(stage_bias=0.0, stage_name="stage_a", seed=args.seed)
    stage_a_result = run_stage(stage_a_settings, objective_a)

    top_candidates = select_top_trials(stage_a_result.study.trials, args.k_top)
    if not top_candidates:
        print("No promotable trials produced in Stage A simulation.")
        return 2

    # Limit promotions to available trial budget.
    max_promotions = min(len(top_candidates), max(1, min(args.k_top, args.trials_b)))
    promoted = top_candidates[:max_promotions]

    remaining_trials = max(0, args.trials_b)
    timeout_b_total = max(1, args.timeout - stage_a_settings.timeout)
    trial_shares = split_budget(remaining_trials, len(promoted))
    timeout_shares = split_budget(timeout_b_total, len(promoted))

    objective_b = _make_simulation_objective(stage_bias=0.2, stage_name="stage_b", seed=args.seed + 1)
    stage_b_results: List[StageResult] = []

    for idx, trial in enumerate(promoted):
        trials_for_candidate = trial_shares[idx] if idx < len(trial_shares) else 0
        timeout_for_candidate = timeout_shares[idx] if idx < len(timeout_shares) else 0
        if trials_for_candidate == 0 or timeout_for_candidate == 0:
            continue

        narrowed = narrow_stage_b_space(trial.params)
        stage_label = f"stage_b_top{idx}"
        progress_cb = _make_progress_callback(stage_label, trials_for_candidate)
        settings_b = _make_stage_b_settings(
            args=args,
            base_seed=args.seed,
            narrowed_space=narrowed.search,
            frozen=narrowed.frozen,
            stage_index=idx,
            n_trials=trials_for_candidate,
            timeout=timeout_for_candidate,
            progress_callback=progress_cb,
        )
        result_b = run_stage(settings_b, objective_b)
        stage_b_results.append(result_b)

    report = {
        "stage_a": {
            "completed_trials": len(stage_a_result.completed_trials),
            "best_value": stage_a_result.best_trial.value,
        },
        "stage_b": [
            {
                "stage_name": res.settings.stage_name,
                "completed_trials": len(res.completed_trials),
                "best_value": res.best_trial.value,
            }
            for res in stage_b_results
        ],
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stage_a_space = stage_a_search_space(default_model=args.model)
    summary = _summarise_space(stage_a_space)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage_a": summary,
                    "message": "Dry run only. Stage execution will be added in a subsequent phase.",
                },
                indent=2,
            )
        )
        return 0

    if args.simulate:
        return _run_simulation(args)

    return _run_training(args)


def _run_training(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mlflow_buffer = init_mlflow(
        tracking_uri=f"sqlite:///{Path(args.output_root) / 'mlflow.db'}",
        experiment_name="deberta_v3_evidence",
        buffer_dir=output_root / "artifacts/mlflow_buffer",
    )

    stage_a_space = stage_a_search_space(default_model=args.model)
    stage_a_settings = _make_stage_a_settings(args, stage_a_space)

    objective_cfg = ObjectiveConfig(
        output_root=output_root,
        default_model=args.model,
        dataset_config=Path(args.dataset_config) if args.dataset_config else None,
        objective_metric=args.objective,
        seed=args.seed,
        use_synthetic=not args.use_real_data,
        synthetic_train_size=max(4, args.synthetic_train_size),
        synthetic_val_size=max(2, args.synthetic_val_size),
        synthetic_seq_len=max(8, args.synthetic_seq_len),
        mlflow_buffer=mlflow_buffer,
    )

    start_time = time.time()
    objective_fn = build_objective(objective_cfg)
    stage_a_start = time.time()
    stage_a_result = run_stage(stage_a_settings, objective_fn)
    stage_a_duration = time.time() - stage_a_start
    export_stage_artifacts(
        stage_a_result,
        output_root / "artifacts/hpo/stage_a",
        mlflow_buffer,
    )
    print(
        json.dumps(
            {
                "stage": "A",
                "completed_trials": len(stage_a_result.completed_trials),
                "best_value": stage_a_result.best_trial.value,
                "duration_seconds": stage_a_duration,
            }
        )
    )

    top_candidates = select_top_trials(stage_a_result.study.trials, args.k_top)
    if not top_candidates:
        print("Stage A produced no promotable trials.")
        return 2

    max_promotions = min(len(top_candidates), max(1, min(args.k_top, args.trials_b)))
    promoted = top_candidates[:max_promotions]

    timeout_a = stage_a_settings.timeout or int(args.timeout * 0.6)
    remaining_trials = max(0, args.trials_b)
    timeout_b_total = max(1, args.timeout - timeout_a)
    trial_shares = split_budget(remaining_trials, len(promoted))
    timeout_shares = split_budget(timeout_b_total, len(promoted))

    stage_b_results: List[StageResult] = []

    for idx, trial in enumerate(promoted):
        trials_for_candidate = trial_shares[idx] if idx < len(trial_shares) else 0
        timeout_for_candidate = timeout_shares[idx] if idx < len(timeout_shares) else 0
        if trials_for_candidate == 0 or timeout_for_candidate == 0:
            continue

        narrowed = narrow_stage_b_space(trial.params)
        settings_b = _make_stage_b_settings(
            args=args,
            base_seed=args.seed,
            narrowed_space=narrowed.search,
            frozen=narrowed.frozen,
            stage_index=idx,
            n_trials=trials_for_candidate,
            timeout=timeout_for_candidate,
        )
        stage_b_start = time.time()
        result_b = run_stage(settings_b, objective_fn)
        stage_b_duration = time.time() - stage_b_start
        export_stage_artifacts(
            result_b,
            output_root / f"artifacts/hpo/{settings_b.stage_name}",
            mlflow_buffer,
        )
        stage_b_results.append(result_b)
        print(
            json.dumps(
                {
                    "stage": settings_b.stage_name,
                    "completed_trials": len(result_b.completed_trials),
                    "best_value": result_b.best_trial.value,
                    "duration_seconds": stage_b_duration,
                }
            )
        )

    best_trial = stage_a_result.best_trial
    for res in stage_b_results:
        if res.best_trial.value and best_trial.value is not None:
            if res.best_trial.value > best_trial.value:
                best_trial = res.best_trial

    report = {
        "stage_a": {
            "completed_trials": len(stage_a_result.completed_trials),
            "best_value": stage_a_result.best_trial.value,
        },
        "stage_b": [
            {
                "stage_name": res.settings.stage_name,
                "completed_trials": len(res.completed_trials),
                "best_value": res.best_trial.value,
            }
            for res in stage_b_results
        ],
        "best_overall": {
            "value": best_trial.value,
            "number": best_trial.number,
        },
        "duration_seconds": time.time() - start_time,
    }

    print(json.dumps(report, indent=2))

    aggregate_dir = output_root / "artifacts/hpo/aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    with (aggregate_dir / "best_trial.json").open("w", encoding="utf-8") as fh:
        json.dump({"number": best_trial.number, "value": best_trial.value, "params": best_trial.params}, fh, indent=2)
    if mlflow_buffer is not None:
        mlflow_buffer.log_artifact(aggregate_dir / "best_trial.json")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
