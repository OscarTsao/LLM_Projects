"""Unified CLI for criteria-only training and two-stage Optuna HPO."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import click
import numpy as np
from dataaug_multi_both.hpo.hpo_driver import HpoRunConfig, StageBudget, run_two_stage_hpo
from dataaug_multi_both.hpo.trial_executor import DatasetConfig, HardwareConfig
from dataaug_multi_both.training.runner import (
    DEFAULT_PARAMS,
    TrainerSettings,
    merge_defaults,
    trainer_entrypoint,
)
from dataaug_multi_both.utils.mlflow_setup import BufferedMLflowLogger, init_mlflow

try:
    from torch import cuda  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - torch required for training but guard for docs
    cuda = None  # type: ignore[assignment]

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_best_params(experiments_dir: Path, experiment_name: str, study_name: str) -> dict[str, Any]:
    candidates = [
        experiments_dir / "artifacts" / "hpo" / study_name / "stage_B" / "best_params.json",
        experiments_dir / experiment_name / "stage_B" / "best_params.json",
        experiments_dir / experiment_name / "best_params.json",
    ]
    for path in candidates:
        if path.exists():
            payload = _load_json(path)
            if isinstance(payload, dict) and "params" in payload:
                return payload["params"]
            if isinstance(payload, dict):
                return payload
    raise click.UsageError(
        "Unable to locate best parameters. Run HPO first or provide --config-path."
    )


def _write_summary(
    results: list[tuple[int, dict[str, Any]]],
    params: dict[str, Any],
    dataset_cfg: DatasetConfig,
    hardware_cfg: HardwareConfig,
    experiment_name: str,
) -> Path:
    out_dir = Path("outputs") / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [float(r["macro_f1"]) for _, r in results]
    summary = {
        "timestamp": int(time.time()),
        "experiment_name": experiment_name,
        "params": params,
        "dataset": {
            "id": dataset_cfg.dataset_id,
            "revision": dataset_cfg.revision,
        },
        "hardware": {
            "max_length": hardware_cfg.max_length,
            "per_device_batch_size": hardware_cfg.per_device_batch_size,
            "grad_accumulation_steps": hardware_cfg.grad_accumulation_steps,
            "num_workers": hardware_cfg.num_workers,
            "pin_memory": hardware_cfg.pin_memory,
            "amp_dtype": hardware_cfg.amp_dtype,
            "gradient_checkpointing": hardware_cfg.gradient_checkpointing,
        },
        "runs": [
            {
                "seed": seed,
                "macro_f1": float(payload["macro_f1"]),
                "threshold": float(payload["threshold"]),
                "run_id": payload.get("run_id"),
                "best_epoch": payload.get("best_epoch"),
            }
            for seed, payload in results
        ],
        "aggregate": {
            "metric_mean": float(np.mean(metrics)) if metrics else 0.0,
            "metric_std": float(np.std(metrics)) if len(metrics) > 1 else 0.0,
            "best_metric": float(max(metrics)) if metrics else 0.0,
        },
    }
    out_path = out_dir / f"{experiment_name}_summary_{summary['timestamp']}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return out_path


def _run_manual_training(
    params: dict[str, Any],
    seeds: Iterable[int],
    epochs: int,
    trainer_settings: TrainerSettings,
    experiments_dir: Path,
) -> list[tuple[int, dict[str, Any]]]:
    results: list[tuple[int, dict[str, Any]]] = []
    for seed in seeds:
        run_dir = experiments_dir / "manual_runs" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = BufferedMLflowLogger(run_dir)
        payload = trainer_entrypoint(
            trial=None,
            stage="manual",
            max_epochs=epochs,
            seed=seed,
            mlflow_logger=logger,
            run_dir=run_dir,
            settings=trainer_settings,
            manual_params=params,
        )
        results.append((seed, payload))
        click.echo(
            f"Seed {seed}: macro_f1={payload['macro_f1']:.4f}, "
            f"threshold={payload['threshold']:.3f}, best_epoch={payload['best_epoch']}"
        )
    return results


def _build_hardware_cfg(
    max_seq_length: int,
    per_device_batch_size: int,
    grad_accumulation_steps: int,
    num_workers: int,
    pin_memory: bool,
    amp_dtype: str,
    gradient_checkpointing: bool,
) -> HardwareConfig:
    cfg = HardwareConfig(
        max_length=max_seq_length,
        per_device_batch_size=per_device_batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        num_workers=num_workers,
        pin_memory=pin_memory,
        amp_dtype=amp_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    cfg.validate()
    return cfg


def _optimize_hardware(cfg: HardwareConfig) -> HardwareConfig:
    """Tune hardware defaults for the local machine when caller kept baseline values."""

    cpu_count = os.cpu_count() or 4
    default_workers = 2
    if cfg.num_workers == default_workers:
        cfg.num_workers = min(8, max(2, cpu_count - 2))

    try:
        if cuda and cuda.is_available():
            props = cuda.get_device_properties(0)
            mem_gb = props.total_memory / (1024**3)

            if cfg.per_device_batch_size == 8:
                if mem_gb >= 40:
                    cfg.per_device_batch_size = 20
                elif mem_gb >= 32:
                    cfg.per_device_batch_size = 16
                elif mem_gb >= 22:
                    cfg.per_device_batch_size = 14
                elif mem_gb >= 16:
                    cfg.per_device_batch_size = 12

            if cfg.amp_dtype == "none" and mem_gb >= 10:
                cfg.amp_dtype = "bf16"

            if not cfg.pin_memory:
                cfg.pin_memory = True
    except Exception:
        pass

    return cfg


@click.command()
@click.option("--dataset-id", default="irlab-udc/redsm5", show_default=True)
@click.option("--dataset-revision", default=None)
@click.option("--cache-dir", default=None)
@click.option("--model", default="microsoft/deberta-v3-base", show_default=True)
@click.option("--epochs", default=3, type=int, show_default=True)
@click.option("--per-device-batch-size", default=8, type=int, show_default=True)
@click.option("--max-seq-length", default=512, type=int, show_default=True)
@click.option("--grad-accumulation-steps", default=1, type=int, show_default=True)
@click.option("--num-workers", default=2, type=int, show_default=True)
@click.option("--pin-memory/--no-pin-memory", default=True, show_default=True)
@click.option(
    "--amp-dtype",
    type=click.Choice(["none", "fp16", "bf16"]),
    default="none",
    show_default=True,
)
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    default=False,
    show_default=True,
)
@click.option("--experiments-dir", default="experiments", show_default=True)
@click.option("--experiment-name", default="criteria_hpo", show_default=True)
@click.option("--study-name", default="criteria_hpo", show_default=True)
@click.option("--study-db", default="experiments/criteria_hpo.db", show_default=True)
@click.option("--seed", default=1337, type=int, show_default=True)
@click.option("--seeds", multiple=True, type=int)
@click.option("--deterministic/--no-deterministic", default=False, show_default=True)
@click.option("--hpo/--no-hpo", default=False, show_default=True)
@click.option("--trials-a", default=380, type=int, show_default=True)
@click.option("--trials-b", default=120, type=int, show_default=True)
@click.option("--epochs-a", default=100, type=int, show_default=True)
@click.option("--epochs-b", default=100, type=int, show_default=True)
@click.option("--timeout-a", default=432_000, type=int, show_default=True)
@click.option("--timeout-b", default=172_800, type=int, show_default=True)
@click.option("--total-timeout", default=604_800, type=int, show_default=True)
@click.option("--top-k", default=8, type=int, show_default=True)
@click.option("--use-best", is_flag=True, default=False)
@click.option("--config-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--retrain-best", is_flag=True, default=False)
@click.option("--retrain-seeds", default=3, type=int, show_default=True)
def main(
    dataset_id: str,
    dataset_revision: str | None,
    cache_dir: str | None,
    model: str,
    epochs: int,
    per_device_batch_size: int,
    max_seq_length: int,
    grad_accumulation_steps: int,
    num_workers: int,
    pin_memory: bool,
    amp_dtype: str,
    gradient_checkpointing: bool,
    experiments_dir: str,
    experiment_name: str,
    study_name: str,
    study_db: str,
    seed: int,
    seeds: Iterable[int],
    deterministic: bool,
    hpo: bool,
    trials_a: int,
    trials_b: int,
    epochs_a: int,
    epochs_b: int,
    timeout_a: int,
    timeout_b: int,
    total_timeout: int,
    top_k: int,
    use_best: bool,
    config_path: str | None,
    retrain_best: bool,
    retrain_seeds: int,
) -> None:
    experiments_path = Path(experiments_dir)
    experiments_path.mkdir(parents=True, exist_ok=True)

    dataset_cfg = DatasetConfig(
        dataset_id=dataset_id,
        revision=dataset_revision,
        cache_dir=cache_dir,
    )
    hardware_cfg = _build_hardware_cfg(
        max_seq_length=max_seq_length,
        per_device_batch_size=per_device_batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        num_workers=num_workers,
        pin_memory=pin_memory,
        amp_dtype=amp_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    hardware_cfg = _optimize_hardware(hardware_cfg)
    LOGGER.info(
        "Hardware configuration -> batch_size=%s grad_accum=%s workers=%s amp=%s checkpointing=%s",
        hardware_cfg.per_device_batch_size,
        hardware_cfg.grad_accumulation_steps,
        hardware_cfg.num_workers,
        hardware_cfg.amp_dtype,
        hardware_cfg.gradient_checkpointing,
    )
    if max_seq_length > 512:
        click.echo("max_seq_length > 512 detected; clamped to 512 to ensure compatibility.", err=True)

    conflicts = sum(
        [
            int(bool(hpo)),
            int(bool(retrain_best)),
            int(bool(use_best)),
            int(config_path is not None),
        ]
    )
    if conflicts > 1:
        raise click.UsageError("Use only one of --hpo, --retrain-best, --use-best, or --config-path.")

    init_mlflow(experiments_dir, experiment_name)

    trainer_settings = TrainerSettings(
        dataset_cfg=dataset_cfg,
        hardware_cfg=hardware_cfg,
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        model_name=model,
        deterministic=deterministic,
    )

    if hpo:
        storage_url = f"sqlite:///{Path(study_db).resolve()}"
        config = HpoRunConfig(
            experiments_dir=experiments_path,
            experiment_name=experiment_name,
            study_name=study_name,
            storage_url=storage_url,
            dataset=dataset_cfg,
            hardware=hardware_cfg,
            deterministic=deterministic,
            base_seed=seed,
            stage_a=StageBudget(trials=trials_a, epochs=epochs_a, timeout_seconds=timeout_a),
            stage_b=StageBudget(trials=trials_b, epochs=epochs_b, timeout_seconds=timeout_b),
            total_timeout=total_timeout,
            top_k=top_k,
            model_name=model,
        )
        run_two_stage_hpo(config)
        return

    seed_list = list(seeds) if seeds else [seed]

    if retrain_best:
        params = _resolve_best_params(experiments_path, experiment_name, study_name)
        params = merge_defaults(params)
        target_seeds = [seed + idx for idx in range(retrain_seeds)]
    elif use_best:
        params = merge_defaults(_resolve_best_params(experiments_path, experiment_name, study_name))
        target_seeds = seed_list
    elif config_path is not None:
        params = merge_defaults(_load_json(Path(config_path)))
        target_seeds = seed_list
    else:
        params = DEFAULT_PARAMS.copy()
        target_seeds = seed_list

    click.echo(f"Running manual training for seeds {target_seeds}...")
    results = _run_manual_training(params, target_seeds, epochs, trainer_settings, experiments_path)
    summary_path = _write_summary(results, params, dataset_cfg, hardware_cfg, experiment_name)

    metrics = [float(payload["macro_f1"]) for _, payload in results]
    mean = float(np.mean(metrics)) if metrics else 0.0
    std = float(np.std(metrics)) if len(metrics) > 1 else 0.0
    click.echo(f"Manual training complete. mean={mean:.4f} std={std:.4f}. Summary -> {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()
