from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

import mlflow
import hydra
from hydra import utils as hydra_utils
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from .config_utils import resolve_config_paths
from .data import create_tokenized_datasets, load_datasets
from .evaluation import compute_metrics, postprocess_qa_predictions
from .modeling import SpanClassificationModel

LOGGER = logging.getLogger(__name__)


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, str]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple, set)):
                v = ",".join(map(str, v))
            elif v is None:
                v = "none"
            items.append((new_key, v))
    return dict(items)


def _setup_mlflow(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logging.tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.logging.experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            cfg.logging.experiment_name,
            artifact_location=cfg.logging.artifact_location,
        )
    mlflow.set_experiment(cfg.logging.experiment_name)
    tags = OmegaConf.to_container(cfg.logging.tags, resolve=True) or {}
    for key, value in tags.items():
        mlflow.set_tag(key, value)


def _ensure_dirs(cfg: DictConfig) -> None:
    Path(cfg.training.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.artifact_location).mkdir(parents=True, exist_ok=True)


def _log_model_signature(tokenizer, model, output_dir: Path) -> None:
    signature = {
        "tokenizer": tokenizer.__class__.__name__,
        "model": model.__class__.__name__,
    }
    with (output_dir / "model_signature.json").open("w", encoding="utf-8") as fp:
        json.dump(signature, fp, indent=2)


def run_training(cfg: DictConfig) -> Dict[str, float]:
    set_seed(cfg.seed)

    raw_datasets = load_datasets(cfg.data, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name, use_fast=cfg.model.use_fast_tokenizer
    )

    tokenized_datasets, raw_eval_examples = create_tokenized_datasets(
        raw_datasets, tokenizer, cfg.data
    )

    if cfg.training.max_train_samples:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(min(len(tokenized_datasets["train"]), cfg.training.max_train_samples))
        )
    if cfg.training.max_eval_samples:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(min(len(tokenized_datasets["validation"]), cfg.training.max_eval_samples))
        )
        raw_eval_examples["validation"] = raw_eval_examples["validation"].select(
            range(min(len(raw_eval_examples["validation"]), cfg.training.max_eval_samples))
        )

    model = SpanClassificationModel.from_pretrained(
        cfg.model.pretrained_model_name, head_cfg=cfg.model.head
    )
    if cfg.model.freeze_base_model:
        model.freeze_base_model()

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=cfg.training.eval_strategy,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        num_train_epochs=cfg.training.num_train_epochs,
        warmup_ratio=cfg.training.warmup_ratio,
        logging_steps=cfg.training.logging_steps,
        eval_steps=cfg.training.eval_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        fp16=cfg.training.fp16,
        max_grad_norm=cfg.training.max_grad_norm,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        seed=cfg.training.seed,
        tpu_num_cores=cfg.training.tpu_num_cores,
        remove_unused_columns=False,
    )

    eval_examples = raw_eval_examples["validation"]

    def compute_metrics_fn(eval_prediction):
        predictions = postprocess_qa_predictions(
            eval_examples,
            tokenized_datasets["validation"],
            eval_prediction.predictions,
            tokenizer=tokenizer,
            max_answer_length=cfg.data.max_answer_length,
        )
        metrics = compute_metrics(predictions, eval_examples)
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_fn,
    )

    if cfg.training.early_stopping_patience is not None:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
        )

    _ensure_dirs(cfg)
    _setup_mlflow(cfg)

    tags = OmegaConf.to_container(cfg.logging.tags, resolve=True) or {}

    run_name = cfg.logging.run_name_template.format(
        data=cfg.data.name,
        model=cfg.model.pretrained_model_name,
    )
    trial_tag = tags.get("trial_number")
    if trial_tag not in (None, "", "none"):
        run_name = f"{run_name}-trial-{trial_tag}"

    with mlflow.start_run(run_name=run_name):
        flattened_cfg = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        mlflow.log_params(flattened_cfg)

        trainer.train()
        trainer.save_model(cfg.training.output_dir)
        tokenizer.save_pretrained(cfg.training.output_dir)
        _log_model_signature(tokenizer, model, Path(cfg.training.output_dir))

        metrics = trainer.evaluate()
        mlflow.log_metrics(metrics)

        if training_args.save_strategy != "no":
            mlflow.log_artifacts(cfg.training.output_dir)

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    base_dir = Path(hydra_utils.get_original_cwd())
    cfg.paths.project_root = str(base_dir)
    resolved_cfg = resolve_config_paths(cfg, base_dir=base_dir)
    run_training(resolved_cfg)


if __name__ == "__main__":
    main()
