"""Hydra entrypoint for preference-based DPO finetuning."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from ..config_schemas import DPOConfig, parse_dpo_config
from ..seed import seed_everything

LOGGER = logging.getLogger(__name__)


def _prepare_dataset(cfg: DPOConfig):
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'datasets' package is required for DPO training.") from exc

    data_files = cfg.data.path_or_name
    if cfg.data.format == "jsonl":
        dataset_dict = load_dataset("json", data_files={"train": data_files})
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported data format: {cfg.data.format}")
    dataset = dataset_dict["train"]
    if cfg.data.max_samples:
        dataset = dataset.select(range(min(len(dataset), cfg.data.max_samples)))
    if cfg.data.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=cfg.data.shuffle_seed)
    required_columns = {"prompt", "chosen", "rejected"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise RuntimeError(f"DPO dataset missing columns: {missing}")
    return dataset


def _load_models(cfg: DPOConfig):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for DPO training.") from exc

    model_kwargs = {
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    tokenizer_name = cfg.model.tokenizer_name or cfg.model.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    torch_dtype = None
    if cfg.hardware.bf16:
        torch_dtype = torch.bfloat16
    elif cfg.hardware.fp16:
        torch_dtype = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if cfg.hardware.device == "auto" else cfg.hardware.device,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    reference_model_path = cfg.model.reference_model_path or cfg.model.base_model_path
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if cfg.hardware.device == "auto" else cfg.hardware.device,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.hardware.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        reference_model.gradient_checkpointing_enable()
    return base_model, reference_model, tokenizer


def _apply_lora(model, cfg: DPOConfig):
    if not cfg.hardware.lora.use_lora:
        return model
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft must be installed for LoRA finetuning.") from exc

    lora_cfg = PeftLoraConfig(
        r=cfg.hardware.lora.r,
        lora_alpha=cfg.hardware.lora.alpha,
        target_modules=cfg.hardware.lora.target_modules,
        lora_dropout=cfg.hardware.lora.dropout,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)


@hydra.main(config_path="../../../conf/train", config_name="llm_dpo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dpo_cfg: DPOConfig = parse_dpo_config(cfg)
    seed_everything(dpo_cfg.seed)
    output_dir = Path(dpo_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _prepare_dataset(dpo_cfg)
    base_model, reference_model, tokenizer = _load_models(dpo_cfg)
    base_model = _apply_lora(base_model, dpo_cfg)

    try:
        from transformers import TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("trl and transformers are required for DPO training.") from exc

    training_args = TrainingArguments(
        per_device_train_batch_size=dpo_cfg.hardware.batch_size_per_device,
        gradient_accumulation_steps=dpo_cfg.hardware.gradient_accumulation_steps,
        learning_rate=dpo_cfg.train.learning_rate,
        warmup_ratio=dpo_cfg.train.warmup_ratio,
        max_steps=dpo_cfg.train.max_steps,
        num_train_epochs=dpo_cfg.train.num_train_epochs,
        logging_steps=dpo_cfg.train.logging_steps,
        save_steps=dpo_cfg.train.save_steps,
        output_dir=output_dir.as_posix(),
        report_to=[],
        bf16=dpo_cfg.hardware.bf16,
        fp16=dpo_cfg.hardware.fp16,
    )

    trainer = DPOTrainer(
        model=base_model,
        ref_model=reference_model,
        tokenizer=tokenizer,
        args=training_args,
        beta=dpo_cfg.train.beta,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir.as_posix())
    tokenizer.save_pretrained(output_dir.as_posix())

    try:
        import mlflow
    except ImportError:  # pragma: no cover - optional
        LOGGER.warning("mlflow not available; skipping logging.")
        return

    mlflow.set_tracking_uri(dpo_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(dpo_cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name=dpo_cfg.mlflow.run_name):
        mlflow.log_params(
            {
                "learning_rate": dpo_cfg.train.learning_rate,
                "beta": dpo_cfg.train.beta,
                "max_steps": dpo_cfg.train.max_steps,
                "base_model_path": dpo_cfg.model.base_model_path,
                "reference_model_path": dpo_cfg.model.reference_model_path or "",
                "use_lora": dpo_cfg.hardware.lora.use_lora,
                "batch_size_per_device": dpo_cfg.hardware.batch_size_per_device,
                "gradient_accumulation_steps": dpo_cfg.hardware.gradient_accumulation_steps,
            }
        )
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        mlflow.log_artifacts(output_dir.as_posix(), artifact_path="dpo_model")


if __name__ == "__main__":
    main()
