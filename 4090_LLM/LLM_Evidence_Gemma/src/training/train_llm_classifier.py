"""
Training entrypoint for LLM + classification head with causal or encoderized modes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from datasets import DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer import Trainer
from transformers.trainer_utils import set_seed

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled via graceful degradation
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_AVAILABLE = False

from src.data.classification_dataset import (
    compute_class_weights,
    load_classification_dataset,
)
from src.data.collation import SmartBatchCollator
from src.data.tokenization import apply_pair_format, build_tokenizer
from src.models.llm_classification import LLMClassificationModel
from src.utils.logger import setup_logger
from src.utils.lora import infer_lora_target_modules

logger = setup_logger("train_llm_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM classification trainer")
    data = parser.add_argument_group("data")
    data.add_argument("--dataset_name", type=str, default=None)
    data.add_argument("--dataset_config", type=str, default=None)
    data.add_argument("--train_file", type=str, default=None)
    data.add_argument("--validation_file", type=str, default=None)
    data.add_argument("--text_column", type=str, default="text")
    data.add_argument("--second_text_column", type=str, default=None)
    data.add_argument("--label_column", type=str, default="label")

    model = parser.add_argument_group("model")
    model.add_argument("--model_name", type=str, required=True)
    model.add_argument("--mode", type=str, choices=["causal", "encoderized"], default="causal")
    model.add_argument("--pooler", type=str, choices=["last", "mean", "attn"], default="mean")
    model.add_argument("--flash_attn2", type=str, choices=["true", "false"], default="true")
    model.add_argument("--max_length", type=int, default=1024)
    model.add_argument("--pad_to_multiple_of", type=int, default=8)
    model.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    model.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    model.set_defaults(gradient_checkpointing=True)

    qlora = parser.add_argument_group("qlora")
    qlora.add_argument("--lora_r", type=int, default=64)
    qlora.add_argument("--lora_alpha", type=int, default=16)
    qlora.add_argument("--lora_dropout", type=float, default=0.05)
    qlora.add_argument("--warmup_head_steps", type=int, default=50)
    qlora.add_argument("--cpu_only", action="store_true", help="Disable QLoRA / quantization and run in fp32 (useful for tests).")
    qlora.add_argument("--disable_lora", action="store_true", help="Skip LoRA adapters and train the full backbone/head.")

    train = parser.add_argument_group("training")
    train.add_argument("--output_dir", type=str, default="outputs/llm_classifier")
    train.add_argument("--num_train_epochs", type=float, default=3.0)
    train.add_argument("--per_device_train_batch_size", type=int, default=4)
    train.add_argument("--per_device_eval_batch_size", type=int, default=4)
    train.add_argument("--gradient_accumulation_steps", type=int, default=16)
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--warmup_ratio", type=float, default=0.03)
    train.add_argument("--logging_steps", type=int, default=10)
    train.add_argument("--save_strategy", type=str, default="epoch")
    train.add_argument("--eval_strategy", type=str, default="epoch")
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--group_by_length", action="store_true", default=True)

    loss = parser.add_argument_group("loss")
    loss.add_argument("--label_smoothing", type=float, default=0.0)
    loss.add_argument("--use_auto_class_weights", action="store_true")
    loss.add_argument("--class_weights", type=str, default=None, help="Comma-separated floats.")

    return parser.parse_args()


def _preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> DatasetDict:
    def preprocess(batch: Dict[str, List]):
        tokenized = apply_pair_format(
            batch,
            tokenizer,
            text_column=args.text_column,
            second_text_column=args.second_text_column,
            max_length=args.max_length,
            pad_to_multiple_of=None,
        )
        tokenized["labels"] = batch["labels"]
        tokenized["seq_len"] = [len(ids) for ids in tokenized["input_ids"]]
        return tokenized

    return dataset.map(preprocess, batched=True)


class ClassificationTrainer(Trainer):
    def __init__(
        self,
        *args,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing if self.label_smoothing > 0 else 0.0,
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class HeadWarmupCallback(TrainerCallback):
    def __init__(self, warmup_steps: int, lora_param_names: List[str]):
        self.warmup_steps = warmup_steps
        self.lora_param_names = set(lora_param_names)
        self.unfroze = warmup_steps == 0

    def on_train_begin(self, args, state, control, **kwargs):
        if self.warmup_steps == 0:
            self.unfroze = True
            return
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if name in self.lora_param_names:
                param.requires_grad = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.unfroze or state.global_step < self.warmup_steps:
            return
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if name in self.lora_param_names:
                param.requires_grad = True
        logger.info("LoRA adapters unfrozen after %d warmup steps.", self.warmup_steps)
        self.unfroze = True


class MonitoringCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.metrics_path = Path(output_dir) / "training_metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("Train log: %s", record)


def build_lora_model(args: argparse.Namespace, num_labels: int) -> LLMClassificationModel:
    use_qlora = (not args.cpu_only) and torch.cuda.is_available()
    if use_qlora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for QLoRA path. Install with `pip install peft bitsandbytes` "
                "or rerun with --cpu_only/--disable_lora."
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if args.flash_attn2 == "true":
            try:
                backbone.config._attn_implementation = "flash_attention_2"
            except Exception:
                logger.warning("FlashAttention-2 requested but not supported for this model.")

        backbone = prepare_model_for_kbit_training(
            backbone, use_gradient_checkpointing=args.gradient_checkpointing
        )
    else:
        logger.info("Running in CPU-only / fp32 mode (QLoRA disabled).")
        backbone = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        if args.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()

    model = LLMClassificationModel(
        backbone=backbone,
        num_labels=num_labels,
        mode=args.mode,
        pooler=args.pooler,
    )

    apply_lora = (not args.disable_lora) and PEFT_AVAILABLE
    if apply_lora:
        target_modules = infer_lora_target_modules(model.backbone)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=target_modules,
        )

        model.backbone = get_peft_model(model.backbone, lora_config)
        model.backbone.print_trainable_parameters()
    else:
        logger.warning(
            "LoRA adapters disabled%s – training full backbone.",
            " because peft is not installed" if not PEFT_AVAILABLE else "",
        )
    return model


def compute_metrics_fn(eval_preds):
    if hasattr(eval_preds, "predictions"):
        logits = eval_preds.predictions
        labels = eval_preds.label_ids
    else:
        logits, labels = eval_preds
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.model_name, right_pad=True)
    dataset, label_list = load_classification_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        train_file=args.train_file,
        validation_file=args.validation_file,
        text_column=args.text_column,
        label_column=args.label_column,
        second_text_column=args.second_text_column,
    )

    dataset = _preprocess_dataset(dataset, tokenizer, args)

    smart_collator = SmartBatchCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    num_labels = len(label_list)
    model = build_lora_model(args, num_labels)

    class_weights = None
    if args.class_weights:
        weights = [float(w) for w in args.class_weights.split(",")]
        if len(weights) != num_labels:
            raise ValueError("class_weights size mismatch.")
        class_weights = torch.tensor(weights, dtype=torch.float32)
    elif args.use_auto_class_weights:
        weights = compute_class_weights(dataset["train"]["labels"], num_labels)
        class_weights = torch.tensor(weights, dtype=torch.float32)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_safetensors=False,
        bf16=args.bf16 or torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length,
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
        report_to="none",
        length_column_name="seq_len",
    )

    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["train"].select(range(min(256, len(dataset["train"]))))

    trainer = ClassificationTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=smart_collator,
        compute_metrics=compute_metrics_fn,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
    )

    lora_param_names = [
        name for name, param in model.named_parameters() if "lora_" in name
    ]

    trainer.add_callback(HeadWarmupCallback(args.warmup_head_steps, lora_param_names))
    trainer.add_callback(MonitoringCallback(args.output_dir))
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_metrics("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
