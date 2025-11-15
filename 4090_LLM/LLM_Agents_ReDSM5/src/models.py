from __future__ import annotations

from typing import Any, Dict

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _infer_dtype(cfg: Dict[str, Any]) -> torch.dtype:
    if cfg.get("bf16"):
        return torch.bfloat16
    if cfg.get("fp16"):
        return torch.float16
    return torch.float32


def _freeze_layers(model: PreTrainedModel, num_layers: int) -> None:
    if num_layers <= 0:
        return
    base = getattr(model, "model", None)
    if base is None and hasattr(model, "base_model"):
        base = getattr(model.base_model, "model", None)
    if base is None or not hasattr(base, "layers"):
        return
    layers = getattr(base, "layers")
    for layer in layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def _apply_lora(model: PreTrainedModel, cfg: Dict[str, Any]) -> PreTrainedModel:
    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=cfg.get("lora_target_modules", TARGET_MODULES),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def build_model(
    cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    num_labels: int,
    checkpoint_path: str | None = None,
) -> PreTrainedModel:
    model_id = cfg["model_id"]
    method = cfg.get("method", "qlora")
    torch_dtype = _infer_dtype(cfg)

    load_kwargs: Dict[str, Any] = {
        "num_labels": num_labels,
        "problem_type": "multi_label_classification",
        "torch_dtype": torch_dtype,
    }

    if method == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("fourbit", "nf4")),
            bnb_4bit_use_double_quant=bool(cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=getattr(torch, str(cfg.get("compute_dtype", "bf16"))),
        )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs.setdefault("device_map", "auto")
    elif method in {"full_ft", "lora"}:
        pass
    else:
        raise ValueError(f"Unsupported fine-tuning method '{method}'")

    source = checkpoint_path if (checkpoint_path and method == "full_ft") else model_id
    model = AutoModelForSequenceClassification.from_pretrained(source, **load_kwargs)

    if tokenizer.pad_token_id is not None and model.config.pad_token_id != tokenizer.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if method in {"lora", "qlora"}:
        if checkpoint_path:
            model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        else:
            model = _apply_lora(model, cfg)
    else:
        if not checkpoint_path:
            freeze_layers = int(cfg.get("freeze_layers", 0))
            _freeze_layers(model, freeze_layers)

    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model


__all__ = ["build_model"]
