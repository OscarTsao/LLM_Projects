"""Search space utilities for two-stage Optuna HPO.

This module provides helper functions to build the Stage A search space and to
derive a narrowed Stage B space from a winning configuration. The concrete
training integration will be added in later phases; for now we expose pure
helpers that other modules can import.

The design intentionally keeps the search space flat (Optuna sampler friendly)
while tracking which keys represent structural choices that should be frozen
between stages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - torch is optional during unit tests
    import torch
except Exception:  # pragma: no cover - keep logic CPU-compatible
    torch = None

# All augmentation methods (11 nlpaug + 9 textattack)
ALL_AUG_METHODS: Tuple[str, ...] = (
    # nlpaug methods
    "nlp_ContextualWordEmbedding",
    "nlp_RandomWord",
    "nlp_Spelling",
    "nlp_Keyboard",
    "nlp_Ocr",
    "nlp_Split",
    "nlp_RandomChar",
    "nlp_ContextualSentence",
    "nlp_Lambada",
    "nlp_CharSwap",
    # textattack methods
    "ta_TextFoolerJin2019",
    "ta_PWWSRen2019",
    "ta_BAEGarg2019",
    "ta_DeepWordBugGao2018",
    "ta_HotFlipEbrahimi2017",
    "ta_IGAWang2019",
    "ta_Kuleshov2017",
    "ta_Alzantot2018",
    "ta_BERTAttack",
)

# Legacy methods (kept for backward compatibility)
SIMPLE_AUG_METHODS: Tuple[str, ...] = (
    "EDA",
    "CharSwap",
    "Embedding",
    "BackTranslation",
    "CLARE",
)

TEXTATTACK_METHODS: Tuple[str, ...] = (
    "TextFoolerJin2019",
    "PWWSRen2019",
    "DeepWordBugGao2018",
    "HotFlipEbrahimi2017",
    "IGAWang2019",
    "Kuleshov2017",
    "BAEGarg2019",
    "Alzantot2018",
    "BERTAttack",
)


# Structural keys remain fixed between Stage A/B.
STRUCTURAL_KEYS: Tuple[str, ...] = (
    "encoder_model_name",
    "encoder_gradient_checkpointing",
    "train_amp_precision",
    "train_batch_size",
    "eval_batch_size",
    "resources_num_workers",
    "train_grad_accum",
    "train_max_length",
    "train_compile",
    "head_ev_pooler",
    "head_ev_attn_dim",
    "head_ev_type",
    "head_ev_layers",
    "head_ev_hidden",
    "head_ev_act",
    "head_ev_norm",
    "head_cr_pooler",
    "head_cr_attn_dim",
    "head_cr_type",
    "head_cr_layers",
    "head_cr_hidden",
    "head_cr_act",
    "head_cr_norm",
    "loss_class_weights",
    # New augmentation keys (two-stage sampling)
    "num_augmentations",
    "aug_compose_mode",
    # Legacy augmentation keys (kept for backward compatibility)
    "aug_simple_mask",
    "aug_simple_compose",
    "aug_ta_mask",
    "aug_cross_family",
)


@dataclass(frozen=True)
class StageSpace:
    """Container for a stage search space and any frozen parameters."""

    frozen: Dict[str, Any]
    search: Dict[str, Dict[str, Any]]


def _enumerate_masks(methods: Sequence[str], max_active: int) -> List[Tuple[str, ...]]:
    """Enumerate allowed method selections respecting the activation budget."""

    selections: List[Tuple[str, ...]] = [tuple()]
    for k in range(1, max_active + 1):
        selections.extend(tuple(sorted(combo)) for combo in combinations(methods, k))
    return sorted(selections)


ALLOWED_SIMPLE_MASKS = _enumerate_masks(SIMPLE_AUG_METHODS, max_active=2)
ALLOWED_TEXTATTACK_MASKS = _enumerate_masks(TEXTATTACK_METHODS, max_active=1)


def encode_mask(mask: Tuple[str, ...]) -> str:
    return "|".join(mask) if mask else "none"


def decode_mask(token: str) -> Tuple[str, ...]:
    if token in {"", "none"}:
        return tuple()
    return tuple(token.split("|"))


ALLOWED_SIMPLE_MASK_CHOICES = [encode_mask(mask) for mask in ALLOWED_SIMPLE_MASKS]
ALLOWED_TEXTATTACK_CHOICES = [encode_mask(mask) for mask in ALLOWED_TEXTATTACK_MASKS]


def _detect_total_gpu_gib() -> float | None:
    """Return total GPU memory (GiB) for the primary device, if available."""

    if torch is None or not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:  # pragma: no cover - defensive guard
        return None
    return float(props.total_memory) / (1024**3)


def auto_train_batch_sizes() -> List[int]:
    """Derive candidate train batch sizes based on available accelerator memory."""

    total_gib = _detect_total_gpu_gib()
    if total_gib is None:
        base = [8, 12, 16, 24]
    elif total_gib >= 80:
        base = [64, 96, 128, 160]
    elif total_gib >= 48:
        base = [48, 64, 96, 128]
    elif total_gib >= 32:
        base = [32, 48, 64, 80]
    elif total_gib >= 24:
        base = [24, 32, 48, 64]
    elif total_gib >= 16:
        base = [16, 24, 32, 40]
    else:
        base = [8, 12, 16, 24]
    return sorted(set(base))


def auto_eval_batch_sizes(train_choices: Sequence[int]) -> List[int]:
    """Derive evaluation batch sizes, allowing slightly larger batches than training."""

    if not train_choices:
        return [8]
    doubled = [choice * 2 for choice in train_choices]
    combined = sorted(set(train_choices) | set(doubled))
    return combined


def auto_worker_counts() -> List[int]:
    """Return candidate DataLoader worker counts (≈50–100% of available CPUs)."""

    cpu_count = os.cpu_count() or 4
    ratios = (0.5, 0.75, 0.9, 1.0)
    counts = {max(1, int(round(cpu_count * ratio))) for ratio in ratios}
    counts.add(max(1, cpu_count - 1))
    return sorted(counts)


def stage_a_search_space(default_model: str = "microsoft/deberta-v3-base") -> Dict[str, Dict[str, Any]]:
    """Return Stage A search space definition in Optuna-compatible format."""

    train_batch_choices = auto_train_batch_sizes()
    eval_batch_choices = auto_eval_batch_sizes(train_batch_choices)
    worker_choices = auto_worker_counts()

    return {
        # Model / tokenizer
        "encoder_model_name": {
            "type": "categorical",
            "choices": [default_model],
        },
        "encoder_gradient_checkpointing": {
            "type": "categorical",
            "choices": [False, True],
        },
        "train_amp_precision": {
            "type": "categorical",
            "choices": ["bf16", "fp16"],
        },
        # Optimizer & LR scheduling
        "optim_optimizer": {
            "type": "categorical",
            "choices": ["adamw", "lion"],
        },
        "opt_lr_enc": {
            "type": "loguniform",
            "low": 5e-6,
            "high": 5e-5,
        },
        "opt_lr_head": {
            "type": "loguniform",
            "low": 5e-5,
            "high": 3e-3,
        },
        "opt_wd": {
            "type": "loguniform",
            "low": 1e-4,
            "high": 2e-1,
        },
        "sched_type": {
            "type": "categorical",
            "choices": ["linear", "cosine", "one_cycle"],
        },
        "sched_warmup_ratio": {
            "type": "float",
            "low": 0.0,
            "high": 0.2,
        },
        # Training budget
        "train_batch_size": {
            "type": "categorical",
            "choices": train_batch_choices,
        },
        "eval_batch_size": {
            "type": "categorical",
            "choices": eval_batch_choices,
        },
        "train_grad_accum": {
            "type": "categorical",
            "choices": [1, 2],
        },
        "train_max_length": {
            "type": "categorical",
            "choices": [256, 384, 512],
        },
        "train_grad_clip": {
            "type": "categorical",
            "choices": [0.0, 0.5, 1.0],
        },
        "train_compile": {
            "type": "categorical",
            "choices": [False, True],
        },
        "resources_num_workers": {
            "type": "categorical",
            "choices": worker_choices,
        },
        # Evidence head
        "head_ev_pooler": {
            "type": "categorical",
            "choices": ["cls", "mean", "max", "attention"],
        },
        "head_ev_attn_dim": {
            "type": "categorical",
            "choices": [128, 256, 384, 512],
        },
        "head_ev_type": {
            "type": "categorical",
            "choices": ["linear", "mlp"],
        },
        "head_ev_layers": {
            "type": "int",
            "low": 1,
            "high": 3,
        },
        "head_ev_hidden": {
            "type": "categorical",
            "choices": [256, 512, 768, 1024],
        },
        "head_ev_act": {
            "type": "categorical",
            "choices": ["gelu", "relu", "silu", "tanh"],
        },
        "head_ev_dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.5,
        },
        "head_ev_norm": {
            "type": "categorical",
            "choices": ["none", "layernorm"],
        },
        # Criteria head mirrors evidence head options
        "head_cr_pooler": {
            "type": "categorical",
            "choices": ["cls", "mean", "max", "attention"],
        },
        "head_cr_attn_dim": {
            "type": "categorical",
            "choices": [128, 256, 384, 512],
        },
        "head_cr_type": {
            "type": "categorical",
            "choices": ["linear", "mlp"],
        },
        "head_cr_layers": {
            "type": "int",
            "low": 1,
            "high": 3,
        },
        "head_cr_hidden": {
            "type": "categorical",
            "choices": [256, 512, 768, 1024],
        },
        "head_cr_act": {
            "type": "categorical",
            "choices": ["gelu", "relu", "silu", "tanh"],
        },
        "head_cr_dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.5,
        },
        "head_cr_norm": {
            "type": "categorical",
            "choices": ["none", "layernorm"],
        },
        # Loss & multi-task weighting
        "loss_ls": {
            "type": "float",
            "low": 0.0,
            "high": 0.2,
        },
        "loss_class_weights": {
            "type": "categorical",
            "choices": ["none", "inverse_freq", "sqrt_inverse"],
        },
        "mtl_weight_evidence": {
            "type": "float",
            "low": 0.3,
            "high": 0.7,
        },
        # Two-stage augmentation sampling
        # Stage 1: Sample number of augmentations (0-28)
        "num_augmentations": {
            "type": "int",
            "low": 0,
            "high": 28,
        },
        # Stage 2: For each augmentation slot, sample which method to use
        **{
            f"aug_method_{i}": {
                "type": "categorical",
                "choices": list(ALL_AUG_METHODS),
            }
            for i in range(28)  # Create 28 slots for method selection
        },
        # Augmentation probability and composition mode
        "aug_prob": {
            "type": "float",
            "low": 0.05,
            "high": 0.30,
        },
        "aug_compose_mode": {
            "type": "categorical",
            "choices": ["sequential", "random_one"],
        },
        # Legacy augmentation parameters (kept for backward compatibility)
        "aug_simple_mask": {
            "type": "categorical",
            "choices": ALLOWED_SIMPLE_MASK_CHOICES,
        },
        "aug_simple_compose": {
            "type": "categorical",
            "choices": ["one_of", "sequence"],
        },
        "aug_ta_mask": {
            "type": "categorical",
            "choices": ALLOWED_TEXTATTACK_CHOICES,
        },
        "aug_cross_family": {
            "type": "categorical",
            "choices": [
                "serial_then_attack",
                "attack_then_serial",
                "attack_only",
                "serial_only",
            ],
        },
        # Legacy augmentation strengths
        **{
            f"aug_simple_{method}_p_token": {
                "type": "float",
                "low": 0.05,
                "high": 0.30,
            }
            for method in SIMPLE_AUG_METHODS
        },
        **{
            f"aug_simple_{method}_tpe": {
                "type": "int",
                "low": 0,
                "high": 2,
            }
            for method in SIMPLE_AUG_METHODS
        },
        "aug_ta_p_token": {
            "type": "float",
            "low": 0.05,
            "high": 0.30,
        },
        "aug_ta_tpe": {
            "type": "int",
            "low": 0,
            "high": 2,
        },
    }


def narrow_stage_b_space(
    best_params: Dict[str, Any],
    base_space: Dict[str, Dict[str, Any]] | None = None,
) -> StageSpace:
    """Return fixed and tunable subsets for Stage B around ``best_params``.

    ``base_space`` defaults to ``stage_a_search_space()`` when omitted.
    The returned ``StageSpace`` includes a frozen mapping (structural keys +
    their values) and a narrowed search space for continuous/tunable keys.
    """

    if base_space is None:
        base_space = stage_a_search_space()

    frozen: Dict[str, Any] = {}
    narrowed: Dict[str, Dict[str, Any]] = {}

    for key, config in base_space.items():
        if key in STRUCTURAL_KEYS:
            frozen[key] = best_params.get(key)
            continue

        value = best_params.get(key)
        if value is None:
            # If the parameter was not part of the winning config yet, keep base definition.
            narrowed[key] = config
            continue

        param_type = config.get("type")
        if param_type in {"loguniform", "float"}:
            low = config.get("low", config.get("min"))
            high = config.get("high", config.get("max"))
            if param_type == "loguniform":
                candidate_low = max(low * 0.5, 3e-6 if "lr" in key else 1e-6)
                candidate_high = min(high * 2.0, 8e-5 if key == "opt_lr_enc" else 5e-3)
                narrowed[key] = {
                    "type": "loguniform",
                    "low": candidate_low,
                    "high": candidate_high,
                }
            else:
                delta = 0.2 if "dropout" in key else 0.05
                candidate_low = max(float(value) - delta, float(config.get("low", 0.0)))
                candidate_high = min(float(value) + delta, float(config.get("high", 1.0)))
                narrowed[key] = {
                    "type": "float",
                    "low": candidate_low,
                    "high": candidate_high,
                }
        elif param_type == "int":
            candidate_low = max(int(value) - 1, int(config.get("low", 0)))
            candidate_high = min(int(value) + 1, int(config.get("high", candidate_low)))
            narrowed[key] = {
                "type": "int",
                "low": candidate_low,
                "high": candidate_high,
            }
        else:
            # Keep categorical choices unchanged for now; later phases may tighten.
            narrowed[key] = config

    return StageSpace(frozen=frozen, search=narrowed)


__all__ = [
    "ALL_AUG_METHODS",
    "SIMPLE_AUG_METHODS",
    "TEXTATTACK_METHODS",
    "STRUCTURAL_KEYS",
    "StageSpace",
    "stage_a_search_space",
    "narrow_stage_b_space",
    "encode_mask",
    "decode_mask",
    "auto_train_batch_sizes",
    "auto_eval_batch_sizes",
    "auto_worker_counts",
]
