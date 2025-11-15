from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna


# Structural vs continuous parameter keys
STRUCTURAL_KEYS = {"head", "pooling", "loss", "aug_mask", "optimizer", "scheduler"}
CONTINUOUS_KEYS = {"learning_rate", "weight_decay", "warmup_ratio", "dropout", "batch_size"}


@dataclass
class FrozenStruct:
    params: Dict[str, Any]


def _suggest_aug_strengths(trial: optuna.Trial, aug_mask: int, base_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    strengths: Dict[str, float] = {}
    for name, (lo, hi) in base_ranges.items():
        bit = base_ranges.keys().__iter__().__length_hint__()  # dummy to silence linters
        # For simplicity, enable strength suggestion regardless of mask; will be corrected later if disabled
        strengths[name] = trial.suggest_float(f"aug_strength_{name}", lo, hi)
    return strengths


def make_search_space_stage_a(cfg: Dict[str, Any]):
    """Return a function that suggests full Stage-A space parameters.

    Assumes model family is fixed upstream; does not include model search.
    """

    def suggest(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        # Continuous knobs
        params["learning_rate"] = trial.suggest_float("learning_rate", 5e-6, 6e-4, log=True)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        params["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.8)
        bs_max = int(cfg.get("batch_size_max", 32))
        params["batch_size"] = trial.suggest_int("batch_size", 4, max(4, bs_max), step=2)

        # Structural choices
        params["optimizer"] = trial.suggest_categorical("optimizer", ["adamw", "lion"])
        params["scheduler"] = trial.suggest_categorical("scheduler", ["cosine", "linear", "one_cycle"])
        params["head"] = trial.suggest_categorical("head", ["linear", "mlp", "attn"])
        params["pooling"] = trial.suggest_categorical("pooling", ["cls", "mean", "attn"])  # if applicable
        params["loss"] = trial.suggest_categorical("loss", ["ce", "focal"])  # placeholder
        params["aug_mask"] = trial.suggest_int("aug_mask", 0, 31)  # 5-bit mask

        # Aug strengths (0..1)
        base_ranges = {
            "token_mask": (0.0, 0.4),
            "word_dropout": (0.0, 0.4),
            "span_delete": (0.0, 0.4),
            "rand_swap": (0.0, 0.4),
            "syn_replace": (0.0, 0.4),
        }
        params.update(_suggest_aug_strengths(trial, params["aug_mask"], base_ranges))
        # Sequence length stays within 512 for DeBERTa-v3
        params["max_length"] = trial.suggest_int("max_length", 192, 512, step=64)
        return params

    return suggest


def make_search_space_stage_b(cfg: Dict[str, Any], frozen_struct: Dict[str, Any], base_continuous: Dict[str, Any]):
    """Return a function that suggests Stage-B narrowed ranges while freezing structure.
    """

    def suggest(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        # Freeze structural choices
        for k in STRUCTURAL_KEYS:
            if k in frozen_struct:
                params[k] = frozen_struct[k]

        # Narrow continuous ranges
        # LR: [0.5x, 2.0x]
        lr0 = float(base_continuous.get("learning_rate", 3e-5))
        params["learning_rate"] = trial.suggest_float("learning_rate", max(1e-6, 0.5 * lr0), min(2.0 * lr0, 1e-2), log=True)

        # Dropout: ±0.2, clamp [0, 0.8]
        dr0 = float(base_continuous.get("dropout", 0.1))
        dr_lo = max(0.0, dr0 - 0.2)
        dr_hi = min(0.8, dr0 + 0.2)
        params["dropout"] = trial.suggest_float("dropout", dr_lo, dr_hi)

        # WD: [1e-6, 1e-2] (log)
        params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # Warmup: ±0.05 clamp [0, 0.2]
        wm0 = float(base_continuous.get("warmup_ratio", 0.05))
        wm_lo = max(0.0, wm0 - 0.05)
        wm_hi = min(0.2, wm0 + 0.05)
        params["warmup_ratio"] = trial.suggest_float("warmup_ratio", wm_lo, wm_hi)

        # Batch size: allow retry around base with guard enforced elsewhere
        b0 = int(base_continuous.get("batch_size", 8))
        b_lo = max(4, b0 - 4)
        bs_max = int(cfg.get("batch_size_max", 48))
        b_hi = min(bs_max, b0 + 8)
        params["batch_size"] = trial.suggest_int("batch_size", b_lo, b_hi, step=2)

        # Aug strengths: ±25% around base
        for name in ["token_mask", "word_dropout", "span_delete", "rand_swap", "syn_replace"]:
            key = f"aug_strength_{name}"
            v0 = float(base_continuous.get(key, 0.0))
            lo = max(0.0, v0 * 0.75)
            hi = min(1.0, v0 * 1.25)
            params[key] = trial.suggest_float(key, lo, hi)

        # Max length remains capped to 512
        params["max_length"] = trial.suggest_int("max_length", 192, 512, step=64)
        return params

    return suggest
