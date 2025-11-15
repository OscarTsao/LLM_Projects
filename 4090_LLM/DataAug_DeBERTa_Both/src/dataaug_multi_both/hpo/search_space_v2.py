from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import optuna
    from optuna.trial import FrozenTrial
    from optuna.trial import Trial
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("Optuna must be installed to use search_space_v2.") from exc

try:  # pragma: no cover - optional dependency
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ALLOWED_EFFECTIVE_BATCH_SIZES = {16, 32, 64, 128}
LOSS_TYPES: Sequence[str] = (
    "bce",
    "weighted_bce",
    "focal",
    "adaptive_focal",
    "hybrid_bce_focal",
    "hybrid_weighted_bce_focal",
    "hybrid_bce_adaptive_focal",
    "hybrid_weighted_bce_adaptive_focal",
)
LR_SCHEDULERS = ("linear", "cosine", "cosine_with_restarts")
AUGMENTATION_METHODS = ("eda", "synonym", "insert", "swap", "char_perturb", "none")
TRUNCATION_SIDES = ("right", "left")
POOLING_STRATEGIES = ("cls", "mean", "max")
OPTIMIZERS = ("adam", "adamw", "sgd")


class Stage(Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"


@dataclass(frozen=True)
class FloatRange:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class IntRange:
    low: int
    high: int
    step: int = 1


@dataclass(frozen=True)
class HardwareCapabilities:
    max_batch_size: int
    amp_dtypes: Tuple[str, ...]


@dataclass
class NarrowedSpace:
    float_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    int_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    categorical_choices: Dict[str, Sequence[Any]] = field(default_factory=dict)
    best_values: Dict[str, Any] = field(default_factory=dict)

    def merge_float(self, name: str, candidate: Tuple[float, float]) -> None:
        self.float_ranges[name] = candidate

    def merge_int(self, name: str, candidate: Tuple[int, int]) -> None:
        self.int_ranges[name] = candidate

    def merge_categorical(self, name: str, candidate: Sequence[Any]) -> None:
        if candidate:
            self.categorical_choices[name] = tuple(dict.fromkeys(candidate))

    def merge_best(self, name: str, value: Any) -> None:
        self.best_values[name] = value


BASE_FLOAT_PARAM_SPECS: Mapping[str, FloatRange] = {
    "learning_rate": FloatRange(5e-6, 3e-3, log=True),
    "weight_decay": FloatRange(1e-6, 1e-1, log=True),
    "adam_beta1": FloatRange(0.80, 0.95),
    "adam_beta2": FloatRange(0.95, 0.999),
    "adam_epsilon": FloatRange(1e-9, 1e-7, log=True),
    "gradient_clip_val": FloatRange(0.0, 1.0),
    "warmup_ratio": FloatRange(0.0, 0.2),
    "layerwise_lr_decay": FloatRange(0.90, 1.0),
    "dropout": FloatRange(0.0, 0.5),
    "label_smoothing": FloatRange(0.0, 0.2),
    "augmentation_prob": FloatRange(0.0, 0.4),
}

CONDITIONAL_FLOAT_PARAM_SPECS: Mapping[str, FloatRange] = {
    "momentum": FloatRange(0.0, 0.95),
    "focal_gamma": FloatRange(1.0, 5.0),
    "focal_alpha": FloatRange(0.0, 0.5),
    "pos_weight": FloatRange(0.5, 4.0),
    "eda_delete_prob": FloatRange(0.0, 0.3),
    "eda_swap_prob": FloatRange(0.0, 0.3),
}

INT_PARAM_SPECS: Mapping[str, IntRange] = {
    "head_hidden_dim": IntRange(256, 768, step=64),
    "augmentation_strength": IntRange(0, 3),
}

CONDITIONAL_INT_PARAM_SPECS: Mapping[str, IntRange] = {
    "cycle_length": IntRange(1, 5),
}

CATEGORICAL_PARAM_SPECS: Mapping[str, Sequence[Any]] = {
    "optimizer": OPTIMIZERS,
    "lr_scheduler_type": LR_SCHEDULERS,
    "pad_to_multiple_of": (None, 8, 16),
    "truncation_side": TRUNCATION_SIDES,
    "freeze_n_layers": (0, 4, 8, 12),
    "pooling_strategy": POOLING_STRATEGIES,
    "reinit_last_n_layers": (0, 1, 2),
    "gradient_checkpointing": (False, True),
    "torch_compile": (False, True),
    "loss_type": LOSS_TYPES,
    "augmentation_method": AUGMENTATION_METHODS,
    "num_workers": (2, 4, 8),
    "pin_memory": (False, True),
    "group_by_length": (False, True),
}


def detect_hardware_capabilities() -> HardwareCapabilities:
    if torch is None:
        return HardwareCapabilities(max_batch_size=32, amp_dtypes=("fp16",))
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        try:
            properties = torch.cuda.get_device_properties(0)
            total_memory_gb = float(properties.total_memory) / (1024**3)
        except Exception:  # pragma: no cover - defensive
            total_memory_gb = 12.0

        if total_memory_gb >= 32:
            max_batch = 128
        elif total_memory_gb >= 24:
            max_batch = 96
        elif total_memory_gb >= 16:
            max_batch = 64
        else:
            max_batch = 48

        amp_choices: List[str] = ["fp16"]
        is_bf16_supported = False
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                is_bf16_supported = bool(torch.cuda.is_bf16_supported())  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive
                is_bf16_supported = False
        elif hasattr(torch.cuda, "get_device_capability"):
            major, _minor = torch.cuda.get_device_capability(0)
            is_bf16_supported = major >= 8
        if is_bf16_supported:
            amp_choices.append("bf16")
        return HardwareCapabilities(max_batch_size=max_batch, amp_dtypes=tuple(amp_choices))

    return HardwareCapabilities(max_batch_size=32, amp_dtypes=("fp16",))


def _apply_float_range(
    name: str,
    spec: FloatRange,
    narrowing: Optional[NarrowedSpace],
) -> Tuple[float, float, bool]:
    low = spec.low
    high = spec.high
    if narrowing and name in narrowing.float_ranges:
        narrowed_low, narrowed_high = narrowing.float_ranges[name]
        low = max(low, narrowed_low)
        high = min(high, narrowed_high)
        if math.isclose(low, high, rel_tol=1e-9):
            low, high = spec.low, spec.high
    if high < low:
        low, high = spec.low, spec.high
    return low, high, spec.log


def _apply_int_range(
    name: str,
    spec: IntRange,
    narrowing: Optional[NarrowedSpace],
) -> Tuple[int, int, int]:
    low = spec.low
    high = spec.high
    if narrowing and name in narrowing.int_ranges:
        narrowed_low, narrowed_high = narrowing.int_ranges[name]
        low = max(low, narrowed_low)
        high = min(high, narrowed_high)
        if low > high:
            low, high = spec.low, spec.high
    return low, high, spec.step


def _apply_categorical_choices(
    name: str,
    base_choices: Sequence[Any],
    narrowing: Optional[NarrowedSpace],
) -> Sequence[Any]:
    narrowed_choices = None
    if narrowing and name in narrowing.categorical_choices:
        narrowed_choices = tuple(narrowing.categorical_choices[name])
    if narrowed_choices:
        valid = [choice for choice in narrowed_choices if choice in base_choices]
        if valid:
            return tuple(dict.fromkeys(valid))
    return tuple(base_choices)


def _allowed_batch_sizes(stage: Stage, hardware: HardwareCapabilities) -> List[int]:
    base_sizes = [8, 16, 32, 48, 64, 96, 128]
    limit = hardware.max_batch_size
    if stage is Stage.STAGE1:
        limit = min(limit, 128)
    return [size for size in base_sizes if size <= limit]


def _allowed_grad_accumulation(stage: Stage) -> List[int]:
    base = [1, 2, 4, 8]
    if stage is Stage.STAGE1:
        return base
    return base


def _resolve_amp_choices(stage: Stage, hardware: HardwareCapabilities, narrowing: Optional[NarrowedSpace]) -> Sequence[str]:
    base_choices = hardware.amp_dtypes or ("fp16",)
    resolved = _apply_categorical_choices("amp_dtype", base_choices, narrowing)
    if not resolved:
        return base_choices
    return resolved


def _build_batch_combinations(
    stage: Stage,
    hardware: HardwareCapabilities,
    narrowing: Optional[NarrowedSpace],
) -> Tuple[List[str], Dict[str, Tuple[int, int, int]]]:
    allowed_batch_sizes = _allowed_batch_sizes(stage, hardware)
    allowed_grad_accums = _allowed_grad_accumulation(stage)
    if narrowing:
        if "batch_size" in narrowing.categorical_choices:
            allowed_batch_sizes = [b for b in allowed_batch_sizes if b in set(narrowing.categorical_choices["batch_size"])]
        if "gradient_accumulation_steps" in narrowing.categorical_choices:
            allowed_grad_accums = [
                g
                for g in allowed_grad_accums
                if g in set(narrowing.categorical_choices["gradient_accumulation_steps"])
            ]
    combos: Dict[str, Tuple[int, int, int]] = {}
    labels: List[str] = []
    for batch_size in allowed_batch_sizes or [16]:
        for grad_accum in allowed_grad_accums or [1]:
            effective = batch_size * grad_accum
            if effective not in ALLOWED_EFFECTIVE_BATCH_SIZES:
                continue
            label = f"bs{batch_size}_ga{grad_accum}_eff{effective}"
            combos[label] = (batch_size, grad_accum, effective)
            labels.append(label)
    if not labels:
        fallback_label = "bs16_ga1_eff16"
        combos[fallback_label] = (16, 1, 16)
        labels.append(fallback_label)
    return labels, combos


def _augment_max_seq_length_choices(
    stage: Stage,
    narrowing: Optional[NarrowedSpace],
) -> Sequence[int]:
    if stage is Stage.STAGE1:
        base = [128, 256, 384]
    else:
        base = [384, 512]
    if narrowing and "max_seq_length" in narrowing.categorical_choices:
        narrowed = [int(value) for value in narrowing.categorical_choices["max_seq_length"]]
    else:
        narrowed = []
    if narrowing and "max_seq_length" in narrowing.best_values:
        narrowed.append(int(narrowing.best_values["max_seq_length"]))
    merged = base + narrowed
    deduplicated = []
    for value in merged:
        if value not in deduplicated:
            deduplicated.append(value)
    return tuple(sorted(deduplicated))


def sample_parameters(
    trial: Trial,
    stage: Stage,
    *,
    narrowing: Optional[NarrowedSpace] = None,
    hardware: Optional[HardwareCapabilities] = None,
) -> Dict[str, Any]:
    hardware = hardware or detect_hardware_capabilities()
    params: Dict[str, Any] = {}

    for name, spec in BASE_FLOAT_PARAM_SPECS.items():
        low, high, is_log = _apply_float_range(name, spec, narrowing)
        params[name] = trial.suggest_float(name, low, high, log=is_log)

    for name, spec in INT_PARAM_SPECS.items():
        low, high, step = _apply_int_range(name, spec, narrowing)
        params[name] = trial.suggest_int(name, low, high, step=step)

    for name, base_choices in CATEGORICAL_PARAM_SPECS.items():
        params[name] = trial.suggest_categorical(name, _apply_categorical_choices(name, base_choices, narrowing))

    amp_choices = _resolve_amp_choices(stage, hardware, narrowing)
    params["amp_dtype"] = trial.suggest_categorical("amp_dtype", amp_choices)

    max_seq_length_choices = _augment_max_seq_length_choices(stage, narrowing)
    params["max_seq_length"] = trial.suggest_categorical("max_seq_length", max_seq_length_choices)

    batch_labels, combo_lookup = _build_batch_combinations(stage, hardware, narrowing)
    selection_label = trial.suggest_categorical("batching_combo", batch_labels)
    batch_size, grad_accum, effective_batch = combo_lookup[selection_label]
    params["batch_size"] = batch_size
    params["gradient_accumulation_steps"] = grad_accum
    params["effective_batch_size"] = effective_batch
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("gradient_accumulation_steps", grad_accum)
    trial.set_user_attr("effective_batch_size", effective_batch)

    if params["optimizer"] == "sgd":
        spec = CONDITIONAL_FLOAT_PARAM_SPECS["momentum"]
        low, high, is_log = _apply_float_range("momentum", spec, narrowing)
        params["momentum"] = trial.suggest_float("momentum", low, high, log=is_log)

    if params["lr_scheduler_type"] == "cosine_with_restarts":
        spec = CONDITIONAL_INT_PARAM_SPECS["cycle_length"]
        low, high, step = _apply_int_range("cycle_length", spec, narrowing)
        params["cycle_length"] = trial.suggest_int("cycle_length", low, high, step=step)

    loss_type = params["loss_type"]
    if "focal" in loss_type:
        gamma_spec = CONDITIONAL_FLOAT_PARAM_SPECS["focal_gamma"]
        low, high, is_log = _apply_float_range("focal_gamma", gamma_spec, narrowing)
        params["focal_gamma"] = trial.suggest_float("focal_gamma", low, high, log=is_log)
        alpha_spec = CONDITIONAL_FLOAT_PARAM_SPECS["focal_alpha"]
        low, high, is_log = _apply_float_range("focal_alpha", alpha_spec, narrowing)
        params["focal_alpha"] = trial.suggest_float("focal_alpha", low, high, log=is_log)
    if "weighted" in loss_type:
        pos_weight_spec = CONDITIONAL_FLOAT_PARAM_SPECS["pos_weight"]
        low, high, is_log = _apply_float_range("pos_weight", pos_weight_spec, narrowing)
        params["pos_weight"] = trial.suggest_float("pos_weight", low, high, log=is_log)

    if params["augmentation_method"] == "eda":
        delete_spec = CONDITIONAL_FLOAT_PARAM_SPECS["eda_delete_prob"]
        low, high, is_log = _apply_float_range("eda_delete_prob", delete_spec, narrowing)
        params["eda_delete_prob"] = trial.suggest_float("eda_delete_prob", low, high, log=is_log)
        swap_spec = CONDITIONAL_FLOAT_PARAM_SPECS["eda_swap_prob"]
        low, high, is_log = _apply_float_range("eda_swap_prob", swap_spec, narrowing)
        params["eda_swap_prob"] = trial.suggest_float("eda_swap_prob", low, high, log=is_log)

    if stage is Stage.STAGE2:
        per_class_choices = _apply_categorical_choices(
            "decision_thresholds_per_class",
            (False, True),
            narrowing,
        )
        params["decision_thresholds_per_class"] = trial.suggest_categorical(
            "decision_thresholds_per_class",
            per_class_choices,
        )

    return params


def _trial_value(trial: FrozenTrial, key: str) -> Any:
    if key in trial.params:
        return trial.params[key]
    return trial.user_attrs.get(key)


def _ensure_min_width(low: float, high: float, base: FloatRange) -> Tuple[float, float]:
    width = high - low
    base_width = base.high - base.low
    min_width = 0.05 * base_width
    if width < min_width:
        mid = (low + high) / 2.0
        low = max(base.low, mid - min_width / 2.0)
        high = min(base.high, mid + min_width / 2.0)
    return low, high


def _quantile_range(values: List[float], base_spec: FloatRange) -> Tuple[float, float]:
    q_low, q_high = np.quantile(values, [0.10, 0.90])
    if not math.isfinite(q_low) or not math.isfinite(q_high):
        return base_spec.low, base_spec.high
    q_low = max(base_spec.low, float(q_low))
    q_high = min(base_spec.high, float(q_high))
    if q_high <= q_low:
        q_low, q_high = base_spec.low, base_spec.high
    return _ensure_min_width(q_low, q_high, base_spec)


def _int_range(values: List[int], base_spec: IntRange) -> Tuple[int, int]:
    low = max(base_spec.low, int(np.min(values)))
    high = min(base_spec.high, int(np.max(values)))
    if low == high:
        if base_spec.step > 1:
            adjusted_high = min(base_spec.high, high + base_spec.step)
            low = max(base_spec.low, adjusted_high - base_spec.step)
            high = adjusted_high
    if low > high:
        low, high = base_spec.low, base_spec.high
    return low, high


def _top_categorical(values: Iterable[Any], limit: int = 2) -> Sequence[Any]:
    counter = Counter(values)
    most_common = [item for item, _count in counter.most_common(limit)]
    return tuple(most_common)


def narrow_stage2_space(
    trials: Sequence[FrozenTrial],
    *,
    top_k: int = 50,
) -> NarrowedSpace:
    completed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        return NarrowedSpace()
    ranked = sorted(completed, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
    top_trials = ranked[:top_k]
    narrowed = NarrowedSpace()

    float_specs: MutableMapping[str, FloatRange] = dict(BASE_FLOAT_PARAM_SPECS)
    float_specs.update(CONDITIONAL_FLOAT_PARAM_SPECS)
    for name, spec in float_specs.items():
        collected = [
            float(value)
            for trial in top_trials
            if (value := _trial_value(trial, name)) is not None
        ]
        if len(collected) >= 2:
            low, high = _quantile_range(collected, spec)
            narrowed.merge_float(name, (low, high))

    int_specs: MutableMapping[str, IntRange] = dict(INT_PARAM_SPECS)
    int_specs.update(CONDITIONAL_INT_PARAM_SPECS)
    for name, spec in int_specs.items():
        collected = [
            int(value)
            for trial in top_trials
            if (value := _trial_value(trial, name)) is not None
        ]
        if collected:
            low, high = _int_range(collected, spec)
            narrowed.merge_int(name, (low, high))

    categorical_keys = set(CATEGORICAL_PARAM_SPECS.keys()) | {
        "amp_dtype",
        "max_seq_length",
        "batch_size",
        "gradient_accumulation_steps",
        "decision_thresholds_per_class",
    }

    for key in categorical_keys:
        values = [
            _trial_value(trial, key)
            for trial in top_trials
            if _trial_value(trial, key) is not None
        ]
        if not values:
            continue
        top_values = _top_categorical(values, limit=2 if key != "amp_dtype" else 2)
        if top_values:
            narrowed.merge_categorical(key, top_values)
        if key == "max_seq_length":
            narrowed.merge_best("max_seq_length", values[0])

    return narrowed


__all__ = [
    "ALLOWED_EFFECTIVE_BATCH_SIZES",
    "Stage",
    "FloatRange",
    "IntRange",
    "HardwareCapabilities",
    "NarrowedSpace",
    "detect_hardware_capabilities",
    "sample_parameters",
    "narrow_stage2_space",
]
