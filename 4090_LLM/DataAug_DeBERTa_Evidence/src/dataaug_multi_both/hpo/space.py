from __future__ import annotations

import itertools
import math
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import optuna

from dataaug_multi_both.config import load_project_config

SIMPLE_METHODS = ["EDA", "CharSwap", "Embedding", "BackTranslation", "CheckList", "CLARE"]
ATTACK_METHODS = [
    "TextFoolerJin2019",
    "PWWSRen2019",
    "DeepWordBugGao2018",
    "HotFlipEbrahimi2017",
    "IGAWang2019",
    "Kuleshov2017",
    "CheckList2020",
    "BAEGarg2019",
]

SIMPLE_COMBOS = [()] + [
    combo for r in (1, 2) for combo in itertools.combinations(SIMPLE_METHODS, r)
]
ATTACK_COMBOS = [()] + [(method,) for method in ATTACK_METHODS]


def _combo_label(methods: Sequence[str]) -> str:
    # CRITICAL: Always sort to ensure consistent labels regardless of input order
    return "|".join(sorted(methods)) if methods else "none"


def _combo_from_label(label: str) -> tuple[str, ...]:
    return tuple(label.split("|")) if label and label != "none" else tuple()


SIMPLE_COMBO_LABELS = [_combo_label(combo) for combo in SIMPLE_COMBOS]
ATTACK_COMBO_LABELS = [_combo_label(combo) for combo in ATTACK_COMBOS]


def _suggest_categorical(
    trial: optuna.Trial,
    name: str,
    choices: Sequence[Any],
    frozen: Optional[Mapping[str, Any]],
) -> Any:
    # CRITICAL: Always use the exact same choices list for all trials
    # to avoid Optuna's "dynamic value space" error.
    # Never modify choices based on frozen values or runtime conditions.
    choices_list = list(choices)

    if frozen and name in frozen:
        value = frozen[name]
        # If frozen value is not in choices, that's a bug in the frozen dict
        # But we still need to call suggest_categorical for Optuna tracking
        # We'll just use the value anyway (Optuna will handle it)
        if value not in choices_list:
            # Warn but don't modify choices - use closest valid value or the frozen value
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Frozen value {value} for {name} not in choices {choices_list}. "
                f"Using frozen value anyway."
            )
        # Always call with the same choices list
        trial.suggest_categorical(name, choices_list)
        return value

    return trial.suggest_categorical(name, choices_list)


def _suggest_float(
    trial: optuna.Trial,
    name: str,
    low: float,
    high: float,
    *,
    log: bool = False,
    frozen: Optional[Mapping[str, Any]] = None,
    narrowed: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> float:
    # Always use suggest_float to maintain distribution consistency
    # between Stage A and Stage B
    if frozen and name in frozen:
        value = float(frozen[name])
        # Use suggest_float with a tiny range around the frozen value
        trial.suggest_float(name, value, value, log=log)
        return value

    if narrowed and name in narrowed:
        range_cfg = narrowed[name]
        low = float(range_cfg["low"])
        high = float(range_cfg["high"])
        log = bool(range_cfg.get("log", log))

    # Always use suggest_float, even when low == high
    # to avoid distribution type mismatch
    return trial.suggest_float(name, low, high, log=log)


def _suggest_int(
    trial: optuna.Trial,
    name: str,
    low: int,
    high: int,
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    narrowed: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> int:
    # Always use suggest_int to maintain distribution consistency
    # between Stage A and Stage B
    if frozen and name in frozen:
        value = int(frozen[name])
        # Use suggest_int with the same value for low and high
        trial.suggest_int(name, value, value)
        return value

    if narrowed and name in narrowed:
        range_cfg = narrowed[name]
        low = int(range_cfg["low"])
        high = int(range_cfg["high"])

    # Always use suggest_int, even when low == high
    # to avoid distribution type mismatch
    return trial.suggest_int(name, low, high)


def _update_head_config(
    trial: optuna.Trial,
    cfg: dict[str, Any],
    key_prefix: str,
    frozen: Optional[Mapping[str, Any]],
    narrowed: Optional[Mapping[str, Mapping[str, Any]]],
) -> None:
    head_cfg = cfg["heads"][key_prefix]
    pooler = _suggest_categorical(
        trial,
        f"head.{key_prefix}.pooler",
        ["cls", "mean", "max", "attention"],
        frozen,
    )
    head_cfg["pooler_type"] = pooler
    if pooler == "attention":
        head_cfg["attn_dim"] = _suggest_categorical(
            trial,
            f"head.{key_prefix}.attn_dim",
            [128, 256, 384, 512],
            frozen,
        )
    else:
        head_cfg.pop("attn_dim", None)

    head_type = _suggest_categorical(
        trial,
        f"head.{key_prefix}.type",
        ["linear", "mlp"],
        frozen,
    )
    head_cfg["type"] = head_type
    if head_type == "mlp":
        head_cfg["layers"] = _suggest_int(
            trial,
            f"head.{key_prefix}.layers",
            1,
            3,
            frozen=frozen,
            narrowed=narrowed,
        )
        head_cfg["hidden"] = _suggest_categorical(
            trial,
            f"head.{key_prefix}.hidden",
            [256, 512, 768, 1024],
            frozen,
        )
    else:
        head_cfg.pop("layers", None)
        head_cfg.pop("hidden", None)

    head_cfg["activation"] = _suggest_categorical(
        trial,
        f"head.{key_prefix}.act",
        ["gelu", "relu", "silu", "tanh"],
        frozen,
    )
    head_cfg["dropout"] = _suggest_float(
        trial,
        f"head.{key_prefix}.dropout",
        0.0,
        0.5,
        frozen=frozen,
        narrowed=narrowed,
    )
    head_cfg["norm"] = _suggest_categorical(
        trial,
        f"head.{key_prefix}.norm",
        ["none", "layernorm"],
        frozen,
    )


def _configure_augmentation(
    trial: optuna.Trial,
    cfg: dict[str, Any],
    frozen: Optional[Mapping[str, Any]],
    narrowed: Optional[Mapping[str, Mapping[str, Any]]],
) -> None:
    aug_cfg = cfg["augmentation"]
    simple_cfg = aug_cfg.get("simple", {})
    ta_cfg = aug_cfg.get("ta", {})

    simple_combo_label = _suggest_categorical(
        trial,
        "aug.simple.combo",
        SIMPLE_COMBO_LABELS,
        frozen,
    )
    simple_methods = _combo_from_label(simple_combo_label)
    simple_enabled = {method: False for method in SIMPLE_METHODS}
    for method in simple_methods:
        simple_enabled[method] = True
    simple_cfg["enabled_mask"] = simple_enabled

    # Always use the same choices to avoid Optuna's dynamic value space error
    simple_cfg["compose"] = _suggest_categorical(
        trial,
        "aug.simple.compose",
        ["one_of", "sequence"],
        frozen,
    )

    simple_params = simple_cfg.get("params", {})
    for method in SIMPLE_METHODS:
        params = simple_params.setdefault(method, {})
        if simple_enabled[method]:
            params["p_token"] = _suggest_float(
                trial,
                f"aug.simple.{method}.p_token",
                0.05,
                0.30,
                frozen=frozen,
                narrowed=narrowed,
            )
            params["tpe"] = _suggest_int(
                trial,
                f"aug.simple.{method}.tpe",
                0,
                2,
                frozen=frozen,
                narrowed=narrowed,
            )
        else:
            params.setdefault("p_token", 0.1)
            params.setdefault("tpe", 1)
    simple_cfg["params"] = simple_params

    ta_combo_label = _suggest_categorical(
        trial,
        "aug.ta.combo",
        ATTACK_COMBO_LABELS,
        frozen,
    )
    ta_methods = _combo_from_label(ta_combo_label)
    ta_enabled = {method: False for method in ATTACK_METHODS}
    for method in ta_methods:
        ta_enabled[method] = True
    ta_cfg["enabled_mask"] = ta_enabled

    ta_params = ta_cfg.get("params", {})
    for method in ATTACK_METHODS:
        params = ta_params.setdefault(method, {})
        if ta_enabled[method]:
            params["p_token"] = _suggest_float(
                trial,
                f"aug.ta.{method}.p_token",
                0.05,
                0.30,
                frozen=frozen,
                narrowed=narrowed,
            )
            params["tpe"] = _suggest_int(
                trial,
                f"aug.ta.{method}.tpe",
                0,
                2,
                frozen=frozen,
                narrowed=narrowed,
            )
        else:
            params.setdefault("p_token", 0.1)
            params.setdefault("tpe", 1)
    ta_cfg["params"] = ta_params

    cross_choices = [
        "serial_then_attack",
        "attack_then_serial",
        "attack_only",
        "serial_only",
    ]

    cross_family = _suggest_categorical(
        trial,
        "aug.compose_cross_family",
        cross_choices,
        frozen,
    )
    if not simple_methods and not ta_methods:
        cross_family = "serial_only"
    elif not simple_methods:
        cross_family = "attack_only"
    elif not ta_methods and cross_family != "serial_only":
        cross_family = "serial_only"
    elif simple_methods and ta_methods and cross_family not in cross_choices:
        cross_family = "serial_then_attack"

    aug_cfg["compose_cross_family"] = cross_family
    aug_cfg["simple"] = simple_cfg
    aug_cfg["ta"] = ta_cfg
    cfg["augmentation"] = aug_cfg


def define_search_space(
    trial: optuna.Trial,
    base_config: Mapping[str, Any] | None = None,
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    narrowed: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    cfg = deepcopy(base_config) if base_config is not None else deepcopy(load_project_config())
    frozen = frozen or {}
    narrowed = narrowed or {}

    model_name = cfg["encoder"].get("model_name", "microsoft/deberta-v3-base")
    cfg["encoder"]["model_name"] = model_name
    cfg["encoder"]["tokenizer_name"] = model_name

    cfg["encoder"]["gradient_checkpointing"] = _suggest_categorical(
        trial,
        "encoder.gc",
        [False, True],
        frozen,
    )

    cfg["train"]["amp"] = _suggest_categorical(
        trial,
        "train.amp",
        ["bf16", "fp16"],
        frozen,
    )
    cfg["train"]["per_device_batch_size"] = _suggest_categorical(
        trial,
        "train.bs",
        [8, 16, 32],
        frozen,
    )
    cfg["train"]["grad_accum_steps"] = _suggest_categorical(
        trial,
        "train.ga",
        [1, 2, 4, 8],
        frozen,
    )
    max_len = _suggest_categorical(
        trial,
        "train.max_len",
        [256, 384, 512],
        frozen,
    )
    cfg["train"]["max_length"] = max_len
    cfg["tokenizer"]["max_length"] = min(int(cfg["tokenizer"].get("max_length", max_len)), max_len)
    cfg["train"]["grad_clip_norm"] = _suggest_categorical(
        trial,
        "train.clip",
        [0.0, 0.5, 1.0],
        frozen,
    )
    cfg["train"]["torch_compile"] = _suggest_categorical(
        trial,
        "train.compile",
        [False, True],
        frozen,
    )

    cfg["optim"]["optimizer"] = _suggest_categorical(
        trial,
        "optim.optimizer",
        ["adamw", "lion"],
        frozen,
    )
    cfg["optim"]["lr_encoder"] = _suggest_float(
        trial,
        "opt.lr_enc",
        5e-6,
        5e-5,
        log=True,
        frozen=frozen,
        narrowed=narrowed,
    )
    cfg["optim"]["lr_head"] = _suggest_float(
        trial,
        "opt.lr_head",
        5e-5,
        3e-3,
        log=True,
        frozen=frozen,
        narrowed=narrowed,
    )
    cfg["optim"]["weight_decay"] = _suggest_float(
        trial,
        "opt.wd",
        1e-4,
        2e-1,
        log=True,
        frozen=frozen,
        narrowed=narrowed,
    )

    cfg["sched"]["type"] = _suggest_categorical(
        trial,
        "sched.type",
        ["linear", "cosine", "one_cycle"],
        frozen,
    )
    cfg["sched"]["warmup_ratio"] = _suggest_float(
        trial,
        "sched.warmup_ratio",
        0.0,
        0.2,
        frozen=frozen,
        narrowed=narrowed,
    )

    _update_head_config(trial, cfg, "evidence", frozen, narrowed)
    _update_head_config(trial, cfg, "criteria", frozen, narrowed)

    cfg["loss"]["label_smoothing"] = _suggest_float(
        trial,
        "loss.ls",
        0.0,
        0.2,
        frozen=frozen,
        narrowed=narrowed,
    )
    cfg["loss"]["class_weighting"] = _suggest_categorical(
        trial,
        "loss.cw",
        ["none", "inverse_freq", "sqrt_inverse"],
        frozen,
    )

    cfg["mtl"]["task_weight_evidence"] = _suggest_float(
        trial,
        "mtl.w_ev",
        0.3,
        0.7,
        frozen=frozen,
        narrowed=narrowed,
    )

    _configure_augmentation(trial, cfg, frozen, narrowed)
    aug_cfg = cfg["augmentation"]
    aug_cfg["apply_to"] = cfg["data"]["fields"]["evidence"]
    cfg["augmentation"] = aug_cfg

    return cfg


def extract_structural_params(cfg: Mapping[str, Any]) -> dict[str, Any]:
    structural: dict[str, Any] = {}
    ev_head = cfg["heads"]["evidence"]
    cr_head = cfg["heads"]["criteria"]

    structural["head.ev.pooler"] = ev_head["pooler_type"]
    if ev_head["pooler_type"] == "attention":
        structural["head.ev.attn_dim"] = ev_head["attn_dim"]
    structural["head.ev.type"] = ev_head["type"]
    if ev_head["type"] == "mlp":
        structural["head.ev.layers"] = ev_head["layers"]
        structural["head.ev.hidden"] = ev_head["hidden"]
    structural["head.ev.act"] = ev_head["activation"]
    structural["head.ev.norm"] = ev_head["norm"]

    structural["head.cri.pooler"] = cr_head["pooler_type"]
    if cr_head["pooler_type"] == "attention":
        structural["head.cri.attn_dim"] = cr_head["attn_dim"]
    structural["head.cri.type"] = cr_head["type"]
    if cr_head["type"] == "mlp":
        structural["head.cri.layers"] = cr_head["layers"]
        structural["head.cri.hidden"] = cr_head["hidden"]
    structural["head.cri.act"] = cr_head["activation"]
    structural["head.cri.norm"] = cr_head["norm"]

    structural["loss.cw"] = cfg["loss"]["class_weighting"]

    simple_mask = cfg["augmentation"]["simple"]["enabled_mask"]
    simple_methods = tuple(sorted([name for name, enabled in simple_mask.items() if enabled]))
    structural["aug.simple.combo"] = _combo_label(simple_methods)
    structural["aug.simple.compose"] = cfg["augmentation"]["simple"].get("compose", "one_of")

    ta_mask = cfg["augmentation"]["ta"]["enabled_mask"]
    ta_methods = tuple(sorted([name for name, enabled in ta_mask.items() if enabled]))
    structural["aug.ta.combo"] = _combo_label(ta_methods)
    structural["aug.compose_cross_family"] = cfg["augmentation"].get(
        "compose_cross_family", "serial_only"
    )

    return structural


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_stage_b_narrowing(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}

    lr_enc = float(cfg["optim"]["lr_encoder"])
    ranges["opt.lr_enc"] = {
        "low": _bounded(lr_enc * 0.5, 3e-6, 8e-5),
        "high": _bounded(lr_enc * 2.0, 3e-6, 8e-5),
        "log": True,
    }

    lr_head = float(cfg["optim"]["lr_head"])
    ranges["opt.lr_head"] = {
        "low": _bounded(lr_head * 0.5, 3e-5, 5e-3),
        "high": _bounded(lr_head * 2.0, 3e-5, 5e-3),
        "log": True,
    }

    weight_decay = float(cfg["optim"]["weight_decay"])
    ranges["opt.wd"] = {
        "low": _bounded(weight_decay * 0.5, 1e-6, 1e-2),
        "high": _bounded(weight_decay * 2.0, 1e-6, 1e-2),
        "log": True,
    }

    for prefix in ["ev", "cri"]:
        dropout = float(cfg["heads"]["evidence" if prefix == "ev" else "criteria"]["dropout"])
        ranges[f"head.{prefix}.dropout"] = {
            "low": _bounded(dropout - 0.2, 0.0, 0.5),
            "high": _bounded(dropout + 0.2, 0.0, 0.5),
        }

    warmup = float(cfg["sched"]["warmup_ratio"])
    ranges["sched.warmup_ratio"] = {
        "low": _bounded(warmup - 0.05, 0.0, 0.2),
        "high": _bounded(warmup + 0.05, 0.0, 0.2),
    }

    simple_params = cfg["augmentation"]["simple"]["params"]
    simple_mask = cfg["augmentation"]["simple"]["enabled_mask"]
    for method, enabled in simple_mask.items():
        if not enabled:
            continue
        params = simple_params[method]
        p_token = float(params["p_token"])
        ranges[f"aug.simple.{method}.p_token"] = {
            "low": _bounded(p_token - 0.05, 0.05, 0.30),
            "high": _bounded(p_token + 0.05, 0.05, 0.30),
        }
        tpe = int(params["tpe"])
        ranges[f"aug.simple.{method}.tpe"] = {
            "low": max(0, tpe - 1),
            "high": min(2, tpe + 1),
        }

    ta_params = cfg["augmentation"]["ta"]["params"]
    ta_mask = cfg["augmentation"]["ta"]["enabled_mask"]
    for method, enabled in ta_mask.items():
        if not enabled:
            continue
        params = ta_params[method]
        p_token = float(params["p_token"])
        ranges[f"aug.ta.{method}.p_token"] = {
            "low": _bounded(p_token - 0.05, 0.05, 0.30),
            "high": _bounded(p_token + 0.05, 0.05, 0.30),
        }
        tpe = int(params["tpe"])
        ranges[f"aug.ta.{method}.tpe"] = {
            "low": max(0, tpe - 1),
            "high": min(2, tpe + 1),
        }

    return ranges
